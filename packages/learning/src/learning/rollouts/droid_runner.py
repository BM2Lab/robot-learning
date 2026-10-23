"""DROID runner: measured robot state + cameras → openpi chunk → franka-runtime, recorded.

Synchronous replanning: observe, infer, submit; wait for the runtime's ``applied``
receipt; then observe again after ``replan_after_rows`` rows so the next chunk
replaces the remaining rows before they run out. The runtime keeps all authority:
faults end the episode, and the runner never retries past a latched fault.
Contract rejections are recorded, never clipped. Standard library + contracts +
the learning openpi adapter; no robot or runtime package imports.
"""
import json
from pathlib import Path
import time
import uuid

from contracts.episodes import SampleContext
from ..integrations.openpi.cameras import droid_packet
from ..integrations.openpi.droid import ChunkRejected


class DroidEpisodeRecorder:
    """Append-only JSON-lines record of one episode; images stay inside the typed packets."""

    def __init__(self, directory, clock=time.monotonic_ns):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.clock = clock
        self._stream = open(self.directory/"episode.jsonl", "w")
        self.counts = {}

    def event(self, kind, **data):
        self.counts[kind] = self.counts.get(kind, 0)+1
        self._stream.write(json.dumps({"kind": kind, "client_ns": self.clock(), **data}, allow_nan=False)+"\n")
        self._stream.flush()

    def finish(self, summary):
        self._stream.close()
        (self.directory/"summary.json").write_text(json.dumps({**summary, "event_counts": self.counts}, indent=2)+"\n")


class DroidRunner:
    def __init__(self, spec, policy, runtime, router, recorder, *, instruction, max_steps, replan_after_rows,
                 max_state_age_ns, max_state_camera_skew_ns, max_consecutive_rejections,
                 clock=time.monotonic_ns, sleep=time.sleep):
        if not 1 <= replan_after_rows <= spec.open_loop_horizon:
            raise ValueError("replan_after_rows must be within the open-loop horizon")
        if not isinstance(instruction, str) or not instruction.strip() or max_steps < 1:
            raise ValueError("Instruction and a positive step budget are required")
        self.spec, self.policy, self.runtime, self.router, self.recorder = spec, policy, runtime, router, recorder
        self.instruction, self.max_steps, self.replan_after_rows = instruction, max_steps, replan_after_rows
        self.max_state_age_ns, self.max_skew_ns = max_state_age_ns, max_state_camera_skew_ns
        self.max_rejections, self.clock, self.sleep = max_consecutive_rejections, clock, sleep
        self.period_ns = round(1e9/spec.action.policy_hz)
        self.receipts = {}
        self.fault = None

    def _poll(self):
        receipts, fault, raw = self.runtime.receipts()
        for receipt in receipts:
            self.receipts.setdefault(receipt.sequence, []).append(receipt)
            self.recorder.event("receipt", receipt=receipt.to_dict())
        if fault is not None and self.fault is None:
            self.fault = fault
            self.recorder.event("fault", fault=fault.to_dict(), authority_state=raw.get("authority_state"))
        return raw

    def _observe(self, step, episode_id):
        state = self.runtime.state().require_valid()
        frames, camera = self.router.capture()
        state_ns = self.runtime.to_client(state.timestamp_ns)
        age = self.clock()-state_ns
        if age > self.max_state_age_ns+self.runtime.estimate.uncertainty_ns:
            raise RuntimeError(f"Robot state is {age/1e6:.1f} ms old")
        skew = camera["newest_capture_ns"]-state_ns
        if abs(skew) > self.max_skew_ns:
            raise RuntimeError(f"Robot state and cameras are {skew/1e6:.1f} ms apart")
        context = SampleContext(episode_id, step, camera["newest_capture_ns"], "monotonic")
        packet = droid_packet(self.spec, context, state.q_rad, state.gripper_width_m, frames, self.instruction)
        return packet, state, dict(camera, state_client_ns=state_ns, state_camera_skew_ms=skew/1e6)

    def _await_applied(self, envelope, accepted):
        """Applied receipt for this submission, or None after its rejection, a fault or its deadline.

        Rejected attempts reuse the sequence number. The runtime drains receipts in issue order
        and accepts a sequence at most once, so only receipts after its ``accepted`` one count.
        """
        deadline = self.runtime.to_client(envelope.deadline_ns)+self.period_ns
        while self.clock() < deadline:
            self._poll()
            history = self.receipts.get(envelope.sequence, [])
            if accepted in history:
                for receipt in history[history.index(accepted)+1:]:
                    if receipt.status == "applied":
                        return receipt
                    if receipt.status == "rejected":
                        return None
            if self.fault is not None:
                return None
            self.sleep(0.005)
        return None

    def _finish_rows(self, timeout_ns):
        """Let the last chunk's rows finish, then return as the runtime starts holding."""
        deadline = self.clock()+timeout_ns
        while self.clock() < deadline and self.fault is None:
            if self.runtime.state().executing_sequence is None:
                return
            self._poll()
            self.sleep(0.005)

    def run(self):
        episode_id = f"droid-{uuid.uuid4().hex}"
        self.recorder.event("episode_start", episode_id=episode_id, instruction=self.instruction,
                            replan_after_rows=self.replan_after_rows)
        outcome, rejections, steps, applied_count = "max_steps", 0, 0, 0
        try:
            for step in range(self.max_steps):
                self._poll()
                if self.fault is not None:
                    outcome = "runtime_fault"
                    break
                packet, state, camera = self._observe(step, episode_id)
                steps += 1
                try:
                    chunk, audit = self.policy.infer(packet)
                except ChunkRejected as error:
                    self.recorder.event("step", step=step, observation=packet.to_dict(), state=state.to_dict(),
                                        camera=camera, policy=error.audit, contract_rejection=str(error))
                    rejections += 1
                    if rejections > self.max_rejections:
                        outcome = "rejections"
                        break
                    continue
                envelope, receipt = self.runtime.submit(chunk)
                self.recorder.event("step", step=step, observation=packet.to_dict(), state=state.to_dict(),
                                    camera=camera, policy=audit, chunk=chunk.to_dict(), envelope=envelope.to_dict(),
                                    receipt=receipt.to_dict())
                if receipt.status != "accepted":
                    rejections += 1
                    if rejections > self.max_rejections:
                        outcome = "rejections"
                        break
                    continue
                rejections = 0
                applied = self._await_applied(envelope, receipt)
                if applied is None:
                    continue
                applied_count += 1
                replan_at = self.runtime.to_client(applied.timestamp_ns)+self.replan_after_rows*self.period_ns
                while self.clock() < replan_at and self.fault is None:
                    self.sleep(min(0.005, max(0., (replan_at-self.clock())/1e9)))
            if outcome == "max_steps":
                self._finish_rows((self.spec.open_loop_horizon+1)*self.period_ns)
        except KeyboardInterrupt:
            outcome = "interrupted"
        finally:
            released = False
            self._poll()
            if self.fault is None:
                self.runtime.release(f"episode {outcome}")
                released = True
                deadline = self.clock()+2_000_000_000
                while self.clock() < deadline and self._poll().get("authority_state") != "complete" and self.fault is None:
                    self.sleep(0.01)
            summary = {"episode_id": episode_id, "outcome": outcome, "steps": steps, "chunks_applied": applied_count,
                       "released": released, "fault": None if self.fault is None else self.fault.to_dict(),
                       "receipt_statuses": {str(k): [r.status for r in v] for k, v in self.receipts.items()}}
            self.recorder.event("episode_end", **summary)
            self.recorder.finish(summary)
        return summary
