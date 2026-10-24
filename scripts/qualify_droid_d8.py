"""D8 commissioning stages against a physical franka-runtime DROID service (no policy, no cameras).

Each stage is one runtime session, so restart the single-session runtime between stages:

- ``connect``: open and register the clock, then release before start; native control never runs.
- ``hold``: hold the measured start pose for ``--duration-s``, then release (stop from hold).
- ``hand``: hold the arm, close the Hand for ``--duration-s``, then open it again.
- ``wrist``: move one joint by ``A(1-cos(2πt/T))/2`` for ``--cycles`` periods. With
  ``--stop-mid-motion`` the session is released a quarter period into the next cycle, at peak speed.

Chunks are absolute DROID targets built from the measured start pose, submitted every
``--replan-after-rows`` rows. Every runtime response is recorded; the runtime keeps all
authority (guards, stop, faults). Pass criteria are evaluated in ``summary.json``.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import uuid

from contracts.droid import DroidActionChunk, DroidIOSpec
from contracts.episodes import SampleContext
from learning.integrations.franka_runtime.client import DroidRuntimeClient
from learning.rollouts.droid_runner import DroidEpisodeRecorder

HOLD_TOLERANCE_RAD = 0.01
HAND_CLOSED_MAX_M = 0.005
HAND_OPEN_TOLERANCE_M = 0.001


class Stage:
    def __init__(self, spec, runtime, recorder, args):
        self.spec, self.runtime, self.recorder, self.args = spec, runtime, recorder, args
        self.period_ns = round(1e9/spec.action.policy_hz)
        self.episode_id = f"d8-{uuid.uuid4().hex}"
        self.receipts, self.fault, self.states, self.step = {}, None, [], 0
        self.q0 = None

    def poll(self):
        receipts, fault, raw = self.runtime.receipts()
        for r in receipts:
            self.receipts.setdefault(r.sequence, []).append(r.status)
            self.recorder.event("receipt", receipt=r.to_dict())
        if fault is not None and self.fault is None:
            self.fault = fault
            self.recorder.event("fault", fault=fault.to_dict(), authority_state=raw.get("authority_state"))
        return raw

    def sample(self):
        s = self.runtime.state()
        self.states.append(dict(t=time.monotonic_ns(), q=list(s.q_rad), dq=list(s.dq_rad_s), width=s.gripper_width_m,
                                valid=s.valid, executing=s.executing_sequence))
        self.recorder.event("state", state=s.to_dict())
        return s

    def await_valid(self):
        end = time.monotonic_ns()+5_000_000_000
        while time.monotonic_ns() < end:
            self.poll()
            if self.fault is not None:
                return None
            s = self.sample()
            if s.valid:
                return s
            time.sleep(0.01)
        return None

    def submit(self, rows_fn):
        """rows_fn(k) -> 8 values for row k of a chunk whose row 0 starts now."""
        s = self.sample().require_valid()
        stamp = self.runtime.to_client(s.timestamp_ns)
        context = SampleContext(self.episode_id, self.step, stamp, "monotonic")
        self.step += 1
        source = (*s.q_rad, self.spec.gripper.closure_from_width(s.gripper_width_m))
        rows = tuple(tuple(rows_fn(k)) for k in range(self.spec.open_loop_horizon))
        chunk = DroidActionChunk(context, self.spec.action, source, rows,
                                 stamp+round(self.spec.open_loop_horizon*1e9/self.spec.action.policy_hz))
        envelope, receipt = self.runtime.submit(chunk)
        self.recorder.event("submit", chunk=chunk.to_dict(), envelope=envelope.to_dict(), receipt=receipt.to_dict())
        if receipt.status == "accepted":
            self.receipts.setdefault(envelope.sequence, []).append("accepted")
        return receipt.status == "accepted"

    def stream(self, duration_s, rows_at):
        """Submit chunks for duration_s; rows_at(t_s) gives the 8 values at stage time t_s."""
        start = time.monotonic_ns()
        while time.monotonic_ns()-start < duration_s*1e9:
            self.poll()
            if self.fault is not None:
                return False
            now = (time.monotonic_ns()-start)/1e9
            if not self.submit(lambda k: rows_at(now+k*self.period_ns/1e9)):
                return False
            replan = time.monotonic_ns()+self.args.replan_after_rows*self.period_ns
            while time.monotonic_ns() < replan:
                self.poll()
                self.sample()
                time.sleep(0.01)
        return True

    def release(self, reason):
        released_ns = time.monotonic_ns()
        self.recorder.event("release_requested", reason=reason)
        self.runtime.release(reason)
        state, end = None, released_ns+5_000_000_000
        while time.monotonic_ns() < end:
            state = self.poll().get("authority_state")
            self.sample()
            if state == "complete" or self.fault is not None:
                break
            time.sleep(0.01)
        return dict(authority_state=state, release_to_complete_s=(time.monotonic_ns()-released_ns)/1e9)


def run(stage, args, spec, runtime, recorder):
    st = Stage(spec, runtime, recorder, args)
    result: dict = dict(stage=args.stage, episode_id=st.episode_id)
    if args.stage == "connect":
        result.update(st.release("d8 connect-only"))
        result["pass"] = result["authority_state"] == "complete" and st.fault is None
        return result, st
    s = st.await_valid()
    if s is None:
        result.update({"outcome": "not_ready", "fault": None if st.fault is None else st.fault.to_dict(), "pass": False})
        return result, st
    q0 = list(s.q_rad)
    result["q0"] = q0
    hold = lambda t: (*q0, 0.)
    if args.stage == "hold":
        ok = st.stream(args.duration_s, hold)
    elif args.stage == "hand":
        ok = (st.stream(1., hold) and st.stream(args.duration_s, lambda t: (*q0, 1.))
              and st.stream(args.duration_s, hold))
    else:
        j, a, period = args.joint, args.amplitude_rad, args.period_s
        def wave(t):
            q = list(q0)
            q[j] += a*(1-math.cos(2*math.pi*t/period))/2
            return (*q, 0.)
        span = args.cycles*period+(period/4 if args.stop_mid_motion else 0.)
        ok = st.stream(span, wave)
        result.update(joint=j, amplitude_rad=a, period_s=period, peak_speed_rad_s=a*math.pi/period)
    stop_q = list(st.states[-1]["q"]) if st.states else None
    result.update(st.release(f"d8 {args.stage} done"), streamed=ok)
    valid = [x for x in st.states if x["valid"]]
    result["chunks"] = {k: v for k, v in sorted(st.receipts.items())}
    result["chunks_applied"] = sum("applied" in v for v in st.receipts.values())
    result["chunks_accepted"] = sum("accepted" in v for v in st.receipts.values())
    result["fault"] = None if st.fault is None else st.fault.to_dict()
    result["max_abs_dq_rad_s"] = [max(abs(x["dq"][i]) for x in valid) for i in range(7)] if valid else None
    if args.stage in ("hold", "hand"):
        result["max_hold_deviation_rad"] = [max(abs(x["q"][i]-q0[i]) for x in valid) for i in range(7)]
    if args.stage == "hand":
        widths = [x["width"] for x in valid]
        result.update(min_width_m=min(widths), final_width_m=widths[-1])
    if args.stage == "wrist" and stop_q is not None:
        result["q_at_release"] = stop_q
        result["q_final"] = valid[-1]["q"]
    passed = ok and st.fault is None and result["authority_state"] == "complete" and \
        result["chunks_applied"] == result["chunks_accepted"] > 0
    if args.stage in ("hold", "hand"):
        passed = passed and max(result["max_hold_deviation_rad"]) <= HOLD_TOLERANCE_RAD
    if args.stage == "hand":
        passed = passed and result["min_width_m"] <= HAND_CLOSED_MAX_M and \
            abs(result["final_width_m"]-spec.gripper.open_width_m) <= HAND_OPEN_TOLERANCE_M
    result["pass"] = passed
    return result, st


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("connect", "hold", "hand", "wrist"), required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--runtime-host", required=True)
    p.add_argument("--runtime-port", type=int, required=True)
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--duration-s", type=float, default=10.)
    p.add_argument("--replan-after-rows", type=int, default=5)
    p.add_argument("--joint", type=int, default=6, help="0-based joint index for --stage wrist")
    p.add_argument("--amplitude-rad", type=float, default=0.2)
    p.add_argument("--period-s", type=float, default=4.)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--stop-mid-motion", action="store_true")
    p.add_argument("--max-uncertainty-us", type=int, default=2000)
    p.add_argument("--offline-mock", action="store_true", help="rehearse against the D4 mock runtime (not D8 evidence)")
    args = p.parse_args()
    if not 0 <= args.joint <= 6 or not 0 < args.amplitude_rad <= 0.3 or args.period_s < 1.5 \
            or args.amplitude_rad*math.pi/args.period_s > 0.55 or not 0 < args.duration_s <= 60:
        p.error("D8 bounds: joint 0..6, amplitude <= 0.3 rad, period >= 1.5 s, peak speed <= 0.55 rad/s, duration <= 60 s")
    from run_droid_runtime import read_token, source_revision
    if args.output_dir.exists():
        p.error(f"{args.output_dir} exists; use a new directory per stage")
    spec = DroidIOSpec.from_json(args.contract.read_text())
    recorder = DroidEpisodeRecorder(args.output_dir)
    runtime = DroidRuntimeClient(args.runtime_host, args.runtime_port, spec, token=read_token(args.token_file),
                                 client_clock_id=f"{os.uname().nodename}:d8:CLOCK_MONOTONIC",
                                 owner_id=f"droid-d8:{os.getpid()}", lease_duration_ns=120*10**9,
                                 max_uncertainty_ns=args.max_uncertainty_us*1000)
    try:
        ready = runtime.open(f"d8-{args.stage}-{time.time_ns()}", start=args.stage != "connect")
        backend = str(runtime.hello.get("versions", {}).get("backend", ""))
        import learning
        recorder.event("session", stage=args.stage, args={k: str(v) for k, v in vars(args).items() if k != "token_file"},
                       hello=runtime.hello, ready=ready.to_dict(), clock=runtime.estimate.to_dict(),
                       sources={"robot-learning": source_revision(learning.__file__)})
        if backend.startswith("shadow") or ("mock" in backend) != args.offline_mock:
            runtime.release("D8 needs the physical backend")
            raise SystemExit(f"Runtime backend {backend!r} does not match --offline-mock={args.offline_mock}; refusing")
        try:
            result, _ = run(args.stage, args, spec, runtime, recorder)
        except BaseException as error:
            try:
                runtime.release(f"d8 client error: {type(error).__name__}")
            finally:
                raise
    finally:
        runtime.close()
    recorder.finish(result)
    (args.output_dir/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
