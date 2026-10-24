from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest

from contracts.clock import ClockOffsetEstimate
from contracts.droid import DroidIOSpec
from contracts.droid_session import capability_id, chunk_envelope
from contracts.franka_state import FrankaStateSample
from contracts.sessions import CommandReceipt, FaultReport, SessionLease
from learning.integrations.openpi.droid import ChunkRejected, bounded_chunk
from learning.rollouts.droid_runner import DroidEpisodeRecorder, DroidRunner

pytest.importorskip("openpi_client")
ROOT = Path(__file__).resolve().parents[3]
SPEC = DroidIOSpec.from_json((ROOT / "configs/contracts/droid_fr3_jointpos_v1.json").read_text())
Q = (0., 0.27, 0., -2.03, 0., 2.35, 0.08)
OFFSET = 5_000_000_000
EXT, WRIST = "observation/exterior_image_1_left", "observation/wrist_image_left"


class Clock:
    def __init__(self):
        self.now = 10_000_000_000

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(1, int(seconds*1e9))


class FakeRuntime:
    """Applies accepted chunks immediately; optional scripted rejections and fault."""

    def __init__(self, clock, reject=(), fault_after=None):
        self.clock, self.reject, self.fault_after = clock, list(reject), fault_after
        self.estimate = ClockOffsetEstimate("c", "rt", OFFSET, 100_000, 8, 0)
        self.lease = SessionLease("fr3_01", "s", "e", "runner", "l", "rt", 0, 10**15)
        self.next_sequence, self.pending, self.fault, self.released, self.submits = 1, [], None, None, 0

    def to_client(self, server_ns):
        return server_ns-OFFSET

    def state(self):
        return FrankaStateSample("fr3_01", "rt", self.clock()+OFFSET-1_000_000, 1, Q, (0.,)*7, 0.08, False, False,
                                 self.fault is None, "l")

    def submit(self, chunk):
        self.submits += 1
        envelope = chunk_envelope(self.lease, self.next_sequence, chunk, self.estimate, client_now_ns=self.clock(),
                                  capability=capability_id(SPEC))
        server_now = self.clock()+OFFSET
        if self.reject and self.reject.pop(0):
            receipt = CommandReceipt("s", "e", "l", envelope.sequence, "rejected", "rt", server_now, None, "too fast")
            self.pending.append(receipt)  # the real authority also queues rejections for drain
        else:
            receipt = CommandReceipt("s", "e", "l", envelope.sequence, "accepted", "rt", server_now)
            self.pending.append(receipt)
            self.pending.append(CommandReceipt("s", "e", "l", envelope.sequence, "applied", "rt", server_now+1, 7))
            self.next_sequence += 1
            if self.fault_after is not None and self.next_sequence > self.fault_after:
                self.fault = FaultReport("s", "e", "rt", server_now, "native_fault:101", "complete")
        return envelope, receipt

    def receipts(self):
        out, self.pending = self.pending, []
        state = "complete" if self.released else ("fault" if self.fault else "running")
        return out, self.fault, {"authority_state": state}

    def release(self, reason):
        self.released, self.released_at = reason, self.clock()


class Router:
    def __init__(self, clock):
        self.clock = clock

    def capture(self):
        frame = SimpleNamespace(image=np.zeros((224, 224, 3), np.uint8), sequence=1, capture_ns=self.clock(),
                                sensor_monotonic_ns=None)
        frames = {EXT: SimpleNamespace(**vars(frame), optical_frame="exterior_optical"),
                  WRIST: SimpleNamespace(**vars(frame), optical_frame="wrist_optical")}
        return frames, {"newest_capture_ns": self.clock(), "capture_skew_ms": 0.}


class Policy:
    def __init__(self, contract_rejections=0):
        self.contract_rejections = contract_rejections

    def infer(self, packet):
        raw = np.array([[*Q, 0.]]*15)
        if self.contract_rejections:
            self.contract_rejections -= 1
            raw[0, 3] += 0.9
        return bounded_chunk(SPEC, packet, raw)


def runner(tmp_path, clock, runtime, policy=None, **changes):
    options = dict(instruction="put the cube in the bowl", max_steps=3, replan_after_rows=6,
                   max_state_age_ns=100_000_000, max_state_camera_skew_ns=100_000_000, max_consecutive_rejections=2)
    options.update(changes)
    return DroidRunner(SPEC, policy or Policy(), runtime, Router(clock), DroidEpisodeRecorder(tmp_path/"ep", clock),
                       clock=clock, sleep=clock.sleep, **options)


def events(tmp_path):
    return [json.loads(line) for line in (tmp_path/"ep/episode.jsonl").read_text().splitlines()]


def test_runner_replans_after_rows_and_releases(tmp_path):
    clock = Clock()
    runtime = FakeRuntime(clock)
    start = clock()
    summary = runner(tmp_path, clock, runtime).run()
    assert summary["outcome"] == "max_steps" and summary["chunks_applied"] == 3 and summary["released"]
    assert runtime.released == "episode max_steps" and runtime.next_sequence == 4
    steps = [e for e in events(tmp_path) if e["kind"] == "step"]
    assert [s["step"] for s in steps] == [0, 1, 2]
    gaps = [b["client_ns"]-a["client_ns"] for a, b in zip(steps, steps[1:])]
    assert all(g >= 6*66_666_667 for g in gaps)  # replanned after six rows, not before
    first = steps[0]
    assert first["state"]["q_rad"] == list(Q) and first["envelope"]["sequence"] == 1
    assert first["observation"]["state"][7] == pytest.approx(0.)  # 0.08 m width = open
    assert json.loads((tmp_path/"ep/summary.json").read_text())["event_counts"]["step"] == 3
    assert clock() > start


def test_rejections_keep_the_sequence_and_stop_after_limit(tmp_path):
    clock = Clock()
    runtime = FakeRuntime(clock, reject=[True, False, True, True, True])
    summary = runner(tmp_path, clock, runtime, max_steps=6).run()
    assert summary["outcome"] == "rejections" and summary["chunks_applied"] == 1
    sequences = [e["envelope"]["sequence"] for e in events(tmp_path) if e["kind"] == "step"]
    assert sequences == [1, 1, 2, 2, 2]  # a rejection never consumes a sequence number


def test_contract_rejection_is_recorded_without_submission(tmp_path):
    clock = Clock()
    runtime = FakeRuntime(clock)
    summary = runner(tmp_path, clock, runtime, Policy(contract_rejections=1), max_steps=2).run()
    steps = [e for e in events(tmp_path) if e["kind"] == "step"]
    assert "deviation" in steps[0]["contract_rejection"] and "chunk" not in steps[0]
    assert runtime.submits == 1 and summary["chunks_applied"] == 1


def test_runtime_fault_ends_episode_without_release(tmp_path):
    clock = Clock()
    runtime = FakeRuntime(clock, fault_after=1)
    summary = runner(tmp_path, clock, runtime, max_steps=5).run()
    assert summary["outcome"] == "runtime_fault" and not summary["released"] and runtime.released is None
    assert summary["fault"]["code"] == "native_fault:101"


def test_stale_state_is_refused(tmp_path):
    clock = Clock()
    runtime = FakeRuntime(clock)
    runtime.state = lambda: FrankaStateSample("fr3_01", "rt", clock()+OFFSET-500_000_000, 1, Q, (0.,)*7, 0.08,
                                              False, False, True, "l")
    with pytest.raises(RuntimeError, match="old"):
        runner(tmp_path, clock, runtime).run()
    assert runtime.released  # still released on the way out


def test_runner_waits_for_first_valid_native_feedback(tmp_path):
    clock = Clock()
    runtime = FakeRuntime(clock)
    valid_at = clock()+300_000_000
    real_state = runtime.state

    def state():
        sample = real_state()
        return sample if clock() >= valid_at else FrankaStateSample(
            "fr3_01", "rt", sample.timestamp_ns, 1, Q, (0.,)*7, 0.08, False, False, False, "l")
    runtime.state = state
    summary = runner(tmp_path, clock, runtime, max_steps=1).run()
    assert summary["chunks_applied"] == 1
    ready = next(e for e in events(tmp_path) if e["kind"] == "runtime_ready")
    assert ready["wait_ms"] >= 300


def test_runner_gives_up_when_feedback_never_becomes_valid(tmp_path):
    clock = Clock()
    runtime = FakeRuntime(clock)
    runtime.state = lambda: FrankaStateSample("fr3_01", "rt", clock()+OFFSET, 1, Q, (0.,)*7, 0.08, False, False,
                                              False, "l")
    summary = runner(tmp_path, clock, runtime, startup_timeout_ns=200_000_000).run()
    assert summary["outcome"] == "runtime_not_ready" and summary["steps"] == 0 and runtime.submits == 0
    assert summary["released"]


def test_episode_summary_reports_timing_rejections_and_candidate_bounds(tmp_path):
    import importlib.util
    clock = Clock()
    runtime = FakeRuntime(clock, reject=[True, False, False])
    runner(tmp_path, clock, runtime, max_steps=3).run()
    spec = importlib.util.spec_from_file_location("summarize", ROOT/"scripts/summarize_droid_episode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = {"max_target_lead_rad": [0.3]*7, "max_consecutive_row_speed_rad_s": [0.435]*7}
    summary = module.summarize(tmp_path/"ep", candidate)
    assert summary["steps"] == 3 and summary["accepted"] == 2 and summary["chunks_applied"] == 2
    assert summary["rejections"] == {"too fast": 1}
    assert summary["accepted_to_applied_ms"]["n"] == 2 and summary["row0_lead_rad"]["max"] == pytest.approx(0.)
    assert summary["candidate_would_reject_accepted"] == {"lead": 0, "row_speed": 0}


def test_shadow_mode_replans_from_acceptance_without_applied_receipts(tmp_path):
    clock = Clock()
    runtime = FakeRuntime(clock)
    real_submit = runtime.submit

    def submit(chunk):  # a shadow runtime admits commands but never confirms native application
        envelope, receipt = real_submit(chunk)
        runtime.pending = [r for r in runtime.pending if r.status != "applied"]
        return envelope, receipt
    runtime.submit = submit
    summary = runner(tmp_path, clock, runtime, shadow=True).run()
    assert summary["shadow"] and summary["chunks_accepted"] == 3 and summary["chunks_applied"] == 0
    assert summary["outcome"] == "max_steps" and summary["released"]
    steps = [e for e in events(tmp_path) if e["kind"] == "step"]
    gaps = [b["client_ns"]-a["client_ns"] for a, b in zip(steps, steps[1:])]
    assert all(6*66_666_667 <= g < 9*66_666_667 for g in gaps)  # paced by acceptance, not stalled to deadlines


def test_fault_during_replan_wait_is_noticed(tmp_path):
    clock = Clock()
    runtime = FakeRuntime(clock)
    real_receipts = runtime.receipts

    def receipts():
        if runtime.next_sequence > 1 and runtime.fault is None:
            runtime.fault = FaultReport("s", "e", "rt", clock()+OFFSET, "native_fault:102", "complete")
        return real_receipts()
    runtime.receipts = receipts
    summary = runner(tmp_path, clock, runtime, max_steps=5).run()
    assert summary["outcome"] == "runtime_fault" and summary["steps"] == 1


def test_release_waits_for_the_last_rows_but_not_past_the_hold(tmp_path):
    # D9 2026-10-24: the runtime reports the last chunk while holding its final row, so waiting for
    # executing_sequence to clear ran past the 200 ms hold and faulted (native 204).
    clock = Clock()
    runtime = FakeRuntime(clock)
    runner(tmp_path, clock, runtime, max_steps=2).run()
    applied = [e for e in events(tmp_path) if e["kind"] == "receipt" and e["receipt"]["status"] == "applied"][-1]
    rows_end = runtime.to_client(applied["receipt"]["timestamp_ns"])+SPEC.open_loop_horizon*66_666_667
    assert rows_end <= runtime.released_at < rows_end+100_000_000
