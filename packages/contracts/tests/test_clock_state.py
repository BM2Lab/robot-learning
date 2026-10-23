from dataclasses import replace

import pytest

from contracts.clock import ClockOffsetEstimate, ClockProbe
from contracts.franka_state import FrankaStateSample

Q = (0., 0.27, 0., -2.03, 0., 2.35, 0.08)


def probe(send, rtt, offset, server_hold=100):
    """Symmetric one-way delay of (rtt)/2; server clock = client clock + offset."""
    receive = send+rtt//2+offset
    return ClockProbe("client:mono", "rt:mono", send, receive, receive+server_hold, send+rtt+server_hold)


def test_probe_offset_and_round_trip():
    p = probe(1_000_000, 400_000, 5_000_000_000)
    assert p.round_trip_ns == 400_000 and p.offset_ns == 5_000_000_000
    with pytest.raises(ValueError):
        ClockProbe("c", "s", 10, 0, 5, 9)  # server time runs backwards
    with pytest.raises(ValueError):
        ClockProbe("c", "s", 10, 0, 100, 20)  # server holds longer than the round trip


def test_estimate_prefers_minimum_round_trip_and_is_conservative():
    probes = [probe(0, 3_000_000, 7_000_000_000+900_000), probe(10_000_000, 200_000, 7_000_000_000),
              probe(20_000_000, 2_000_000, 7_000_000_000-500_000)]
    estimate = ClockOffsetEstimate.from_probes(probes, max_uncertainty_ns=1_000_000)
    assert estimate.offset_ns == 7_000_000_000 and estimate.uncertainty_ns == 100_000 and estimate.samples == 3
    assert estimate.to_server(1_000) == 7_000_001_000
    assert estimate.deadline_to_server(1_000) == 7_000_001_000-100_000
    assert ClockOffsetEstimate.from_json(estimate.to_json()) == estimate
    with pytest.raises(ValueError, match="exceeds"):
        ClockOffsetEstimate.from_probes(probes[:1], max_uncertainty_ns=1_000_000)
    with pytest.raises(ValueError, match="mix"):
        ClockOffsetEstimate.from_probes([probes[1], replace(probes[1], server_clock_id="x")], max_uncertainty_ns=10**9)


def sample(**changes):
    fields = dict(robot_id="fr3_01", clock_id="rt:mono", timestamp_ns=5, sequence=3, q_rad=Q, dq_rad_s=(0.,)*7,
                  gripper_width_m=0.079, gripper_moving=False, gripper_grasped=False, valid=True)
    fields.update(changes)
    return FrankaStateSample(**fields)


def test_state_sample_roundtrip_and_validity():
    value = sample(lease_id="l1", executing_sequence=2, executing_row=4)
    assert FrankaStateSample.from_json(value.to_json()) == value
    assert value.require_valid() is value
    with pytest.raises(ValueError, match="Invalid"):
        sample(valid=False, fault_code="reflex").require_valid()


@pytest.mark.parametrize("changes", [{"q_rad": Q[:6]}, {"dq_rad_s": (float("nan"),)*7}, {"gripper_width_m": -0.001},
                                     {"valid": 1}, {"fault_code": "reflex"}, {"executing_sequence": 2},
                                     {"executing_sequence": 2, "executing_row": 0},
                                     {"joint_names": tuple(reversed([f"panda_joint{i}" for i in range(1, 8)]))}])
def test_state_sample_rejects_ambiguity(changes):
    with pytest.raises(ValueError):
        sample(**changes)
