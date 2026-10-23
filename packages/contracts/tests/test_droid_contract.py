import base64
from dataclasses import replace
from pathlib import Path

import pytest

from contracts.droid import DroidActionChunk, DroidIOSpec, FrankaHandClosureMap
from contracts.episodes import SampleContext
from contracts.policies import PolicyIOSpec
from contracts.samples import ObservationPacket, RGBSample

ROOT = Path(__file__).resolve().parents[3]
HOME = (0., 0.27, 0., -2.03, 0., 2.35, 0.08, 0.)


@pytest.fixture
def spec():
    return DroidIOSpec.from_json((ROOT / "configs/contracts/droid_fr3_jointpos_v1.json").read_text())


@pytest.fixture
def context():
    return SampleContext("episode", 3, 1_000_000_000, "monotonic")


def chunk(spec, context, rows, deadline=1_500_000_000):
    return DroidActionChunk(context, spec.action, HOME, rows, deadline)


def test_contract_roundtrip_and_distinct_semantics(spec):
    assert DroidIOSpec.from_json(spec.to_json()) == spec
    assert spec.action.components[-1] == "gripper_closure"
    assert spec.action.joint_unit == "rad_absolute_target"
    assert (spec.chunk_size, spec.open_loop_horizon, spec.action.policy_hz) == (15, 8, 15.)
    with pytest.raises((ValueError, TypeError, KeyError)):
        PolicyIOSpec.from_json(spec.to_json())


@pytest.mark.parametrize("field,value", [("mode", "droid_joint_velocity_gripper"), ("joint_unit", "rad_s"),
                                       ("policy_hz", 20.), ("max_deviation_from_source_rad", 0.)])
def test_reject_relabeled_action_semantics(spec, field, value):
    with pytest.raises(ValueError):
        replace(spec.action, **{field: value})


def test_reject_wrong_robot_horizon_and_observation(spec):
    with pytest.raises(ValueError):
        replace(spec, robot=replace(spec.robot, joint_names=tuple(reversed(spec.robot.joint_names))))
    for horizon in (0, 16):
        with pytest.raises(ValueError):
            replace(spec, open_loop_horizon=horizon)
    with pytest.raises(ValueError):
        replace(spec, chunk_size=10)
    data = spec.to_dict()
    data["observation"]["image_preprocessing"] = "resize"
    with pytest.raises(ValueError):
        DroidIOSpec.from_dict(data)


@pytest.mark.parametrize("row", [HOME[:7], (*HOME[:7], 1.2), (*HOME[:7], -0.1), (float("nan"),)*8, (True,)*8])
def test_bad_rows_rejected(spec, context, row):
    with pytest.raises(ValueError):
        chunk(spec, context, (row,))


def test_absolute_targets_bounded_against_source_state(spec, context):
    near = (*(q+0.5 for q in HOME[:7]), 1.)
    assert chunk(spec, context, (near,)).values == (near,)
    with pytest.raises(ValueError):
        chunk(spec, context, ((*(q+0.7 for q in HOME[:7]), 0.),))
    with pytest.raises(ValueError):  # Velocity-style [-1, 1] rows are far from absolute joint state.
        chunk(spec, context, ((0.,)*8,))
    with pytest.raises(ValueError):
        chunk(spec, context, (HOME,)*16)


def test_chunk_roundtrip_horizon_and_acceptance_window(spec, context):
    rows = tuple((*HOME[:7], i/14) for i in range(15))
    value = chunk(spec, context, rows)
    assert DroidActionChunk.from_json(value.to_json()) == value
    assert len(value.executable_rows(spec)) == 8
    value.validate_for(spec, replace(context, step_index=5, timestamp_ns=1_200_000_000))
    for current in (replace(context, episode_id="other"), replace(context, clock="simulation"),
                    replace(context, step_index=2), replace(context, timestamp_ns=999_999_999),
                    replace(context, timestamp_ns=1_500_000_000)):
        with pytest.raises(ValueError):
            value.validate_for(spec, current)


def test_franka_hand_closure_map(spec):
    hand = spec.gripper
    assert hand.closure_from_width(0.08) == 0. and hand.closure_from_width(0.) == 1.
    assert hand.closure_from_width(0.04) == pytest.approx(0.5)
    assert hand.closure_from_width(0.081) == 0.
    assert (hand.command_from_closure(0.5), hand.command_from_closure(0.51)) == ("open", "close")
    for bad in (-0.001, float("nan")):
        with pytest.raises(ValueError):
            hand.closure_from_width(bad)
    with pytest.raises(ValueError):
        hand.command_from_closure(1.5)
    with pytest.raises(ValueError):
        FrankaHandClosureMap(0.08, close_above=0.4)


def test_observation_packet_carries_droid_state_and_images(spec, context):
    pixels = base64.b64encode(bytes(224*224*3)).decode()
    images = tuple(RGBSample(c.key, c.optical_frame, 224, 224, 0, context, pixels) for c in spec.observation.cameras)
    packet = ObservationPacket(spec.contract_id, context, HOME, images, "put the cube in the bowl")
    packet.validate_for(spec)
    ObservationPacket.from_json(packet.to_json()).validate_for(spec)
    with pytest.raises(ValueError):
        replace(packet, state=(*HOME[:7], 2.)).validate_for(spec)
    with pytest.raises(ValueError):
        replace(packet, images=images[:1]).validate_for(spec)
