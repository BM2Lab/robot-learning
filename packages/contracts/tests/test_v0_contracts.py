from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from contracts.actions import ActionChunk, CartesianDeltaSpec
from contracts.episodes import SampleContext
from contracts.observations import CameraSpec
from contracts.policies import PolicyIOSpec

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def spec():
    return PolicyIOSpec.from_json((ROOT / "configs/contracts/fr3_visual_reach_v1.json").read_text())


def test_contract_json_roundtrip_and_dimensions(spec):
    assert PolicyIOSpec.from_json(spec.to_json()) == spec
    assert len(spec.observation.joint_names) == 7
    assert spec.action.components == ("dx", "dy", "dz")
    assert spec.action.policy_hz == 20
    assert spec.action.orientation == "hold_episode_initial"
    assert spec.observation.validate_state([0.]*7) == (0.,)*7


@pytest.mark.parametrize("field,value", [("unit", "rad"), ("frame", ""), ("reference", "previous_command"),
                                       ("mode", "joint_position"), ("policy_hz", 0),
                                       ("max_translation_norm_m", float("nan"))])
def test_reject_ambiguous_or_wrong_action_semantics(spec, field, value):
    with pytest.raises(ValueError):
        replace(spec.action, **{field: value})


@pytest.mark.parametrize("values", [[0., 0.], [0.]*6, [float("nan"), 0., 0.],
                                    [float("inf"), 0., 0.], [True, 0., 0.], [.006, 0., 0.]])
def test_reject_invalid_physical_action_values(spec, values):
    with pytest.raises(ValueError):
        spec.action.validate_values(values)


def test_robot_frame_joint_order_and_schema_validation(spec):
    with pytest.raises(ValueError):
        replace(spec, action=replace(spec.action, frame="world"))
    with pytest.raises(ValueError):
        replace(spec, observation=replace(spec.observation, joint_names=tuple(reversed(spec.robot.joint_names))))
    with pytest.raises(ValueError):
        replace(spec.robot, joint_names=("joint1", "joint1"))
    with pytest.raises(ValueError):
        replace(spec, schema_version=2)
    with pytest.raises(ValueError):
        spec.observation.validate_state([0.]*6)
    with pytest.raises(ValueError):
        spec.observation.validate_state([float("nan")]*7)


def test_camera_validation_and_duplicate_features(spec):
    with pytest.raises(ValueError):
        CameraSpec("image", "camera", 256, 256, color_space="BGR")
    with pytest.raises(ValueError):
        CameraSpec("image", "", 256, 256)
    with pytest.raises(ValueError):
        replace(spec.observation, cameras=(spec.observation.cameras[0],)*2)
    with pytest.raises(ValueError):
        replace(spec.observation, cameras=())


def test_chunk_roundtrip_deadline_and_episode_validation(spec):
    context = SampleContext("episode_1", 0, 100, "simulation")
    chunk = ActionChunk(context, spec.action, ((.001, 0., 0.),), 200)
    assert ActionChunk.from_json(chunk.to_json()) == chunk
    chunk.validate_for(spec, replace(context, timestamp_ns=199))
    for current in (replace(context, timestamp_ns=200), replace(context, timestamp_ns=99),
                    replace(context, episode_id="episode_2"), replace(context, step_index=1),
                    replace(context, clock="monotonic")):
        with pytest.raises(ValueError):
            chunk.validate_for(spec, current)
    with pytest.raises(ValueError):
        ActionChunk(context, spec.action, (), 200)
    with pytest.raises(ValueError):
        ActionChunk(context, spec.action, ((0., 0., 0.),), 100)
    with pytest.raises(ValueError):
        replace(chunk, values=((0., 0., 0.),)*(spec.chunk_size+1)).validate_for(spec, context)
    with pytest.raises(ValueError):
        replace(chunk, action_spec=replace(spec.action, frame="world")).validate_for(spec, context)


def test_contract_import_isolation():
    code = "import contracts.actions, contracts.observations, contracts.policies, sys; assert not any(n.split('.')[0] in {'torch','numpy','isaaclab','transformers','lerobot','franky'} for n in sys.modules)"
    subprocess.run([sys.executable, "-I", "-B", "-c", code], check=True)
