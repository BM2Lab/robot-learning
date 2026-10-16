import json
import math
import pytest

from contracts.episodes import SampleContext
from contracts.libero import LiberoActionChunk, LiberoActionSpec, LiberoIOSpec
from contracts.policies import PolicyIOSpec


def test_native_contract_roundtrip_and_distinct_semantics():
    spec = LiberoIOSpec()
    assert LiberoIOSpec.from_json(spec.to_json()) == spec
    assert len(spec.observation.components) == 8
    assert "NOT_meters" in spec.action.unit
    with pytest.raises((ValueError, TypeError, KeyError)):
        PolicyIOSpec.from_json(spec.to_json())


@pytest.mark.parametrize("row", [(0.,)*6, (2.,)*7, (float("nan"),)*7, (True,)*7])
def test_bad_native_actions_rejected(row):
    with pytest.raises(ValueError):
        LiberoActionChunk(SampleContext("episode", 0, 0, "simulation"), LiberoActionSpec(), (row,), 100)


def test_chunk_context_expiry_and_roundtrip():
    spec = LiberoIOSpec()
    context = SampleContext("episode", 0, 0, "simulation")
    chunk = LiberoActionChunk(context, spec.action, ((0.,)*7,), 50_000_000)
    assert LiberoActionChunk.from_json(chunk.to_json()) == chunk
    chunk.validate_for(spec, context)
    with pytest.raises(ValueError):
        chunk.validate_for(spec, SampleContext("other", 0, 0, "simulation"))
    with pytest.raises(ValueError):
        chunk.validate_for(spec, SampleContext("episode", 1, 50_000_000, "simulation"))


def test_observation_rejects_joint_state_relabeling():
    spec = LiberoIOSpec()
    with pytest.raises(ValueError):
        spec.observation.validate_state([0.]*7)
    data = spec.to_dict()
    data["observation"]["frame"] = "robot_base"
    with pytest.raises(ValueError):
        LiberoIOSpec.from_dict(data)
