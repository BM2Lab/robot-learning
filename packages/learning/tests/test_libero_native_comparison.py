"""Comparison selection is checked without importing ML or a simulator."""
from pathlib import Path
import runpy

import pytest


validate = runpy.run_path(str(Path(__file__).resolve().parents[3]/
    "scripts/compare_libero_native.py"))["validate_baseline"]


def baseline():
    return {"complete": True, "pin": {"revision": "fixed"}, "seed": 42,
            "noise_seed_schedule": "seed + policy_step", "execute_steps_per_inference": 1,
            "steps": 280, "endpoint": {"receipt": {"visual_reference_sha256": "reference"}}}


def test_matching_baseline():
    data = baseline()
    assert validate(data, data["pin"], "reference") == data["endpoint"]["receipt"]


@pytest.mark.parametrize("key,value", [("complete", False), ("seed", 43),
    ("steps", 279), ("execute_steps_per_inference", 50), ("noise_seed_schedule", "constant")])
def test_unmatched_schedule_rejected(key, value):
    data = baseline()
    data[key] = value
    with pytest.raises(ValueError):
        validate(data, data["pin"], "reference")


def test_changed_model_or_reference_rejected():
    with pytest.raises(ValueError):
        validate(baseline(), {"revision": "other"}, "reference")
    with pytest.raises(ValueError):
        validate(baseline(), baseline()["pin"], "different reference")
