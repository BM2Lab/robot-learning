import json
from pathlib import Path
import subprocess
import sys

import pytest

from learning.integrations.libero.config import load_recipe

ROOT = Path(__file__).resolve().parents[3]


def test_native_pin_and_default_horizon():
    cfg, pin = load_recipe(ROOT / "configs/evaluations/smolvla_libero_smoke.yaml")
    assert cfg["n_action_steps"] == 1
    assert pin["checkpoint_features"]["state_shape"] == [8]
    assert pin["checkpoint_features"]["action_shape"] == [7]
    assert "NOT physical meters" in pin["native_semantics"]["action"]
    assert len(pin["policy"]["files_sha256"]) == 6


@pytest.mark.parametrize("field,value", [("episodes", True), ("episodes", 0), ("episodes", 21),
    ("seed", -1), ("task_id", 10), ("suite", "custom"), ("max_steps", 281),
    ("n_action_steps", 0), ("video", "yes"), ("unknown", 1)])
def test_bad_config_rejected(tmp_path, field, value):
    import yaml
    source = ROOT / "configs/evaluations/smolvla_libero_smoke.yaml"
    cfg = yaml.safe_load(source.read_text())
    cfg["policy_pin"] = str(ROOT / "configs/policies/smolvla_libero.json")
    cfg[field] = value
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_recipe(path)


def test_floating_revision_rejected(tmp_path):
    import yaml
    cfg, pin = load_recipe(ROOT / "configs/evaluations/smolvla_libero_smoke.yaml")
    pin["assets"]["revision"] = "main"
    (tmp_path / "pin.json").write_text(json.dumps(pin))
    cfg["policy_pin"] = "pin.json"
    (tmp_path / "recipe.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="Floating"):
        load_recipe(tmp_path / "recipe.yaml")


def test_optional_import_isolation():
    code = ("import sys; import learning.integrations.libero.config; "
            "import learning.integrations.libero.preflight; import learning.evaluation.libero; "
            "import learning.evaluation.libero_suite; "
            "assert not any(x in sys.modules for x in ['torch','mujoco','libero','lerobot','isaaclab'])")
    subprocess.run([sys.executable, "-B", "-c", code], check=True)
