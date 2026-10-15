"""Strict, bounded native benchmark recipe; independent of Isaac contracts."""
import json
from pathlib import Path
import re

SUITES = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def load_recipe(path):
    import yaml
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    fields = {"schema_version", "policy_pin", "suite", "task_id", "episodes", "seed",
              "max_steps", "n_action_steps", "device", "video"}
    if not isinstance(cfg, dict) or set(cfg) != fields or cfg["schema_version"] != 1:
        raise ValueError("Invalid native LIBERO recipe fields/schema")
    for name, low, high in (("task_id", 0, 9), ("episodes", 1, 20), ("seed", 0, 2**31-21),
                            ("max_steps", 1, 520), ("n_action_steps", 1, 50)):
        if type(cfg[name]) is not int or not low <= cfg[name] <= high:
            raise ValueError(f"Invalid {name}")
    if cfg["suite"] not in SUITES or cfg["max_steps"] > SUITES[cfg["suite"]]:
        raise ValueError("Unknown suite or horizon above native benchmark limit")
    if cfg["device"] not in ("cpu", "cuda") or type(cfg["video"]) is not bool:
        raise ValueError("Invalid device/video")
    pin_path = (path.parent / cfg["policy_pin"]).resolve()
    pin = json.loads(pin_path.read_text())
    for key in ("policy", "vlm_and_tokenizer", "assets"):
        if not re.fullmatch(r"[0-9a-f]{40}", pin[key]["revision"]):
            raise ValueError("Floating model/asset revision rejected")
    if pin["policy"]["repo_id"] != "HuggingFaceVLA/smolvla_libero":
        raise ValueError("This runner is qualified only for the LIBERO checkpoint")
    expected = pin["checkpoint_features"]
    if (expected["state_shape"] != [8] or expected["action_shape"] != [7]
            or expected["chunk_size"] != 50):
        raise ValueError("Wrong native checkpoint contract")
    return cfg, pin
