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
    optional = {"episode_seeds", "initial_state_indices", "video_episodes"}
    if (not isinstance(cfg, dict) or not fields <= set(cfg) or set(cfg) - fields - optional
            or type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1):
        raise ValueError("Invalid native LIBERO recipe fields/schema")
    for name, low, high in (("task_id", 0, 9), ("episodes", 1, 20), ("seed", 0, 2**31-1),
                            ("max_steps", 1, 520), ("n_action_steps", 1, 50)):
        if type(cfg[name]) is not int or not low <= cfg[name] <= high:
            raise ValueError(f"Invalid {name}")
    if cfg["suite"] not in SUITES or cfg["max_steps"] > SUITES[cfg["suite"]]:
        raise ValueError("Unknown suite or horizon above native benchmark limit")
    if cfg["device"] not in ("cpu", "cuda") or type(cfg["video"]) is not bool:
        raise ValueError("Invalid device/video")
    for key, upper in (("episode_seeds", 2**31-1), ("initial_state_indices", 49)):
        if key in cfg:
            values = cfg[key]
            if (not isinstance(values, list) or len(values) != cfg["episodes"]
                    or any(type(v) is not int or not 0 <= v <= upper for v in values)):
                raise ValueError(f"Invalid {key} schedule")
    if "video_episodes" in cfg and (type(cfg["video_episodes"]) is not int
            or not 0 <= cfg["video_episodes"] <= cfg["episodes"]):
        raise ValueError("Invalid video_episodes")
    if "episode_seeds" not in cfg and cfg["seed"] + cfg["episodes"] > 2**31:
        raise ValueError("Default seed schedule exceeds native seed bounds")
    pin_path = (path.parent / cfg["policy_pin"]).resolve()
    pin = load_pin(pin_path)
    return cfg, pin


def load_pin(path):
    pin = json.loads(Path(path).read_text())
    for key in ("policy", "vlm_and_tokenizer", "assets"):
        if not re.fullmatch(r"[0-9a-f]{40}", pin[key]["revision"]):
            raise ValueError("Floating model/asset revision rejected")
    if pin["policy"]["repo_id"] != "HuggingFaceVLA/smolvla_libero":
        raise ValueError("This runner is qualified only for the LIBERO checkpoint")
    expected = pin["checkpoint_features"]
    if (expected["state_shape"] != [8] or expected["action_shape"] != [7]
            or expected["chunk_size"] != 50):
        raise ValueError("Wrong native checkpoint contract")
    return pin
