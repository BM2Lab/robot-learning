"""V0 synthetic inference; never creates a robot/environment or physical actions."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--pin", type=Path, default=root / "configs/policies/smolvla_base.json")
    parser.add_argument("--contract", type=Path, default=root / "configs/contracts/fr3_visual_reach_v1.json")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--download", action="store_true", help="Allow exact pinned downloads to the model cache")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cache = args.cache_dir
    if cache is None:
        if not os.environ.get("HF_HOME"):
            parser.error("Configure HF_HOME with direnv or specify --cache-dir; no fallback cache")
        cache = Path(os.environ["HF_HOME"]) / "hub"
    if cache.expanduser().resolve().is_relative_to(root):
        parser.error("Model cache must be outside the source repository")
    from contracts.policies import PolicyIOSpec
    from learning.integrations.lerobot.smolvla import SmolVLAInference
    import numpy as np
    pin = json.loads(args.pin.read_text())
    physical_contract = PolicyIOSpec.from_json(args.contract.read_text())
    features = pin["checkpoint_features"]
    # Explicitly synthetic patterns, not spoofed camera observations from a robot.
    c, h, w = features["camera_shape_chw"]
    rng = np.random.default_rng(args.seed)
    images = {name: rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
              for name in features["cameras"]}
    state = np.zeros(features["state_shape"], dtype=np.float32)
    with contextlib.redirect_stdout(sys.stderr):
        model = SmolVLAInference(pin, cache, device=args.device, download=args.download)
        first, latency = model.predict_native(state, images, "Move the end effector to the red marker.", seed=args.seed)
        model.reset()
        second, second_latency = model.predict_native(state, images, "Move the end effector to the red marker.", seed=args.seed)
    difference = float(np.max(np.abs(first-second)))
    if difference > 1e-5:
        raise RuntimeError(f"Seeded reset replay differs by {difference}")
    result = {"status": "passed", "stage": "V0", "synthetic_inputs": True,
              "policy_revision": pin["policy"]["revision"],
              "companion_revision": pin["vlm_and_tokenizer"]["revision"],
              "device": args.device, "seed": args.seed, "action_shape": list(first.shape),
              "actions_finite": True, "native_action_min": float(first.min()),
              "native_action_max": float(first.max()), "first_inference_wall_s": latency,
              "second_inference_wall_s": second_latency, "reset_replay_max_difference": difference,
              "target_contract": physical_contract.contract_id,
              "target_state_dim": len(physical_contract.observation.joint_names),
              "target_action_dim": len(physical_contract.action.components),
              "target_contract_compatible": False, "robot_actions_applied": 0,
              "note": "6-D checkpoint-native values are NOT 3-D FR3 Cartesian increments; adaptation/fine-tuning required."}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
