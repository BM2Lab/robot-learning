"""Matched native control for a completed P3 Isaac explicit-noise rollout."""
import argparse
import json
from pathlib import Path
import time
import traceback

from learning.data.storage import validate_artifact_path
from learning.integrations.libero.config import load_pin
from learning.integrations.libero.preflight import check_runtime, prepare_assets, file_hash


def validate_baseline(baseline, pin, reference_hash):
    """Reject an unmatched control before loading models or starting physics."""
    if (not baseline["complete"] or baseline["pin"] != pin
            or baseline["seed"] != 42 or baseline["noise_seed_schedule"] != "seed + policy_step"
            or baseline["execute_steps_per_inference"] != 1 or baseline["steps"] != 280):
        raise ValueError("Expected completed seed-42 P3 fixed-fixture baseline with matching pins")
    endpoint = baseline["endpoint"]["receipt"]
    if reference_hash != endpoint["visual_reference_sha256"]:
        raise ValueError("Reference differs from the Isaac endpoint's reference")
    return endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pin", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    baseline = json.loads((args.isaac_run/"manifest.json").read_text())
    reference = json.loads(args.reference.read_text())
    pin = load_pin(args.pin)
    ref_hash = file_hash(args.reference)
    endpoint = validate_baseline(baseline, pin, ref_hash)
    output = validate_artifact_path(args.output, 64*1024**2, smoke=args.smoke)
    runtime, versions = check_runtime(pin, args.runtime)
    output.mkdir(parents=True, exist_ok=False)
    report = {"complete": False, "task_success": None,
              "backend": "native_libero_mujoco_not_isaac", "seed": 42,
              "noise_seed_schedule": "seed + policy_step", "execute_steps_per_inference": 1,
              "runtime_versions": versions, "pin": pin, "runner_sha256": file_hash(__file__),
              "inference_source_sha256": file_hash(Path(__file__).resolve().parents[1]/
                  "packages/learning/src/learning/integrations/lerobot/smolvla.py"),
              "isaac_manifest_sha256": file_hash(args.isaac_run/"manifest.json"),
              "reference_sha256": ref_hash, "steps": 0}
    env = writer = None
    rows = []
    start = time.perf_counter()
    try:
        prepare_assets(pin, runtime)
        import numpy as np
        import imageio.v2 as imageio
        from lerobot.envs import make_env_pre_post_processors
        from learning.integrations.libero.environment import (
            make_environment, select_initial_state, processed_observation,
            validate_observation, environment_receipt, bounded_native_action)
        from learning.integrations.lerobot.smolvla import SmolVLAInference
        model = SmolVLAInference(pin, runtime/"huggingface/hub", device="cuda")
        model.reset()
        recipe = {"suite": "libero_object", "task_id": 0, "max_steps": 280}
        cfg, env = make_environment(recipe)
        native = env.unwrapped.envs[0].unwrapped
        index, digest = select_initial_state(native, 0)
        if (digest != reference["fixture"]["sha256"] or digest != endpoint["fixture_sha256"]
                or reference["fixture"]["index"] != index or reference["fixture"]["seed"] != 42):
            raise ValueError("Native initial-state fixture differs from the Isaac fixture")
        report.update(initial_state_index=index, initial_state_sha256=digest)
        raw, _ = env.reset(seed=42)
        report["environment"] = environment_receipt(env, recipe)
        if report["environment"] != reference["environment"]:
            raise ValueError("Native task/controller receipt differs from exported reference")
        pre, _ = make_env_pre_post_processors(cfg, policy_cfg=None)
        pre.reset()
        writer = imageio.get_writer(output/"rollout.mp4", fps=20)

        def features_and_frame(raw):
            obs = processed_observation(env, raw, pre)
            validate_observation(obs)
            features = {"state": obs["observation.state"][0].cpu().numpy(),
                        "instruction": obs["task"][0], "images": {}}
            for key in pin["checkpoint_features"]["cameras"]:
                features["images"][key] = obs[key][0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
            writer.append_data(np.concatenate(list(features["images"].values()), axis=1))
            return features

        features = features_and_frame(raw)
        first = np.asarray(features["state"]).copy()
        reset_error = float(np.abs(first-reference["visual"]["state"]).max())
        report["native_cached_reset_state_max_error"] = reset_error
        if reset_error > 2e-5:
            raise ValueError("Native reset does not match the exported fixture observation")
        for step in range(280):
            actions, latency = model.predict_native(**features, seed=42+step)
            action, clipped = bounded_native_action(actions[:, 0])
            previous = features["state"].tolist()
            raw, reward, terminated, truncated, info = env.step(action)
            features = features_and_frame(raw)
            success = bool(np.asarray(info["is_success"])[0])
            done = bool(terminated[0] or truncated[0] or step == 279)
            terminal = {"reason": "success" if success else "horizon" if step == 279
                        else "native_done", "success": success} if done else None
            rows.append({"step_index": step, "noise_seed": 42+step,
                         "source_state": previous, "raw_action": actions[0, 0].tolist(),
                         "applied_action": action[0].tolist(), "clipped_components": clipped,
                         "inference_seconds": latency, "next_state": features["state"].tolist(),
                         "reward": float(reward[0]), "terminal": terminal})
            report["steps"] = len(rows)
            if len(rows) % 20 == 0:
                print(f"Native control: {len(rows)}/280 steps; success={success}", flush=True)
            if done:
                report.update(complete=True, task_success=success, terminal=terminal)
                break
        isaac_rows = json.loads((args.isaac_run/"steps.json").read_text())
        report["comparison"] = {
            "isaac_task_success": baseline["task_success"],
            "isaac_steps": baseline["steps"], "native_steps": len(rows),
            "reset_state_max_abs_difference": float(np.abs(first-isaac_rows[0]["source_state"]).max()),
            "first_raw_action_max_abs_difference": float(np.abs(
                np.asarray(rows[0]["raw_action"])-isaac_rows[0]["raw_action"]).max()),
            "interpretation": "Matched inference/settings, different closed-loop observations and physics; not action replay"}
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if writer is not None:
                writer.close()
            if env is not None:
                env.close()
        finally:
            report["wall_seconds"] = time.perf_counter()-start
            if rows:
                report["mean_inference_seconds"] = sum(r["inference_seconds"] for r in rows)/len(rows)
                report["clipped_components"] = sum(r["clipped_components"] for r in rows)
            (output/"steps.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
            (output/"manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Matched native artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
