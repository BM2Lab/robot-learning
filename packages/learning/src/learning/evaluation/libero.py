"""Bounded single-task reference using upstream LeRobot's native rollout loop."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time


def summarize_episode(data, *, episode, seed, initial_state, horizon, audit):
    """Keep benchmark success separate from infrastructure completion."""
    success = bool(data["success"].any())
    steps = int(data["action"].shape[1])
    reason = "success" if success else ("time_limit" if steps == horizon else "native_done")
    return {"episode": episode, "seed": seed, "initial_state_index": initial_state,
            "success": success, "policy_steps": steps, "simulated_seconds": steps / 20,
            "termination_reason": reason, "reward_sum": float(data["reward"].sum()), **audit}


def _git_receipt():
    root = Path(__file__).resolve().parents[5]
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    source_files = [*root.glob("packages/learning/src/learning/**/*.py"),
                    *root.glob("configs/policies/*.json"), *root.glob("configs/evaluations/*"),
                    *root.glob("scripts/*libero*.py"), *root.glob("requirements/libero*")]
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "diff_sha256": sha256((git("diff", "HEAD") or "").encode()).hexdigest(),
            "files_sha256": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                             for p in sorted(source_files)},
            "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}


def run_evaluation(recipe_path, output_dir, runtime_dir, *, download=False, smoke=False, verify_only=False):
    from learning.integrations.libero.config import load_recipe
    from learning.integrations.libero.preflight import check_runtime, prepare_assets, asset_receipt
    from learning.data.storage import validate_artifact_path
    cfg, pin = load_recipe(recipe_path)
    if smoke and cfg["episodes"] > 2:
        raise ValueError("/tmp smoke is bounded to at most two episodes")
    output = validate_artifact_path(output_dir, 256 * 1024**2, smoke=smoke)
    runtime, versions = check_runtime(pin, runtime_dir, download=download)
    output.mkdir(parents=True)
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "environment_only" if verify_only else "pretrained_policy_rollout",
                "runtime": versions, "python": sys.version, "recipe": cfg, "model_pin": pin,
                "source": _git_receipt(), "backend": "native_libero_mujoco_not_isaac",
                "model_actions_applied": 0}
    def write_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    write_manifest()
    (output / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
    env = None
    writer = None
    try:
        assets = prepare_assets(pin, runtime, download=download)
        manifest["assets"] = asset_receipt(assets)
        import numpy as np
        import torch
        import gymnasium as gym
        from importlib.metadata import distributions
        manifest["installed_packages"] = {d.metadata["Name"]: d.version for d in distributions()}
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.utils.random_utils import set_seed
        from learning.integrations.libero.environment import (
            make_environment, processed_observation, validate_observation, environment_receipt,
            bounded_native_action, select_initial_state)
        env_cfg, base = make_environment(cfg)

        class AuditedEnvironment(gym.vector.VectorWrapper):
            """Explicit native Box clipping, observation checks, latency/step counters."""
            def call(self, name, *args, **kwargs):
                return self.env.call(name, *args, **kwargs)

            def get_attr(self, name):
                return self.env.get_attr(name)

            def reset(self, **kwargs):
                self.rows = self.clipped = 0
                self.peak = 0.0
                self.first_observation = None
                obs, info = self.env.reset(**kwargs)
                self.first_observation = obs
                return obs, info

            def step(self, action):
                action = np.asarray(action)
                bounded, clipped = bounded_native_action(action)
                self.rows += 1
                self.clipped += clipped
                self.peak = max(self.peak, float(np.abs(action).max()))
                transition = self.env.step(bounded)
                if not verify_only:
                    manifest["model_actions_applied"] += 1
                return transition

        env = AuditedEnvironment(base)
        env_pre, env_post = make_env_pre_post_processors(env_cfg, policy_cfg=None)
        if verify_only:
            raw, _ = env.reset(seed=[cfg["seed"]])
            obs = processed_observation(env, raw, env_pre)
            validate_observation(obs)
            raw_start = raw["robot_state"]["joints"]["pos"].copy()
            for _ in range(5):
                raw, _, _, _, _ = env.step(np.array([[0., 0., 0., 0., 0., 0., -1.]], dtype=np.float32))
                validate_observation(processed_observation(env, raw, env_pre))
            from PIL import Image
            for key in ("observation.images.image", "observation.images.image2"):
                image = obs[key][0].permute(1, 2, 0).mul(255).round().byte().numpy()
                Image.fromarray(image).save(output / (key.rsplit(".", 1)[1] + ".png"))
            manifest["environment"] = environment_receipt(env, cfg)
            summary = {"completed": True, "model_actions_applied": 0, "neutral_steps": 5,
                       "state_shape": list(obs["observation.state"].shape), "camera_checks": True,
                       "max_neutral_joint_change_rad": float(np.abs(raw["robot_state"]["joints"]["pos"] - raw_start).max())}
        else:
            from learning.integrations.lerobot.smolvla import SmolVLAInference
            from lerobot.scripts.lerobot_eval import rollout
            model = SmolVLAInference(pin, runtime / "huggingface/hub", device=cfg["device"], download=download)
            if model.policy.config.n_action_steps != pin["checkpoint_features"]["n_action_steps"]:
                raise RuntimeError("Checkpoint execution horizon differs from manifest")
            model.policy.config.n_action_steps = cfg["n_action_steps"]
            model.reset()
            manifest["execution_horizon_override"] = cfg["n_action_steps"] != 1
            manifest["policy_snapshot"] = str(model.snapshots.policy)
            latencies = []
            original = model.policy.select_action
            original_chunk = model.policy._get_action_chunk

            def checked_chunk(*args, **kwargs):
                chunk = original_chunk(*args, **kwargs)
                if tuple(chunk.shape) != (1, 50, 7) or not torch.isfinite(chunk).all():
                    raise RuntimeError("Invalid full action chunk")
                manifest["inference_check"] = {"chunk_shape": list(chunk.shape), "finite": True}
                return chunk

            model.policy._get_action_chunk = checked_chunk

            def select_action(batch, **kwargs):
                validate_observation(batch)
                if cfg["device"] == "cuda":
                    torch.cuda.synchronize()
                begin = time.perf_counter()
                result = original(batch, **kwargs)
                if cfg["device"] == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - begin)
                if result.shape != (1, 7) or not torch.isfinite(result).all():
                    raise RuntimeError("Non-finite or malformed normalized action")
                return result

            # Instance hook, not a fork or weight change. Upstream rollout still owns execution.
            model.policy.select_action = select_action
            episodes = []
            for episode in range(cfg["episodes"]):
                seed = cfg.get("episode_seeds", list(range(cfg["seed"], cfg["seed"] + cfg["episodes"])))[episode]
                set_seed(seed)
                model.reset()
                env_pre.reset()
                env_post.reset()
                native = base.envs[0].unwrapped
                requested_index = cfg.get("initial_state_indices", [None] * cfg["episodes"])[episode]
                initial_index, initial_hash = select_initial_state(native, requested_index)
                start_latency = len(latencies)
                save_video = cfg["video"] and episode < cfg.get("video_episodes", cfg["episodes"])
                if save_video:
                    import imageio.v2 as imageio
                    video_dir = output / "videos"
                    video_dir.mkdir(exist_ok=True)
                    writer = imageio.get_writer(video_dir / f"episode_{episode:03d}.mp4", fps=20)

                def render_callback(vector):
                    image = vector.unwrapped.envs[0].render()
                    if writer is not None:
                        writer.append_data(image)

                started = time.perf_counter()
                data = rollout(env, model.policy, env_pre, env_post, model.pre, model.post,
                               seeds=[seed], render_callback=render_callback if save_video else None)
                validate_observation(processed_observation(env, env.first_observation, env_pre))
                if writer is not None:
                    writer.close()
                    writer = None
                measured = latencies[start_latency:]
                audit = {"action_components_clipped": env.clipped, "peak_raw_controller_input": env.peak,
                         "initial_state_sha256": initial_hash,
                         "video": f"videos/episode_{episode:03d}.mp4" if save_video else None,
                         "action_clip_fraction": env.clipped / (7 * env.rows),
                         "wall_seconds": time.perf_counter() - started,
                         "mean_policy_call_seconds": float(np.mean(measured)),
                         "p95_policy_call_seconds": float(np.percentile(measured, 95))}
                row = summarize_episode(data, episode=episode, seed=seed, initial_state=initial_index,
                                        horizon=cfg["max_steps"], audit=audit)
                episodes.append(row)
                manifest["environment"] = environment_receipt(env, cfg)
                (output / "episodes.jsonl").write_text("".join(json.dumps(e, allow_nan=False) + "\n" for e in episodes))
                write_manifest()
                print(f"Episode {episode}: success={row['success']}, steps={row['policy_steps']}", flush=True)
            summary = {"completed": True, "episodes": len(episodes),
                       "success_rate": sum(e["success"] for e in episodes) / len(episodes),
                       "mean_policy_steps": sum(e["policy_steps"] for e in episodes) / len(episodes),
                       "model_actions_applied": manifest["model_actions_applied"],
                       "benchmark_claim": "bounded smoke only; not a full published benchmark"}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        manifest["status"] = "complete"
        return summary
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if writer is not None:
                writer.close()
            if env is not None:
                env.close()
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_manifest()


def main():
    """Also available as python -m learning.evaluation.libero for suite child processes."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--download", action="store_true", help="Allow pinned downloads")
    parser.add_argument("--smoke", action="store_true", help="At most two episodes; /tmp only")
    args = parser.parse_args()
    print(json.dumps(run_evaluation(args.config, args.output_dir, args.runtime_dir,
                                  download=args.download, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
