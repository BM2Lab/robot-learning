"""Replay recorded native actions and export sparse measured poses/RGB, no policy."""
import argparse
import json
from pathlib import Path

from learning.data.storage import validate_artifact_path
from learning.integrations.libero.config import load_pin
from learning.integrations.libero.preflight import check_runtime, prepare_assets, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("native-run", "reference", "pin", "runtime", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    pin = load_pin(a.pin)
    baseline = json.loads((a.native_run/"manifest.json").read_text())
    ref = json.loads(a.reference.read_text())
    if not baseline["complete"] or baseline["pin"] != pin or baseline["reference_sha256"] != file_hash(a.reference):
        raise ValueError("Native run/reference mismatch")
    out = validate_artifact_path(a.output, 32*1024**2, smoke=a.smoke)
    runtime, versions = check_runtime(pin, a.runtime)
    prepare_assets(pin, runtime)
    import numpy as np
    from PIL import Image
    from lerobot.envs import make_env_pre_post_processors
    from learning.integrations.libero.environment import make_environment, select_initial_state, processed_observation
    out.mkdir(parents=True)
    cfg, env = make_environment({"suite": "libero_object", "task_id": 0, "max_steps": 280})
    rows = json.loads((a.native_run/"steps.json").read_text())
    targets = {0, 20, 40, 80, 120, 160, 200, len(rows)}
    report = {"complete": False, "reference_sha256": file_hash(a.reference), "frames": [],
              "native_run_sha256": file_hash(a.native_run/"manifest.json"), "runtime": versions,
              "runner_sha256": file_hash(__file__), "pin": pin}
    try:
        native = env.unwrapped.envs[0].unwrapped
        _, digest = select_initial_state(native, 0)
        if digest != ref["fixture"]["sha256"]:
            raise ValueError("Fixture mismatch")
        env.reset(seed=42)
        pre, _ = make_env_pre_post_processors(cfg, policy_cfg=None)
        sim = native._env.sim
        model = sim.model
        names = list(ref["scene"]["joint_qpos"])
        joints = [n for n in names if n.startswith(("robot0_joint", "gripper0_finger_joint"))]
        objects = [n for n in ref["scene"]["body_pose_wxyz"] if n.endswith("_main")]

        def batch(x):
            return {k: batch(v) for k, v in x.items()} if isinstance(x, dict) else np.asarray(x)[None]

        for step in range(len(rows)+1):
            if step in targets:
                # MuJoCo integration can leave Cartesian caches one substep
                # behind qpos. Refresh FK before pairing measured q with RGB.
                sim.forward()
                fresh = native._env.env._get_observations(force_update=True)
                obs = processed_observation(env, batch(native._format_raw_obs(fresh)), pre)
                frame = {"step": step, "state": obs["observation.state"][0].tolist(),
                         "instruction": obs["task"][0], "joints": {}, "objects": {}}
                for n in joints:
                    idx = model.joint_name2id(n)
                    frame["joints"][n] = float(sim.data.qpos[model.jnt_qposadr[idx]])
                for n in objects:
                    idx = model.body_name2id(n)
                    frame["objects"][n] = np.r_[sim.data.body_xpos[idx],sim.data.body_xquat[idx]].tolist()
                for key, short in (("observation.images.image", "agentview"), ("observation.images.image2", "eye_in_hand")):
                    rgb = obs[key][0].permute(1, 2, 0).mul(255).round().byte().numpy()
                    Image.fromarray(rgb).save(out/f"native_{step}_{short}.png")
                report["frames"].append(frame)
            if step < len(rows):
                env.step(np.asarray([rows[step]["applied_action"]], dtype=np.float32))
        report["complete"] = True
        report["native_replay_success"] = bool(native._env.check_success())
    finally:
        env.close()
        report["files_sha256"] = {x.name: file_hash(x) for x in out.glob("*.png")}
        (out/"poses.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Pose sequence: {out}", flush=True)


if __name__ == "__main__":
    main()
