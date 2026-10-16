"""Offline native OSC oracle export. No learned policy or hardware connection."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from .config import load_pin
from .preflight import check_runtime, prepare_assets, file_hash


def export_reference(pin_path, runtime_dir, output):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < 100 * 1024**2:
        raise RuntimeError("Explicit existing output parent with 100 MiB free required")
    pin = load_pin(Path(pin_path))
    runtime, versions = check_runtime(pin, runtime_dir)
    assets = prepare_assets(pin, runtime)
    import numpy as np
    import robosuite.controllers.osc as osc
    import robosuite.utils.control_utils as control_utils
    import robosuite.utils.transform_utils as transforms
    from .environment import make_environment, select_initial_state, environment_receipt

    # The oracle executes upstream set_goal/run_controller unchanged, replacing
    # only acquisition of simulator state for deterministic algebraic test cases.
    class Oracle(osc.OperationalSpaceController):
        def update(self, *args, **kwargs):
            pass

        @property
        def torque_compensation(self):
            return self.export_bias

    recipe = {"suite": "libero_object", "task_id": 0, "max_steps": 280}
    _, env = make_environment(recipe)
    cases = []
    try:
        native = env.unwrapped.envs[0].unwrapped
        index, digest = select_initial_state(native, 0)
        env.reset(seed=42)
        robot = native._env.robots[0]
        controller = robot.controller
        controller.update(force=True)
        receipt = environment_receipt(env, recipe)
        if (not controller.use_delta or controller.impedance_mode != "fixed"
                or controller.interpolator_pos is not None or controller.interpolator_ori is not None
                or not np.all(controller.kp == 150) or not controller.uncoupling
                or controller.control_dim != 6
                or not np.all(controller.input_min == -1) or not np.all(controller.input_max == 1)
                or not np.allclose(controller.output_max, [.05]*3+[.5]*3, atol=0, rtol=0)
                or not np.allclose(controller.output_min, [-.05]*3+[-.5]*3, atol=0, rtol=0)):
            raise RuntimeError("Native OSC profile changed; requalify exporter")
        keys = {"position": "ee_pos", "rotation": "ee_ori_mat", "linear_velocity": "ee_pos_vel",
                "angular_velocity": "ee_ori_vel", "joint_position": "joint_pos", "joint_velocity": "joint_vel",
                "mass": "mass_matrix", "jacobian": "J_full", "bias": "torque_compensation"}
        base = {key: np.array(getattr(controller, attr), dtype=np.float64).copy() for key, attr in keys.items()}
        lower, upper = robot.torque_limits
        anchor = np.array(controller.initial_joint).copy()
        rng = np.random.default_rng(73)
        for label in ("native_zero", "native_delta", "moving", "clip", "zero_rotation_latch", "rank_deficient",
                      "near_singular", "pi_rotation", "coupled"):
            state = {key: value.copy() for key, value in base.items()}
            action = rng.uniform(-.8, .8, 6)
            warm_action = np.array([0., 0., 0., .3, -.2, .1])
            if label == "native_zero":
                action[:] = 0
                warm_action[:] = 0
            if label == "clip":
                action *= 10
            if label == "zero_rotation_latch":
                action[3:] = 0
            if label == "moving":
                state["joint_position"] += rng.normal(0, .1, 7)
                state["joint_velocity"] = rng.normal(0, .1, 7)
                velocity = state["jacobian"] @ state["joint_velocity"]
                state["linear_velocity"], state["angular_velocity"] = velocity[:3], velocity[3:]
            if label in ("rank_deficient", "near_singular"):
                # Exact diagonal fixtures avoid an unspecified singular-vector
                # basis; near-singular retains a nonzero 1e-4 singular value.
                state["mass"] = np.diag(np.arange(1., 8.))
                state["jacobian"] = np.eye(6, 7)
                state["jacobian"][5, 5] = 0 if label == "rank_deficient" else 1e-4
            oracle = object.__new__(Oracle)
            for key, attr in keys.items():
                if key != "bias":
                    setattr(oracle, attr, state[key])
            oracle.export_bias = state["bias"]
            oracle.J_pos, oracle.J_ori = state["jacobian"][:3], state["jacobian"][3:]
            oracle.initial_joint = anchor.copy()
            for name in ("input_min", "input_max", "output_min", "output_max", "kp", "kd"):
                setattr(oracle, name, np.array(getattr(controller, name)).copy())
            oracle.use_delta = oracle.use_ori = True
            oracle.impedance_mode = "fixed"
            oracle.position_limits = oracle.orientation_limits = None
            oracle.interpolator_pos = oracle.interpolator_ori = None
            oracle.action_scale = None
            oracle.uncoupling = label != "coupled"
            oracle.goal_pos = state["position"].copy()
            oracle.goal_ori = state["rotation"].copy()
            # Every case exercises sequential latched goals, including a zero
            # rotation following a nonzero rotation command.
            oracle.set_goal(warm_action)
            oracle.set_goal(action)
            if label == "pi_rotation":
                oracle.goal_ori = np.diag([1., -1., -1.]) @ state["rotation"]
            raw = oracle.run_controller().copy()
            nullspace = control_utils.opspace_matrices(state["mass"], state["jacobian"],
                state["jacobian"][:3], state["jacobian"][3:])[3]
            cases.append({"name": label, "state": state, "initial_joint": anchor,
                          "warm_action": warm_action, "action": action,
                          "goal_override": label == "pi_rotation", "uncoupling": oracle.uncoupling,
                          "goal_position": oracle.goal_pos, "goal_rotation": oracle.goal_ori,
                          "raw_torque": raw, "torque": np.clip(raw, lower, upper), "nullspace": nullspace})
        output.mkdir()
        xml = native._env.sim.model.get_xml()
        (output / "model.xml").write_text(xml)
        model = native._env.sim.model
        metadata = {"schema_version": 1, "purpose": "P0 algebra reference, NOT Isaac physics parity",
                    "runtime_versions": versions, "assets": pin["assets"], "environment": receipt,
                    "fixture": {"seed": 42, "index": index, "sha256": digest,
                                "state": np.asarray(native._init_states[index]).tolist()},
                    "model": {"xml_sha256": file_hash(output / "model.xml"), "asset_root": str(assets),
                              "body_names": list(model.body_names), "joint_names": list(model.joint_names),
                              "body_mass": model.body_mass.tolist(), "body_inertia": model.body_inertia.tolist(),
                              "body_ipos": model.body_ipos.tolist(), "body_iquat_wxyz": model.body_iquat.tolist(),
                              "joint_range": model.jnt_range.tolist(), "timestep": float(model.opt.timestep),
                              "arm_qpos_indices": list(controller.qpos_index),
                              "arm_qvel_indices": list(controller.qvel_index),
                              "initial_joint": anchor.tolist(), "eef_site": robot.gripper.important_sites["grip_site"]},
                    "sources": {Path(module.__file__).name: file_hash(module.__file__)
                                for module in (osc, control_utils, transforms)},
                    "exporter_sha256": file_hash(__file__),
                    "conventions": {"frame": "native world", "rotation": "3x3 columns are axes",
                                    "jacobian": "linear rows then angular; native arm joint order",
                                    "bias": "qfrc_bias (gravity + Coriolis/centrifugal)",
                                    "pinv_rtol": 1e-15, "dtype": "float64; native goal quaternion math float32"},
                    "torque_lower": lower.tolist(), "torque_upper": upper.tolist(), "cases": cases}
        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)
        encoded = json.dumps(metadata, default=encode, indent=2, allow_nan=False)
        (output / "reference.json").write_text(encoded + "\n")
        (output / "receipt.json").write_text(json.dumps({"reference_sha256": sha256((encoded+"\n").encode()).hexdigest()}, indent=2)+"\n")
        return output
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-pin", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_reference(args.policy_pin, args.runtime_dir, args.output))


if __name__ == "__main__":
    main()
