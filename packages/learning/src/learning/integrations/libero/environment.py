"""Thin native environment factory; upstream owns physics and camera conventions."""


def bounded_native_action(action):
    """Native Box projection, not Cartesian-unit conversion; report every clipped component."""
    import numpy as np
    action = np.asarray(action)
    if action.shape != (1, 7) or not np.isfinite(action).all():
        raise RuntimeError("Invalid native policy action; aborting before physics")
    return np.clip(action, -1, 1), int(np.count_nonzero(np.abs(action) > 1))


def make_environment(recipe):
    from lerobot.envs.configs import LiberoEnv
    cfg = LiberoEnv(task=recipe["suite"], task_ids=[recipe["task_id"]], fps=20,
                    episode_length=recipe["max_steps"], observation_height=256, observation_width=256,
                    control_mode="relative", init_states=True, hard_reset=True)
    env = cfg.create_envs(n_envs=1, use_async_envs=False)[recipe["suite"]][recipe["task_id"]]
    return cfg, env


def processed_observation(env, raw, processor):
    from lerobot.envs import preprocess_observation
    obs = preprocess_observation(raw)
    obs["task"] = list(env.call("task_description"))
    return processor(obs)


def validate_observation(obs):
    import torch
    state = obs["observation.state"]
    if tuple(state.shape) != (1, 8) or not torch.isfinite(state).all():
        raise RuntimeError("Invalid native state")
    for key in ("observation.images.image", "observation.images.image2"):
        image = obs[key]
        if (tuple(image.shape) != (1, 3, 256, 256) or not torch.isfinite(image).all()
                or image.min() < 0 or image.max() > 1 or image.std() < 0.001):
            raise RuntimeError(f"Invalid or blank camera: {key}")


def environment_receipt(env, recipe):
    """Called after first reset; capture actual native controller and scene identities."""
    from .preflight import file_hash
    from pathlib import Path
    from libero.libero import get_libero_path
    from lerobot.envs.libero import _get_suite
    native = env.unwrapped.envs[0].unwrapped
    task_suite = _get_suite(recipe["suite"])
    controller = native._env.robots[0].controller
    keys = ("input_min", "input_max", "output_min", "output_max", "kp", "kd", "use_delta",
            "impedance_mode", "uncoupling", "control_freq", "control_dim")
    settings = {"class": type(controller).__name__}
    for name in keys:
        value = getattr(controller, name, None)
        if value is not None:
            settings[name] = value.tolist() if hasattr(value, "tolist") else value
    result = {"task_name": native.task, "instruction": native.task_description,
              "bddl_sha256": file_hash(native._task_bddl_file), "policy_hz": native.control_freq,
              "settle_steps": native.num_steps_wait, "hard_reset": native.hard_reset,
              "controller": settings}
    if task_suite is not None:
        task = task_suite.get_task(native.task_id)
        initial = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        result["initial_states_sha256"] = file_hash(initial)
        result["initial_state_count"] = len(native._init_states)
    return result
