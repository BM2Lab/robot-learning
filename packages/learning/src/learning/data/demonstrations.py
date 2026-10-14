"""Buffered episode archives: observation-before-action BC alignment, no reset-frame leakage."""
import hashlib
import json
from pathlib import Path
import numpy as np
from contracts.samples import ObservationPacket
from contracts.policies import PolicyIOSpec


class EpisodeBuffer:
    def __init__(self, spec, max_steps=200):
        self.spec, self.max_steps = spec, max_steps
        self.observations, self.actions = [], []

    def append(self, observation: ObservationPacket, action):
        observation.validate_for(self.spec)
        row = self.spec.action.validate_values(action)
        if len(self.actions) >= self.max_steps:
            raise ValueError("Episode buffer step budget exhausted")
        if self.observations:
            previous = self.observations[-1].context
            current = observation.context
            if (current.episode_id != previous.episode_id or current.step_index != previous.step_index+1
                or current.clock != "simulation" or current.timestamp_ns-previous.timestamp_ns != round(1e9/self.spec.action.policy_hz)):
                raise ValueError("Discontinuous or reset-crossing episode samples")
        elif observation.context.step_index != 0 or observation.context.clock != "simulation":
            raise ValueError("Episode must start at simulation step zero")
        self.observations.append(observation)
        self.actions.append(row)

    def save(self, path, terminal, seed, split):
        path = Path(path)
        if not self.actions or path.exists() or split not in ("train", "validation"):
            raise ValueError("Empty episode, existing path or invalid split")
        episode_id = self.observations[0].context.episode_id
        if terminal is None or terminal["episode_id"] != episode_id:
            raise ValueError("Complete matching terminal episode required")
        metadata = {"schema_version": 1, "episode_id": episode_id, "seed": seed, "split": split,
            "terminal": terminal, "frames": len(self.actions), "contract": self.spec.to_dict(),
            "label_source": "privileged_scripted_ik_expert", "alignment": "observation_t_action_t",
            "instructions": [o.instruction for o in self.observations]}
        arrays = {"metadata_json": np.array(json.dumps(metadata, allow_nan=False)),
            "state": np.asarray([o.state for o in self.observations], dtype=np.float64),
            "action": np.asarray(self.actions, dtype=np.float64),
            "timestamp_ns": np.asarray([o.context.timestamp_ns for o in self.observations], dtype=np.int64),
            "step_index": np.asarray([o.context.step_index for o in self.observations], dtype=np.int64)}
        for index, camera in enumerate(self.spec.observation.cameras):
            images = [next(i for i in o.images if i.key == camera.key) for o in self.observations]
            arrays[f"rgb_{index}"] = np.stack([np.frombuffer(i.decode(), dtype=np.uint8).reshape(camera.height, camera.width, 3) for i in images])
            arrays[f"camera_frame_{index}"] = np.asarray([i.frame_index for i in images], dtype=np.int64)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": seed, "split": split, "frames": len(self.actions), "terminal": terminal}


def load_episode(path, expected_spec=None):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(data.pop("metadata_json")))
    spec = PolicyIOSpec.from_dict(metadata["contract"])
    if metadata["schema_version"] != 1 or metadata["alignment"] != "observation_t_action_t" or metadata["label_source"] != "privileged_scripted_ik_expert":
        raise ValueError("Unsupported demonstration archive semantics")
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("Archive policy contract mismatch")
    n = metadata["frames"]
    if type(n) is not int or not 1 <= n <= 200 or data["state"].shape != (n, len(spec.robot.joint_names)) or data["action"].shape != (n, 3):
        raise ValueError("Invalid state/action archive shapes")
    if not np.isfinite(data["state"]).all() or not np.isfinite(data["action"]).all():
        raise ValueError("Nonfinite demonstration data")
    if data["timestamp_ns"].shape != (n,) or data["timestamp_ns"].dtype != np.int64 or data["step_index"].dtype != np.int64 or not np.array_equal(data["step_index"], np.arange(n)) or not np.all(np.diff(data["timestamp_ns"]) == round(1e9/spec.action.policy_hz)):
        raise ValueError("Nonconsecutive episode samples")
    for row in data["action"]:
        spec.action.validate_values(row)
    for index, camera in enumerate(spec.observation.cameras):
        rgb = data[f"rgb_{index}"]
        if rgb.shape != (n, camera.height, camera.width, 3) or rgb.dtype != np.uint8:
            raise ValueError("Invalid camera archive")
        if data[f"camera_frame_{index}"].shape != (n,) or not np.all(np.diff(data[f"camera_frame_{index}"]) > 0):
            raise ValueError("Stale camera frames within episode")
    if len(metadata["instructions"]) != n or metadata["terminal"]["episode_id"] != metadata["episode_id"]:
        raise ValueError("Invalid episode language/terminal metadata")
    return metadata, data, spec
