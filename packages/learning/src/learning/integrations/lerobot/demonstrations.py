"""LeRobot 0.6.1 image/Parquet export; no hub upload or policy normalization here."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np
from contracts.policies import PolicyIOSpec
from learning.data.demonstrations import load_episode
from learning.data.storage import validate_artifact_path


def export_demonstrations(source, output, repo_id="local/fr3_visual_reach_v3", smoke=False):
    if not os.environ.get("HF_HOME") and not os.environ.get("HF_DATASETS_CACHE"):
        raise RuntimeError("Configure explicit HF_HOME/HF_DATASETS_CACHE with the learning direnv; no cache fallback")
    source = Path(source).resolve()
    manifest = json.loads((source/"manifest.json").read_text())
    if not manifest.get("complete") or manifest.get("schema_version") != 1:
        raise ValueError("Complete V3 collection manifest required")
    if importlib.metadata.version("lerobot") != "0.6.1":
        raise RuntimeError("This exporter is qualified against LeRobot 0.6.1 only")
    spec = PolicyIOSpec.from_dict(manifest["contract"])
    if manifest.get("label_source") != "privileged_scripted_ik_expert":
        raise ValueError("Unexpected collection label source")
    entries = manifest["episodes"]
    if not entries or (smoke and len(entries) > 4):
        raise ValueError("Empty collection or temporary export exceeds smoke budget")
    if len({e["file"] for e in entries}) != len(entries) or len({e["seed"] for e in entries}) != len(entries):
        raise ValueError("Duplicate episode archives or seeds would leak across splits")
    reserved = sum(e["frames"] for e in entries)*sum(c.height*c.width*3 for c in spec.observation.cameras)*2
    output = validate_artifact_path(output, reserved, smoke)
    validated = []
    for entry in entries:
        path = (source/entry["file"]).resolve()
        if path.parent != source or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Archive path or checksum mismatch")
        metadata, _, _ = load_episode(path, spec)
        terminal = metadata["terminal"]
        if not terminal["success"] or terminal["timeout"] or terminal["invalid_state"]:
            raise ValueError("This BC exporter requires complete successful demonstrations only")
        if (metadata["split"], metadata["seed"], metadata["frames"]) != (entry["split"], entry["seed"], entry["frames"]):
            raise ValueError("Episode metadata differs from collection index")
        if metadata["terminal"] != entry["terminal"]:
            raise ValueError("Terminal outcome differs from collection index")
        validated.append(path)
    if {e["split"] for e in entries} != {"train", "validation"}:
        raise ValueError("Both whole-episode train and validation splits required")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {
        spec.observation.state_key: {"dtype": "float32", "shape": (7,), "names": list(spec.robot.joint_names)},
        "action": {"dtype": "float32", "shape": (3,), "names": list(spec.action.components)},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None}}
    for c in spec.observation.cameras:
        features[c.key] = {"dtype": "image", "shape": (c.height, c.width, 3), "names": ["height", "width", "channels"]}
    dataset = LeRobotDataset.create(repo_id=repo_id, root=output, fps=round(spec.action.policy_hz),
        robot_type=spec.robot.model, features=features, use_videos=False, video_backend="pyav")
    try:
        for path in validated:
            metadata, data, _ = load_episode(path, spec)
            for index in range(metadata["frames"]):
                frame = {spec.observation.state_key: data["state"][index].astype(np.float32),
                    "action": data["action"][index].astype(np.float32),
                    "next.done": np.array([index == metadata["frames"]-1], dtype=bool),
                    "task": metadata["instructions"][index]}
                frame.update({c.key: data[f"rgb_{i}"][index] for i, c in enumerate(spec.observation.cameras)})
                dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=False)
    finally:
        dataset.finalize()
    # Replay every saved row through the official reader, not just file metadata.
    replay = LeRobotDataset(repo_id, root=output, return_uint8=True, video_backend="pyav")
    checks = {"episode_count": replay.num_episodes == len(entries),
        "frame_count": len(replay) == sum(e["frames"] for e in entries),
        "state": True, "physical_action": True, "rgb_exact": True, "language": True,
        "episode_boundary": True, "relative_timestamp": True}
    offset = 0
    for episode_index, path in enumerate(validated):
        metadata, data, _ = load_episode(path, spec)
        for index in range(metadata["frames"]):
            row = replay[offset+index]
            checks["state"] &= np.array_equal(row[spec.observation.state_key].numpy(), data["state"][index].astype(np.float32))
            checks["physical_action"] &= np.array_equal(row["action"].numpy(), data["action"][index].astype(np.float32))
            checks["language"] &= row["task"] == metadata["instructions"][index]
            checks["episode_boundary"] &= (int(row["episode_index"]) == episode_index and bool(row["next.done"].reshape(-1)[0]) == (index == metadata["frames"]-1))
            checks["relative_timestamp"] &= abs(float(row["timestamp"])-index/spec.action.policy_hz) < 1e-5
            for i, c in enumerate(spec.observation.cameras):
                rgb = row[c.key].permute(1, 2, 0).numpy()
                # LeRobot's image reader returns CHW float32 in [0,1], even
                # with return_uint8=True (that flag controls video decoding).
                if np.issubdtype(rgb.dtype, np.floating):
                    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                        checks["rgb_exact"] = False
                    rgb = np.rint(rgb*255).astype(np.uint8)
                checks["rgb_exact"] &= rgb.dtype == np.uint8 and np.array_equal(rgb, data[f"rgb_{i}"][index])
        offset += metadata["frames"]
    splits = {split: [i for i, e in enumerate(entries) if e["split"] == split] for split in ("train", "validation")}
    # LeRobot's writer defaults all episodes to train and aggregates stats over
    # everything. Declare actual splits and provide training-only statistics.
    train_count = len(splits["train"])
    if splits["train"] != list(range(train_count)) or splits["validation"] != list(range(train_count, len(entries))):
        raise ValueError("V3 split layout must be contiguous train then validation episodes")
    from lerobot.datasets.io_utils import write_info
    from lerobot.datasets.compute_stats import aggregate_stats
    import pyarrow.parquet as pq
    replay.meta.info.splits = {"train": f"0:{train_count}", "validation": f"{train_count}:{len(entries)}"}
    write_info(replay.meta.info, output)
    # The official metadata reader deliberately drops per-episode stats columns;
    # read those columns from its written Parquet metadata, without recomputing
    # or substituting all-episode global statistics.
    episode_stats_rows = {}
    for path in sorted((output/"meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            episode_stats_rows[row["episode_index"]] = row
    per_episode = []
    for episode_index in splits["train"]:
        stats = {}
        for key, value in episode_stats_rows[episode_index].items():
            if key.startswith("stats/"):
                _, feature, statistic = key.split("/")
                stats.setdefault(feature, {})[statistic] = np.asarray(value)
        per_episode.append(stats)
    train_stats = aggregate_stats(per_episode)
    checks["training_only_statistics"] = spec.observation.state_key in train_stats and "action" in train_stats
    (output/"train_normalization_stats.json").write_text(json.dumps(
        {key: {stat: value.tolist() for stat, value in stats.items()} for key, stats in train_stats.items()}, indent=2)+"\n")
    report = {"passed": all(checks.values()), "checks": {k: bool(v) for k, v in checks.items()},
        "repo_id": repo_id, "frames": len(replay), "episodes": replay.num_episodes,
        "splits": splits, "contract": spec.to_dict(), "label_source": manifest["label_source"],
        "collection_manifest_sha256": hashlib.sha256((source/"manifest.json").read_bytes()).hexdigest(),
        "simulator_receipt": manifest["simulator_receipt"],
        "runtime": {n: importlib.metadata.version(n) for n in ("lerobot", "datasets", "numpy", "pillow", "pyarrow", "fsspec")},
        "action_units": "meters; base-frame increments relative to measured TCP at execution",
        "normalized": False, "pretrained_checkpoint_compatible": False}
    report["training_statistics"] = "train_normalization_stats.json; do not use all-episode meta/stats.json for training"
    report["collection_source_sha256"] = manifest.get("source_sha256", {})
    report["exporter_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["source_episodes"] = entries
    (output/"demonstration_contract.json").write_text(json.dumps(report, indent=2)+"\n")
    (output/"splits.json").write_text(json.dumps(splits, indent=2)+"\n")
    if not report["passed"]:
        raise RuntimeError(f"LeRobot round-trip verification failed: {checks}")
    return report
