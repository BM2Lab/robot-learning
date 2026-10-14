import base64
import json
from types import SimpleNamespace
import numpy as np
import pytest
from contracts.actions import CartesianDeltaSpec
from contracts.episodes import SampleContext
from contracts.observations import CameraSpec, ObservationSpec
from contracts.policies import PolicyIOSpec
from contracts.robot import RobotSpec
from contracts.samples import ObservationPacket, RGBSample
from learning.data.demonstrations import EpisodeBuffer, load_episode
from learning.data.storage import validate_artifact_path


def spec():
    joints = tuple(f"panda_joint{i}" for i in range(1, 8))
    cameras = tuple(CameraSpec(f"observation.images.{name}", f"{name}_optical", 2, 2) for name in ("overhead", "wrist"))
    return PolicyIOSpec("test", RobotSpec("sim", "fr3", "arm", joints, "base", "tip"),
        ObservationSpec("observation.state", joints, cameras, "task"),
        CartesianDeltaSpec("arm", "base", "tip", 20, .005), 50)


def packet(episode, step):
    context = SampleContext(episode, step, step*50_000_000, "simulation")
    images = tuple(RGBSample(c.key, c.optical_frame, 2, 2, step+1, context,
        base64.b64encode(np.full((2, 2, 3), step+20*i, dtype=np.uint8).tobytes()).decode()) for i, c in enumerate(spec().observation.cameras))
    return ObservationPacket("test", context, (step*.01,)*7, images, "move")


def save_episode(path, episode, seed, split):
    buffer = EpisodeBuffer(spec())
    buffer.append(packet(episode, 0), (.001, 0., 0.))
    buffer.append(packet(episode, 1), (.0, .001, 0.))
    terminal = {"episode_id": episode, "success": True, "invalid_state": False, "timeout": False, "position_error_m": .01}
    return buffer.save(path, terminal, seed, split)


def test_episode_roundtrip_and_observation_action_alignment(tmp_path):
    path = tmp_path/"episode.npz"
    save_episode(path, "ep", 1, "train")
    metadata, data, _ = load_episode(path, spec())
    assert metadata["alignment"] == "observation_t_action_t"
    assert data["state"][0, 0] == 0
    assert data["state"][1, 0] == .01
    assert np.array_equal(data["action"], [[.001, 0, 0], [0, .001, 0]])
    assert metadata["frames"] == 2


def test_no_cross_episode_or_missing_terminal_samples(tmp_path):
    buffer = EpisodeBuffer(spec())
    buffer.append(packet("ep", 0), (0., 0., 0.))
    with pytest.raises(ValueError): buffer.append(packet("other", 1), (0., 0., 0.))
    with pytest.raises(ValueError): buffer.append(packet("ep", 2), (0., 0., 0.))
    with pytest.raises(ValueError): buffer.save(tmp_path/"incomplete.npz", None, 1, "train")
    with pytest.raises(ValueError): buffer.append(packet("ep", 1), (.1, 0., 0.))


def test_persistent_storage_is_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("LAB_DATA_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="LAB_DATA_ROOT"):
        validate_artifact_path(tmp_path/"dataset", 1024)
    assert validate_artifact_path(tmp_path/"dataset", 1024, smoke=True) == tmp_path/"dataset"
    with pytest.raises(ValueError): validate_artifact_path("/home/not-allowed", 1024, smoke=True)
    monkeypatch.setattr("learning.data.storage.shutil.disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match="capacity"): validate_artifact_path(tmp_path/"dataset", 1024, smoke=True)


def test_lerobot_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path/"hf-cache"))
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    import datasets.config
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path/"hf-cache")
    from learning.integrations.lerobot.demonstrations import export_demonstrations
    source = tmp_path/"source"
    source.mkdir()
    entries = [save_episode(source/f"episode_{i}.npz", f"ep{i}", i, split) for i, split in enumerate(("train", "validation"))]
    manifest = {"schema_version": 1, "complete": True, "contract": spec().to_dict(), "episodes": entries,
        "label_source": "privileged_scripted_ik_expert", "simulator_receipt": {"backend": "simulation"}}
    (source/"manifest.json").write_text(json.dumps(manifest))
    report = export_demonstrations(source, tmp_path/"lerobot", smoke=True)
    assert report["passed"] and report["frames"] == 4 and report["splits"] == {"train": [0], "validation": [1]}
    stats = json.loads((tmp_path/"lerobot/train_normalization_stats.json").read_text())
    assert stats["action"]["count"] == [2]
    assert np.allclose(stats["action"]["mean"], [.0005, .0005, 0])
    info = json.loads((tmp_path/"lerobot/meta/info.json").read_text())
    assert info["splits"] == {"train": "0:1", "validation": "1:2"}
