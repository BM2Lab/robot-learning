import json
from pathlib import Path
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("huggingface_hub")

from learning.integrations.lerobot.smolvla import SmolVLAInference
from learning.integrations.lerobot.snapshots import prepare_snapshots

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def pin():
    return json.loads((ROOT / "configs/policies/smolvla_base.json").read_text())


@pytest.fixture
def fake_model(pin):
    model = object.__new__(SmolVLAInference)
    model.pin = pin
    model.device = "cpu"
    return model


def sample(pin):
    c, h, w = pin["checkpoint_features"]["camera_shape_chw"]
    return {name: np.zeros((h, w, c), dtype=np.uint8) for name in pin["checkpoint_features"]["cameras"]}


def test_bad_inputs_rejected_before_model_execution(fake_model, pin):
    images = sample(pin)
    for state in ([0.]*7, [float("nan")]*6):
        with pytest.raises(ValueError):
            fake_model.predict_native(state, images, "reach")
    with pytest.raises(ValueError):
        fake_model.predict_native([0.]*6, {}, "reach")
    with pytest.raises(ValueError):
        fake_model.predict_native([0.]*6, images, "")
    with pytest.raises(ValueError):
        fake_model.predict_native([0.]*6, {k: v.astype(np.float32) for k, v in images.items()}, "reach")
    with pytest.raises(ValueError):
        fake_model.predict_native([0.]*6, images, "reach", seed=-1)


def test_reset_clears_all_upstream_state(fake_model):
    fake_model.policy, fake_model.pre, fake_model.post = Mock(), Mock(), Mock()
    fake_model.reset()
    for item in (fake_model.policy, fake_model.pre, fake_model.post):
        item.reset.assert_called_once()


def test_no_floating_model_revisions(pin, tmp_path):
    pin["policy"]["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        prepare_snapshots(pin, tmp_path)


def test_snapshots_are_pinned_offline_and_companion_has_no_weights(pin, tmp_path, monkeypatch):
    from hashlib import sha256
    paths = [tmp_path / pin[k]["revision"] for k in ("policy", "vlm_and_tokenizer")]
    for p in paths:
        p.mkdir()
    config = {"input_features": {"observation.state": {"shape": [6]}},
              "output_features": {"action": {"shape": [6]}}, "chunk_size": 50}
    (paths[0] / "config.json").write_text(json.dumps(config))
    for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        (paths[0] / name).write_text("test fixture, not model weights")
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json"):
        (paths[1] / name).write_text("{}")
    pin["policy"]["config_sha256"] = sha256((paths[0] / "config.json").read_bytes()).hexdigest()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(paths[len(calls)-1])

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    result = prepare_snapshots(pin, tmp_path)
    assert result.policy == paths[0]
    assert all(c["local_files_only"] for c in calls)
    assert calls[0]["revision"] == pin["policy"]["revision"]
    assert "*.safetensors" not in calls[1]["allow_patterns"]
    calls.clear()
    pin["policy"]["config_sha256"] = "0"*64
    with pytest.raises(RuntimeError, match="hash"):
        prepare_snapshots(pin, tmp_path)
