import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from learning.integrations.libero.environment import bounded_native_action, validate_observation
from learning.evaluation.libero import summarize_episode, run_evaluation

ROOT = Path(__file__).resolve().parents[3]


def test_native_clipping_is_explicit_and_does_not_scale_units():
    raw = np.array([[2., -.3, 0., -2., .2, 1., -1.]])
    bounded, clipped = bounded_native_action(raw)
    assert clipped == 2
    assert bounded.tolist() == [[1., -.3, 0., -1., .2, 1., -1.]]
    assert raw[0, 0] == 2


@pytest.mark.parametrize("action", [np.zeros(7), np.zeros((1, 3)), np.full((1, 7), np.nan),
                                     np.full((1, 7), np.inf)])
def test_invalid_actions_fail_before_physics(action):
    with pytest.raises(RuntimeError):
        bounded_native_action(action)


def test_summary_does_not_claim_infrastructure_completion_is_task_success():
    data = {"action": torch.zeros(1, 3, 7), "success": torch.zeros(1, 3, dtype=torch.bool),
            "reward": torch.zeros(1, 3)}
    row = summarize_episode(data, episode=0, seed=42, initial_state=3, horizon=3, audit={})
    assert not row["success"] and row["termination_reason"] == "time_limit"
    assert row["initial_state_index"] == 3 and row["simulated_seconds"] == .15
    data["success"][0, 1] = True
    assert summarize_episode(data, episode=0, seed=42, initial_state=3, horizon=3,
                             audit={})["termination_reason"] == "success"


def test_upstream_state_and_camera_conventions():
    from lerobot.envs import preprocess_observation
    from lerobot.envs.configs import LiberoEnv
    processor, _ = LiberoEnv().get_env_processors()
    pixels = np.arange(256*256*3, dtype=np.uint8).reshape(1, 256, 256, 3)
    raw = {"pixels": {"image": pixels, "image2": pixels.copy()}, "robot_state": {
        "eef": {"pos": np.array([[.1, .2, .3]]), "quat": np.array([[0., 0., 0., 1.]])},
        "gripper": {"qpos": np.array([[.02, -.02]])}}}
    obs = processor(preprocess_observation(raw))
    validate_observation(obs)
    np.testing.assert_allclose(obs["observation.state"].numpy(), [[.1,.2,.3,0,0,0,.02,-.02]])
    np.testing.assert_allclose(obs["observation.images.image"][0, :, 0, 0].numpy(),
                               pixels[0, -1, -1, :] / 255, rtol=1e-6)
    obs["observation.images.image2"].zero_()
    with pytest.raises(RuntimeError, match="blank"):
        validate_observation(obs)


def test_preflight_failure_has_failed_manifest(tmp_path, monkeypatch):
    import learning.integrations.libero.preflight as preflight
    monkeypatch.setattr(preflight, "check_runtime", lambda *a, **kw: (tmp_path, {}))
    def fail(*a, **kw):
        raise RuntimeError("Missing offline assets")
    monkeypatch.setattr(preflight, "prepare_assets", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="offline"):
        run_evaluation(ROOT / "configs/evaluations/smolvla_libero_smoke.yaml", output,
                       tmp_path, smoke=True, verify_only=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["model_actions_applied"] == 0
    assert not (output / "summary.json").exists()
