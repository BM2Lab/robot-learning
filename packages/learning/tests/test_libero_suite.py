import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from learning.evaluation.libero_suite import (
    LiberoSuiteRunner, compile_results, job_recipe, load_suite, validate_job, write_json)
from learning.integrations.libero.config import load_recipe

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "configs/evaluations/smolvla_libero_suite.yaml"


def artifact(directory, recipe, pin, successes=(True, False)):
    directory.mkdir(parents=True)
    rows = [{"episode": i, "seed": seed, "initial_state_index": recipe["initial_state_indices"][i],
             "initial_state_sha256": "a" * 64, "success": success,
             "policy_steps": 10 if success else recipe["max_steps"],
             "termination_reason": "success" if success else "time_limit",
             "mean_policy_call_seconds": .2, "p95_policy_call_seconds": .22, "wall_seconds": 3.,
             "video": f"videos/episode_{i:03d}.mp4" if i < recipe["video_episodes"] else None}
            for i, (seed, success) in enumerate(zip(recipe["episode_seeds"], successes, strict=True))]
    write_json(directory / "manifest.json", {"status": "complete", "mode": "pretrained_policy_rollout",
               "backend": "native_libero_mujoco_not_isaac", "recipe": recipe, "model_pin": pin,
               "environment": {"task_name": "fixture_task"},
               "model_actions_applied": sum(r["policy_steps"] for r in rows)})
    write_json(directory / "summary.json", {"completed": True, "episodes": len(rows),
                                           "success_rate": sum(successes) / len(rows)})
    (directory / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for row in rows:
        if row["video"]:
            video = directory / row["video"]
            video.parent.mkdir(exist_ok=True)
            video.write_bytes(b"mock video fixture, not actual encoded media")
    return rows


def test_default_suite_is_preselected_and_job_recipes_are_portable(tmp_path):
    cfg, pin = load_suite(RECIPE)
    assert len(cfg["tasks"]) == 3
    assert cfg["episodes"] == [{"seed": 42, "initial_state_index": 0}, {"seed": 43, "initial_state_index": 1}]
    write_json(tmp_path / "policy_pin.json", pin)
    job = tmp_path / "jobs/test"
    job.mkdir(parents=True)
    write_json(job / "recipe.json", job_recipe(cfg, cfg["tasks"][0]))
    loaded, same_pin = load_recipe(job / "recipe.json")
    assert same_pin == pin and loaded["initial_state_indices"] == [0, 1]


@pytest.mark.parametrize("change", [
    lambda c: c.update(name="../escape"), lambda c: c.update(job_timeout_seconds=True),
    lambda c: c.update(video_episodes=3), lambda c: c.update(n_action_steps=0),
    lambda c: c.update(tasks=c["tasks"] * 2), lambda c: c.update(tasks=[]),
    lambda c: c.update(episodes=c["episodes"] * 2), lambda c: c["episodes"][0].update(seed=-1),
    lambda c: c["episodes"][0].update(initial_state_index=50),
    lambda c: c["tasks"][0].update(task_id=True), lambda c: c.update(extra=1),
])
def test_bad_suite_rejected(tmp_path, change):
    cfg, _ = load_suite(RECIPE)
    cfg["policy_pin"] = str(ROOT / "configs/policies/smolvla_libero.json")
    change(cfg)
    write_json(tmp_path / "suite.json", cfg)
    with pytest.raises(ValueError):
        load_suite(tmp_path / "suite.json")


@pytest.mark.parametrize("change", [
    lambda m: m.update(status="failed"), lambda m: m["model_pin"]["policy"].update(revision="b"*40),
    lambda m: m["recipe"].update(n_action_steps=10), lambda m: m.update(model_actions_applied=0),
])
def test_zero_exit_does_not_certify_wrong_artifacts(tmp_path, change):
    cfg, pin = load_suite(RECIPE)
    recipe = job_recipe(cfg, cfg["tasks"][0])
    directory = tmp_path / "artifacts"
    artifact(directory, recipe, pin)
    manifest = json.loads((directory / "manifest.json").read_text())
    change(manifest)
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError):
        validate_job(directory, recipe, pin)


@pytest.mark.parametrize("change", [
    lambda r: r[0].update(initial_state_index=2), lambda r: r[0].update(seed=99),
    lambda r: r[0].update(initial_state_sha256=""), lambda r: r[0].update(success="yes"),
    lambda r: r[0].update(policy_steps=0), lambda r: r[0].update(termination_reason="time_limit"),
    lambda r: r[0].update(mean_policy_call_seconds=float("nan")),
    lambda r: r[1].update(video="videos/episode_001.mp4"), lambda r: r.pop(),
])
def test_bad_episode_schedule_or_coverage_rejected(tmp_path, change):
    cfg, pin = load_suite(RECIPE)
    recipe = job_recipe(cfg, cfg["tasks"][0])
    directory = tmp_path / "artifacts"
    rows = artifact(directory, recipe, pin)
    change(rows)
    (directory / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        validate_job(directory, recipe, pin)


def test_compiler_keeps_missing_jobs_out_of_overall_score(tmp_path):
    cfg, pin = load_suite(RECIPE)
    jobs = [{"task": t, "directory": f"jobs/{i}", "status": "failed"} for i, t in enumerate(cfg["tasks"])]
    jobs[0]["status"] = "complete"
    artifact(tmp_path / "jobs/0/artifacts", job_recipe(cfg, cfg["tasks"][0]), pin)
    summary = compile_results(tmp_path, cfg, pin, jobs)
    assert not summary["complete"] and summary["success_rate"] is None
    assert summary["episodes_validated"] == 2 and summary["episodes_planned"] == 6
    assert summary["partial_success_rate_completed_jobs"] == .5
    assert (tmp_path / "compiled/tasks.csv").exists()
    assert "videos/episode_000.mp4" in (tmp_path / "compiled/report.md").read_text()


def test_runner_continues_after_failure_and_returns_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr("learning.evaluation.libero._git_receipt", lambda: {})
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(returncode=1)
        if len(calls) == 2:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        recipe = json.loads(Path(command[command.index("--config")+1]).read_text())
        pin = json.loads((tmp_path / "suite/policy_pin.json").read_text())
        artifact(Path(command[command.index("--output-dir")+1]), recipe, pin)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", execute)
    runner = LiberoSuiteRunner(RECIPE, tmp_path / "suite", tmp_path / "cache", smoke=True)
    result = runner.run()
    assert len(calls) == 3 and not result["complete"]
    manifest = json.loads((tmp_path / "suite/suite_manifest.json").read_text())
    assert [j["status"] for j in manifest["jobs"]] == ["failed", "timed_out", "complete"]
    assert result["success_rate"] is None and result["episodes_validated"] == 2


def test_runtime_exception_does_not_leave_suite_marked_running(tmp_path, monkeypatch):
    monkeypatch.setattr("learning.evaluation.libero._git_receipt", lambda: {})
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(subprocess, "run", interrupt)
    runner = LiberoSuiteRunner(RECIPE, tmp_path / "suite", tmp_path / "cache", smoke=True)
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    manifest = json.loads((tmp_path / "suite/suite_manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["jobs"][0]["status"] == "interrupted"


def test_exact_fixture_selection_rejects_wrapping():
    import numpy as np
    from learning.integrations.libero.environment import select_initial_state
    native = SimpleNamespace(_init_states=np.arange(12).reshape(3, 4), init_state_id=0)
    index, digest = select_initial_state(native, 2)
    assert index == native.init_state_id == 2 and len(digest) == 64
    assert select_initial_state(native, 1)[1] != digest
    for bad in (3, -1, True):
        with pytest.raises(ValueError):
            select_initial_state(native, bad)


@pytest.mark.parametrize("key,value", [("initial_state_indices", [0]), ("initial_state_indices", [0, True]),
    ("initial_state_indices", [0, 50]), ("episode_seeds", [42]), ("episode_seeds", [42, -1]),
    ("video_episodes", True), ("video_episodes", 3)])
def test_bad_single_job_extensions_rejected(tmp_path, key, value):
    cfg, pin = load_suite(RECIPE)
    recipe = job_recipe(cfg, cfg["tasks"][0])
    recipe["policy_pin"] = str(ROOT / "configs/policies/smolvla_libero.json")
    recipe[key] = value
    write_json(tmp_path / "recipe.json", recipe)
    with pytest.raises(ValueError):
        load_recipe(tmp_path / "recipe.json")


def test_compiler_invalidates_corrupt_completed_job(tmp_path):
    cfg, pin = load_suite(RECIPE)
    jobs = [{"task": t, "directory": f"jobs/{i}", "status": "complete"} for i, t in enumerate(cfg["tasks"])]
    for job in jobs:
        artifact(tmp_path / job["directory"] / "artifacts", job_recipe(cfg, job["task"]), pin)
    # Task success false is a legitimate policy outcome, not an infrastructure failure.
    result = compile_results(tmp_path, cfg, pin, jobs)
    assert result["complete"] and result["success_rate"] == .5
    (tmp_path / jobs[1]["directory"] / "artifacts/summary.json").unlink()
    result = compile_results(tmp_path, cfg, pin, jobs)
    assert not result["complete"] and result["success_rate"] is None
    assert result["tasks"][1]["status"] == "invalid_artifacts"
    assert "libero_object/0, episode" not in (tmp_path / "compiled/report.md").read_text()
