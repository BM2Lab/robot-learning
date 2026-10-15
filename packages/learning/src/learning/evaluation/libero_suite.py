"""Sequential isolated native LIBERO jobs and strict, dependency-light compilation."""
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

from learning.integrations.libero.config import SUITES, load_pin


def write_json(path, value):
    """Publish complete JSON records rather than exposing a partially written manifest."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def load_suite(path):
    import yaml
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    fields = {"schema_version", "name", "policy_pin", "device", "n_action_steps",
              "job_timeout_seconds", "video_episodes", "episodes", "tasks"}
    if (not isinstance(cfg, dict) or set(cfg) != fields or type(cfg["schema_version"]) is not int
            or cfg["schema_version"] != 1):
        raise ValueError("Invalid suite fields/schema")
    if not isinstance(cfg["name"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", cfg["name"]):
        raise ValueError("Unsafe suite name")
    if cfg["device"] not in ("cpu", "cuda"):
        raise ValueError("Invalid device")
    for key, low, high in (("n_action_steps", 1, 50), ("job_timeout_seconds", 30, 7200)):
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError(f"Invalid {key}")
    if not isinstance(cfg["episodes"], list) or not 1 <= len(cfg["episodes"]) <= 20:
        raise ValueError("Expected 1..20 episode pairs")
    if type(cfg["video_episodes"]) is not int or not 0 <= cfg["video_episodes"] <= len(cfg["episodes"]):
        raise ValueError("Invalid video count")
    pairs = []
    for ep in cfg["episodes"]:
        if not isinstance(ep, dict) or set(ep) != {"seed", "initial_state_index"}:
            raise ValueError("Invalid episode pair")
        if (type(ep["seed"]) is not int or not 0 <= ep["seed"] < 2**31
                or type(ep["initial_state_index"]) is not int or not 0 <= ep["initial_state_index"] < 50):
            raise ValueError("Invalid episode seed/fixture index")
        pairs.append((ep["seed"], ep["initial_state_index"]))
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate episode pair")
    if not isinstance(cfg["tasks"], list) or not 1 <= len(cfg["tasks"]) <= 40:
        raise ValueError("Expected 1..40 tasks")
    tasks = []
    for task in cfg["tasks"]:
        if (not isinstance(task, dict) or set(task) != {"suite", "task_id"}
                or task["suite"] not in SUITES or type(task["task_id"]) is not int
                or not 0 <= task["task_id"] < 10):
            raise ValueError("Invalid native task")
        tasks.append((task["suite"], task["task_id"]))
    if len(set(tasks)) != len(tasks):
        raise ValueError("Duplicate task")
    return cfg, load_pin(path.parent / cfg["policy_pin"])


def job_recipe(cfg, task):
    return {"schema_version": 1, "policy_pin": "../../policy_pin.json", **task,
            "episodes": len(cfg["episodes"]), "seed": cfg["episodes"][0]["seed"],
            "max_steps": SUITES[task["suite"]], "n_action_steps": cfg["n_action_steps"],
            "device": cfg["device"], "video": cfg["video_episodes"] > 0,
            "video_episodes": cfg["video_episodes"],
            "episode_seeds": [e["seed"] for e in cfg["episodes"]],
            "initial_state_indices": [e["initial_state_index"] for e in cfg["episodes"]]}


def validate_job(directory, recipe, pin):
    """A zero process exit is insufficient: validate identity, coverage and selected fixtures."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
    if (manifest.get("status") != "complete" or manifest.get("mode") != "pretrained_policy_rollout"
            or manifest.get("backend") != "native_libero_mujoco_not_isaac"
            or manifest.get("recipe") != recipe or manifest.get("model_pin") != pin
            or summary.get("completed") is not True or len(rows) != recipe["episodes"]):
        raise ValueError("Wrong run identity, incomplete manifest or episode coverage")
    for index, row in enumerate(rows):
        if (row.get("episode") != index or row.get("seed") != recipe["episode_seeds"][index]
                or row.get("initial_state_index") != recipe["initial_state_indices"][index]
                or not re.fullmatch(r"[a-f0-9]{64}", row.get("initial_state_sha256", ""))
                or type(row.get("success")) is not bool or type(row.get("policy_steps")) is not int
                or not 1 <= row["policy_steps"] <= recipe["max_steps"]
                or row.get("termination_reason") not in ("success", "time_limit", "native_done")
                or row["success"] != (row["termination_reason"] == "success")):
            raise ValueError("Episode schedule, fixture digest or metrics mismatch")
        video = f"videos/episode_{index:03d}.mp4" if index < recipe["video_episodes"] else None
        if row.get("video") != video or (video and (not (directory / video).is_file()
                                                    or (directory / video).stat().st_size == 0)):
            raise ValueError("Missing or incorrectly indexed representative video")
        for key in ("mean_policy_call_seconds", "p95_policy_call_seconds", "wall_seconds"):
            import math
            if not isinstance(row.get(key), (float, int)) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid latency: {key}")
    if (summary.get("episodes") != len(rows)
            or summary.get("success_rate") != sum(r["success"] for r in rows) / len(rows)
            or manifest.get("model_actions_applied") != sum(r["policy_steps"] for r in rows)):
        raise ValueError("Summary disagrees with episode rows")
    return rows, manifest["environment"]


def compile_results(output, cfg, pin, jobs):
    output = Path(output)
    compiled = output / "compiled"
    compiled.mkdir(exist_ok=True)
    table, all_rows = [], []
    for job in jobs:
        task = job["task"]
        entry = {**task, "status": job["status"], "episodes": 0, "successes": None,
                 "success_rate": None, "mean_policy_steps": None, "mean_success_steps": None,
                 "time_limits": None, "mean_policy_call_ms": None, "error": job.get("error")}
        if job["status"] == "complete":
            try:
                rows, native = validate_job(output / job["directory"] / "artifacts", job_recipe(cfg, task), pin)
                successes = [r for r in rows if r["success"]]
                entry.update(task_name=native["task_name"], episodes=len(rows), successes=len(successes),
                             success_rate=len(successes) / len(rows),
                             mean_policy_steps=sum(r["policy_steps"] for r in rows) / len(rows),
                             mean_success_steps=(sum(r["policy_steps"] for r in successes) / len(successes)
                                                 if successes else None),
                             time_limits=sum(r["termination_reason"] == "time_limit" for r in rows),
                             mean_policy_call_ms=1000 * sum(r["mean_policy_call_seconds"] * r["policy_steps"]
                                                           for r in rows) / sum(r["policy_steps"] for r in rows))
                all_rows.extend({**task, **row} for row in rows)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                entry.update(status="invalid_artifacts", error=str(exc))
        table.append(entry)
    complete = all(row["status"] == "complete" for row in table) and len(table) == len(cfg["tasks"])
    successes = sum(row["success"] for row in all_rows)
    partial = successes / len(all_rows) if all_rows else None
    report = {"name": cfg["name"], "complete": complete, "tasks_planned": len(cfg["tasks"]),
              "tasks_validated": sum(t["status"] == "complete" for t in table),
              "episodes_planned": len(cfg["tasks"]) * len(cfg["episodes"]),
              "episodes_validated": len(all_rows), "successes_validated": successes,
              "success_rate": partial if complete else None,
              "partial_success_rate_completed_jobs": partial,
              "mean_task_success_rate": (sum(t["success_rate"] for t in table) / len(table)
                                         if complete else None),
              "tasks": table, "claim": "Small preselected reference slice, not a full LIBERO benchmark"}
    write_json(compiled / "summary.json", report)
    (compiled / "episodes.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in all_rows))
    columns = ["suite", "task_id", "task_name", "status", "episodes", "successes", "success_rate",
               "mean_policy_steps", "mean_success_steps", "time_limits", "mean_policy_call_ms", "error"]
    with (compiled / "tasks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    lines = [f"# {cfg['name']}", "", report["claim"], "",
             f"Validated: {report['tasks_validated']}/{len(cfg['tasks'])} tasks, "
             f"{len(all_rows)}/{report['episodes_planned']} episodes.", "",
             "| Task | Status | Successes | Mean steps (all) |", "|---|---|---|---|"]
    for row in table:
        score = f"{row['successes']}/{row['episodes']}" if row["status"] == "complete" else "—"
        steps = f"{row['mean_policy_steps']:.1f}" if row["mean_policy_steps"] is not None else "—"
        lines.append(f"| {row['suite']}/{row['task_id']} | {row['status']} | {score} | {steps} |")
    lines.extend(["", "Task descriptions:", ""])
    for row in table:
        if row.get("task_name"):
            lines.append(f"- {row['suite']}/{row['task_id']}: {row['task_name'].replace('_', ' ')}")
    lines.extend(["", "## Representative videos", ""])
    validated_tasks = {(r["suite"], r["task_id"]) for r in table if r["status"] == "complete"}
    for job in jobs:
        for index in range(cfg["video_episodes"]):
            video = output / job["directory"] / "artifacts/videos" / f"episode_{index:03d}.mp4"
            if (job["task"]["suite"], job["task"]["task_id"]) in validated_tasks and video.is_file():
                lines.append(f"- [{job['task']['suite']}/{job['task']['task_id']}, episode {index}]"
                             f"(../{video.relative_to(output).as_posix()})")
    (compiled / "report.md").write_text("\n".join(lines) + "\n")
    from learning.evaluation.libero import _git_receipt
    from learning.integrations.libero.preflight import file_hash
    inputs = {}
    for job in jobs:
        directory = output / job["directory"] / "artifacts"
        paths = [directory / name for name in ("manifest.json", "summary.json", "episodes.jsonl")]
        paths += sorted((directory / "videos").glob("*.mp4"))
        inputs.update({str(path.relative_to(output)): file_hash(path) for path in paths if path.is_file()})
    write_json(compiled / "manifest.json", {
        "schema_version": 1, "compiled_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete, "source": _git_receipt(), "input_files_sha256": inputs,
        "suite_sha256": sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "policy_pin_sha256": sha256(json.dumps(pin, sort_keys=True).encode()).hexdigest()})
    return report


class LiberoSuiteRunner:
    def __init__(self, config, output, runtime_dir, *, smoke=False, download=False):
        from learning.data.storage import validate_artifact_path
        self.cfg, self.pin = load_suite(config)
        if smoke and (len(self.cfg["tasks"]) > 3 or len(self.cfg["episodes"]) > 2):
            raise ValueError("/tmp suite smoke is bounded to three tasks × two episodes")
        self.output = validate_artifact_path(output, 256 * 1024**2, smoke=smoke)
        self.runtime = Path(runtime_dir).expanduser().resolve()
        self.smoke, self.download = smoke, download

    def run(self):
        from learning.evaluation.libero import _git_receipt
        self.output.mkdir(parents=True)
        write_json(self.output / "resolved_suite.json", self.cfg)
        write_json(self.output / "policy_pin.json", self.pin)
        jobs = [{"task": task, "directory": f"jobs/{task['suite']}_{task['task_id']:02d}", "status": "pending"}
                for task in self.cfg["tasks"]]
        manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "suite": self.cfg, "policy_pin": self.pin, "source": _git_receipt(), "jobs": jobs}
        def publish():
            write_json(self.output / "suite_manifest.json", manifest)
        publish()
        try:
            for index, job in enumerate(jobs):
                directory = self.output / job["directory"]
                directory.mkdir(parents=True)
                recipe = job_recipe(self.cfg, job["task"])
                write_json(directory / "recipe.json", recipe)
                command = [sys.executable, "-u", "-m", "learning.evaluation.libero",
                           "--config", str(directory / "recipe.json"), "--output-dir", str(directory / "artifacts"),
                           "--runtime-dir", str(self.runtime)]
                command += ["--smoke"] if self.smoke else []
                command += ["--download"] if self.download else []
                job.update(status="running", command=command)
                publish()
                print(f"[{index+1}/{len(jobs)}] {job['task']['suite']}/{job['task']['task_id']}", flush=True)
                try:
                    with (directory / "stdout.log").open("w") as stream:
                        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                                timeout=self.cfg["job_timeout_seconds"])
                    job["returncode"] = result.returncode
                    if result.returncode != 0:
                        raise RuntimeError(f"Evaluator exited {result.returncode}; see {directory/'stdout.log'}")
                    validate_job(directory / "artifacts", recipe, self.pin)
                    job["status"] = "complete"
                except subprocess.TimeoutExpired:
                    job.update(status="timed_out", error="Job exceeded configured wall-clock limit")
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                    job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                publish()
                print(f"  {job['status']}", flush=True)
        except BaseException as exc:
            for job in jobs:
                if job["status"] == "running":
                    job.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            report = compile_results(self.output, self.cfg, self.pin, jobs)
            manifest["status"] = "complete" if report["complete"] else "failed"
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            publish()
        print(f"Compiled results: {self.output/'compiled'}", flush=True)
        return report
