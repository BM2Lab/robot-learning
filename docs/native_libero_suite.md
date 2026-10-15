# L3 — Small native SmolVLA LIBERO evaluation suite

L3 extends the [L0–L2 native reference](native_libero.md) to several preselected
tasks, explicit episode pairs, representative recordings and compiled results.
It changes neither the model weights nor the validated inference/controller
settings. It adds no live viewer and touches no Isaac or hardware integration.

## Default experiment

`configs/evaluations/smolvla_libero_suite.yaml` selects task 0 from each of
LIBERO Spatial, Object and Goal, before inspecting their results. Each runs two
episodes with explicit `(policy/environment seed, initial-state fixture index)`
pairs `(42,0)` and `(43,1)`. One video (episode 0) per task is saved regardless
of outcome; failed attempts are not selectively discarded.

This is **six episodes across three tasks**, not a full benchmark, unseen-task
generalization test or statistically precise success estimate. The long-horizon
suite is intentionally deferred. Do not select only successful tasks for later
comparisons. The checkpoint's exact training/task provenance remains bounded by
its upstream metadata; a matching benchmark interface alone does not prove
training-data overlap for every scene.

Native control stays at 20 Hz, relative OSC_POSE, hard resets, 256×256 external
and wrist cameras, saved normalization, ten native settling steps and
`n_action_steps=1`. The native horizons are 280 steps for Spatial/Object, 300
for Goal, and 520 for Long if explicitly added. Simulation pauses for inference;
video playback uses simulation time, not wall-clock inference time.

## Implementation

| File | Owner/responsibility |
|---|---|
| `configs/evaluations/smolvla_libero_suite.yaml` | Preselected tasks, episode pairs, representative video count and wall-clock timeout |
| `learning/evaluation/libero_suite.py` | `load_suite()`, `job_recipe()`, `LiberoSuiteRunner`, `validate_job()`, `compile_results()` |
| `scripts/run_libero_suite.py` | Thin command-line entry point; nonzero exit for incomplete infrastructure |
| `learning/integrations/libero/environment.py` | `select_initial_state()` chooses the exact fixture row and computes its digest |
| `learning/evaluation/libero.py` | Existing upstream rollout; optional explicit schedules and video limit; module entry point for children |
| `tests/test_libero_suite.py` | Schema/schedule validation, isolated job orchestration, failure accounting and compilation |

Paths beginning `learning/` are under `packages/learning/src/`; tests are under
`packages/learning/tests/`. No new dependency or environment install is required.
The existing L0–L2 single-task commands/config remain valid.

Each task runs serially in a fresh process using the coordinator's interpreter.
This isolates MuJoCo/EGL and model state between tasks and avoids GPU contention.
It reloads the model per task, so startup time is additional to rollout latency.
The parent uses no ML/simulator imports; the child uses the pinned LIBERO profile.
An explicit timeout bounds each job. No automatic retries or silent resume occur.

Before each episode the child sets the exact fixture row, then upstream reset
loads that row and performs settling. Out-of-range indices fail rather than wrap.
Rows record the selected fixture's dtype/shape-aware SHA-256, the complete fixture
file hash, seed and task identity. Identical fixture indices across **different
tasks** are not identical robot/object states: each task has its own fixture file.
Pairing is meaningful for rerunning the same task with another policy/settings.
Exact fixture identity does not promise bit-identical physics on another stack.

## Commands on this workstation

From `/home/chen-lab/isaac/lab-workspace/robot-learning`, using the existing
isolated LIBERO environment and cache:

```bash
direnv exec . ../state/venvs/libero-eval/bin/python -u scripts/run_libero_suite.py \
  --config configs/evaluations/smolvla_libero_suite.yaml \
  --runtime-dir /home/chen-lab/isaac/lab-workspace/state/cache/libero-eval \
  --output-dir /tmp/libero-reference-suite --smoke
```

This is offline by default. Add `--download` only when explicitly allowing pinned
model/asset downloads on a new machine. Never reuse an existing output directory.
The coordinator validates mounted/writable/capacity-checked `LAB_DATA_ROOT` for
persistent results; without `--smoke`, `--output-dir` must be a child of that root.
The explicit `/tmp` smoke is limited to at most three tasks × two episodes.
No fallback to the checkout, home directory or historical Franka data occurs.

The default suite may take several minutes: each task includes model startup,
and unsuccessful episodes run the whole native horizon. The JSON config and
actual commands are retained. Child progress/warnings live in each `stdout.log`.

## Artifacts

```text
<output>/
├── resolved_suite.json
├── policy_pin.json
├── suite_manifest.json
├── jobs/
│   ├── libero_spatial_00/
│   │   ├── recipe.json
│   │   ├── stdout.log
│   │   └── artifacts/
│   │       ├── manifest.json
│   │       ├── summary.json
│   │       ├── episodes.jsonl
│   │       └── videos/episode_000.mp4
│   ├── libero_object_00/...
│   └── libero_goal_00/...
└── compiled/
    ├── manifest.json         # compiler source/config and input-artifact hashes
    ├── summary.json
    ├── tasks.csv
    ├── episodes.jsonl
    └── report.md
```

The compiler requires a zero child exit **and** a complete matching manifest,
model pin, resolved recipe, full episode schedule, fixture digests, valid metrics
and requested video files. It rejects missing/wrong/stale artifacts. Infrastructure
failure does not count as a policy timeout or success. Other tasks still run;
completed valid jobs remain available for diagnosis.

`compiled/summary.json` separates planned versus validated coverage. Overall
`success_rate` and mean task success rate are **null unless all tasks validated**;
`partial_success_rate_completed_jobs` is explicitly labeled. A valid run with
zero task successes still has `complete=true`: that is a legitimate policy result.

Task tables report success counts/rates, mean steps over all episodes, mean steps
over successes only, timeout counts and step-weighted policy-call latency. The
Markdown report links the predetermined representative videos. No plot or score
implies continuous-time tracking or hardware readiness.

## Verification

Verified 2026-10-15 using cached models/assets, with no network downloads:

| Task | Successes | Policy steps per episode | Mean policy-call latency |
|---|---:|---|---:|
| Spatial 0: bowl onto plate | 2/2 | 76, 79 | 216 ms |
| Object 0: alphabet soup into basket | 2/2 | 151, 175 | 212 ms |
| Goal 0: open middle drawer | 1/2 | 300 (timeout), 144 (success) | 215 ms |

All three jobs and all six episode records validated; five episodes succeeded.
The descriptive pooled success rate is 83.3% for **this six-episode slice only**.
No tasks/episodes were removed or substituted after seeing results. All three
predetermined first-episode videos are retained, including Goal's timeout.

Results: `/tmp/lab-libero-l3-01/compiled`; per-task artifacts under the sibling
`jobs` directory. These are temporary smoke artifacts, not persistent lab data.
The compiler can be called again on unchanged raw artifacts; its own manifest
records compiler source, configuration identity and input-file hashes.

All 94 contracts/learning tests passed in both the isolated LIBERO venv and the
existing learning venv; all five workspace source checks passed. Tests cover
nonzero exits, timeouts, interruption, missing/mismatched artifacts and explicit
initial-state schedules, not only successful rollouts. Existing Isaac/hardware
environments, historical `franka-rl`, and model inference settings were unchanged.
