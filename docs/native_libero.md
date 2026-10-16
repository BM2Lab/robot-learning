# L0–L2 — Native SmolVLA LIBERO reference

This is an optional, synchronous **MuJoCo/robosuite/LIBERO** benchmark integration
in `learning`, not an Isaac Lab task, FR3 policy adapter or hardware deployment.
It requires no new demonstrations and downloads no demonstration dataset.
Existing V0–V3, `contracts`, `isaac-tasks` and hardware runtimes remain separate.

## Ownership

| Location | Responsibility |
|---|---|
| `configs/policies/smolvla_libero.json` | Immutable policy/companion/asset revisions, feature semantics, model/processor hashes, runtime versions |
| `configs/evaluations/smolvla_libero_smoke.yaml` | One task, episodes, seeds, horizon, device and video setting |
| `integrations/libero/config.py` | Strict bounded recipe loading |
| `integrations/libero/preflight.py` | Explicit runtime paths, version checks, pinned assets and checksums |
| `integrations/libero/environment.py` | Upstream environment factory, observation validation, actual controller/initial-state receipt |
| `integrations/lerobot/smolvla.py`, `snapshots.py` | Existing strict native checkpoint loader; optional per-file hash validation added |
| `evaluation/libero.py` | Upstream LeRobot `rollout()` orchestration, action audit, episode summaries and artifacts |
| `scripts/verify_libero_env.py` | L0 reset, two-camera render and five neutral steps; zero model actions |
| `scripts/evaluate_smolvla_libero.py` | L1 finite full-chunk checks + L2 closed-loop episodes |

All integration imports are lazy. Neither `contracts` nor Isaac requires these
dependencies. Do **not** send the native 7D action through `CartesianDeltaSpec`.

## Reproducible installation

Linux x86-64 / Python 3.12 / CUDA 12.8 profile:

```bash
uv venv /absolute/path/to/libero-eval --python python3.12
uv pip sync requirements/libero-py312-cu128.lock \
  --python /absolute/path/to/libero-eval/bin/python \
  --require-hashes --index-strategy unsafe-best-match
uv pip install --python /absolute/path/to/libero-eval/bin/python --no-deps \
  -e packages/contracts -e packages/learning
uv pip check --python /absolute/path/to/libero-eval/bin/python
```

The lock uses explicitly trusted PyPI and PyTorch indexes. Do not install this
profile in an existing Isaac/learning/hardware venv. The `.in` file constrains
the prior inference versions; its resolved lock also pins all new dependencies.
Re-resolving is an intentional maintenance operation, not a rollout step.

On this workstation the new venv is
`/home/chen-lab/isaac/lab-workspace/state/venvs/libero-eval`.
Use the repository direnv to clear unrelated native paths, but explicitly invoke
that interpreter. Merely activating the existing learning venv does not install
LIBERO. No `.envrc` changes are required.

## Run L0

From `robot-learning`:

```bash
direnv exec . ../state/venvs/libero-eval/bin/python -u scripts/verify_libero_env.py \
  --config configs/evaluations/smolvla_libero_smoke.yaml \
  --runtime-dir /home/chen-lab/isaac/lab-workspace/state/cache/libero-eval \
  --output-dir /tmp/libero-environment-check --download --smoke
```

`--runtime-dir` is an explicit POSIX hot-cache/config location. Before importing
GL/ML modules, the entry point selects EGL, redirects HF/Numba/Matplotlib/CUDA
caches and requires the qualified runtime versions. No Kit launcher is used.
L0 checks finite state, nonblank camera images, five neutral native actions and
clean close. It saves two processed camera PNGs for visual inspection.

`hf-libero==0.1.4` otherwise prompts for configuration and automatically downloads
**unpinned** assets into a home cache. The adapter writes its own config only if
absent and supplies its pinned asset snapshot through that version's private
`_assets_path_cache` hook. Existing conflicting config fails, rather than being
overwritten. No installed upstream source is patched. Requalify this hook if
the package changes. Native initial states use upstream trusted package fixtures;
never load untrusted replacement pickle/torch initial-state files.

## Run L1/L2

```bash
direnv exec . ../state/venvs/libero-eval/bin/python -u scripts/evaluate_smolvla_libero.py \
  --config configs/evaluations/smolvla_libero_smoke.yaml \
  --runtime-dir /home/chen-lab/isaac/lab-workspace/state/cache/libero-eval \
  --output-dir /tmp/libero-policy-rollout --download --smoke
```

Omit `--download` after the first run for **offline-only** cached evaluation.
Use a fresh output directory: existing results are never overwritten.
Downloads are pinned/capacity checked (about 422 MB assets and 1.22 GB weights,
plus small tokenizer/config files). Companion VLM weights are not downloaded;
the full policy already includes them. Saved processors and all six policy files
are SHA-256 checked, and weights are restored strictly.

The default task is spatial task 0: pick up the black bowl between the plate and
ramekin and place it on the plate. Two episodes use seeds 42/43 and initial-state
indices 0/1. Hard reset and ten native settling steps precede each episode.
Initial-state index and seed are separate: seeds do not select the fixture index.

### Exact execution contract

- Two RGB views: `agentview` and `robot0_eye_in_hand`, 256×256. Upstream environment
  preprocessing converts to float CHW and rotates each image 180 degrees **once**.
- State is EE position(3), orientation rotation vector(3), finger qpos(2), not q7.
  Upstream quaternion conversion expects XYZW. Position uses native LIBERO world
  coordinates, not our FR3 base/flange coordinates.
- The policy predicts finite `[1, 50, 7]` normalized actions, then saved postprocessing
  unnormalizes them to native controller inputs. Default execution is one action
  before replanning (`n_action_steps=1`). An explicit recipe override is recorded.
- Native Box bounds are enforced explicitly before physics: clipping to [-1,1]
  with component counts/fraction and raw peaks recorded. This does **not** map
  actions to meters. Robosuite OSC performs its own scaling: translation ±0.05 m,
  rotation ±0.5 rad for full-scale inputs; the seventh component drives the gripper.
- Simulated control rate is 20 Hz. Physics pauses during inference; no 20 Hz
  wall-clock guarantee, asynchronous execution, ensemble or physical-robot timing
  is implied. Policy-call latency uses CUDA synchronization and includes queued
  retrieval if execution horizon is increased.
- Upstream `rollout()` owns observation/action preprocessing, termination and
  success extraction. The wrapper rejects nonfinite/malformed inputs before
  stepping. Policy/processors reset per episode. No target state is injected.

## Artifacts and interpretation

```text
<output>/
├── manifest.json           # running/failed/complete, pins, source/package hashes,
│                           # assets/BDDL/initial-state hashes, actual controller
├── resolved_config.json
├── episodes.jsonl          # completed episode records, including task failures
├── summary.json            # infrastructure complete is distinct from task success
└── videos/episode_000.mp4   # optional external-camera playback at simulation rate
```

Episode rows include task success, steps, simulated duration, seed, initial-state
index, termination reason, reward, latency and action saturation. Time limit is
explicit when the configured rollout horizon is reached; early native termination
without success is `native_done`. Exceptions retain a failed manifest and nonzero
exit code; partial attempts are never reported as successful evaluations.

`--smoke` allows at most two bounded episodes with explicitly chosen `/tmp`
artifacts. Persistent output instead requires a mounted, writable, capacity-checked
`LAB_DATA_ROOT` child and omission of `--smoke`. The nearly full historical Franka
data volume is not repurposed. Models and reusable assets are hot caches, not run
results. No telemetry files are written on every physics step.

Two episodes of one task are an integration check, **not a published benchmark
score** or evidence of transfer to Isaac/FR3/hardware. L3 small-suite evaluation is
now implemented; see [native LIBERO suite](native_libero_suite.md). An Isaac port
remains separate work.

The first port milestone, P0's offline native OSC/model reference exporter, is
implemented separately: see [P0 reference export](libero_reference_export.md).
This does not yet provide an Isaac scene or policy rollout.

P1/P2 now provide an Isaac qualification scene, native visual contract and
neutral-action local bridge. See [P2 Isaac bridge](isaac_libero_bridge.md).
Pretrained rollout and task-success qualification are still separate from the
native MuJoCo benchmark documented here.

## Verification

Verified on this workstation, 2026-10-15:

- Hash-locked isolated installation: 164 distributions including local editable
  packages; `uv pip check` passed. Core versions are recorded in the model pin.
- L0 final offline run: `/tmp/lab-libero-l0-04`, five neutral steps, finite 8D state,
  both processed cameras visually inspected, zero policy actions, clean close.
  Maximum measured neutral joint change was 0.00120 rad.
- L1/L2 final offline run: `/tmp/lab-libero-l2-02`, strict checkpoint restoration,
  finite `[1,50,7]` chunks, 155 applied native actions; **both episodes succeeded**.
  Seeds/initial-state indices: 42/0 and 43/1. Completion: 76 and 79 policy steps
  (3.80 and 3.95 simulated seconds). A prior download-enabled run reproduced the
  same step counts, successes and action clipping counts.
- Mean policy-call latency: 226 and 218 ms; episode wall times about 21 and 20 s
  after model loading. This is slower than real-time 20 Hz. Videos play at 20 FPS
  simulation time and contain 77/80 frames including the initial view.
- Explicit clipping affected 41/532 and 60/553 action components (7.7% and 10.8%);
  raw controller-input peaks were 1.041 and 1.098. These are normalized-controller
  bounds, not measured joint-speed or hardware safety claims.
- All 57 contracts/learning tests passed in both the new LIBERO venv and existing
  learning venv. All five workspace source checks passed; Git whitespace check
  passed. Existing `isaac-tasks` and historical `franka-rl` worktrees were unchanged.

These `/tmp` results are temporary verification artifacts, not a persistent lab
dataset. Failed development attempts remain failed; do not use them as evidence.
The two-episode success rate does not imply generalization across other tasks.

Upstream references: [checkpoint](https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102),
[LeRobot 0.6.1 rollout](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/lerobot_eval.py),
[environment adapter](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/envs/libero.py).
