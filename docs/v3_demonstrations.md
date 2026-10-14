# V3 — Expert demonstrations and LeRobot dataset round trip

V3 implements scripted-expert recording through the local simulator bridge and
lossless LeRobot export. It does **not** fine-tune SmolVLA or deploy a policy.
The V1 scene, cameras, physical action contract and controller are unchanged.

## Data contract

Each training row pairs the **pre-action** observation $o_t$ with the physical
Cartesian increment $a_t$ actually submitted for that step:

$$
o_t=(q_t,I_t^{\mathrm{overhead}},I_t^{\mathrm{wrist}},\text{instruction}),
\qquad a_t=\Delta p_t\in\mathbb R^3.
$$

Actions remain meters in the robot-base frame, relative to the measured TCP at
execution. Norm is ≤ 5 mm; policy sampling is 20 Hz. The existing DLS action term
holds episode-initial flange orientation. No joint target, gripper dimension,
normalized policy coordinate or native pretrained VLA output is substituted.

The simulator must explicitly enable `--enable-expert-demonstrations`. Its new
`expert_action` RPC generates a **single-step** privileged label, tagged with
the current capture context. The learning collector submits that label through
the same validated queue/advance API used in V2. The expert may access the target
internally; target coordinates never enter policy observations or exported features.
Without opt-in the RPC is rejected. The expert gain is 0.3; its norm limit is
5 mm minus a 10 nm serialization margin to keep float32 labels within the strict
wire-contract bound.

The terminal action is included. Gym's auto-reset observation belongs to the
next episode and is **not** appended as a terminal observation to the previous
one. Original capture times, camera counters, episode ID, outcome and seed are
retained in the raw archive. LeRobot generates episode-relative timestamps.

## Implementation and artifacts

| File | Responsibility |
|---|---|
| `isaac_tasks/bridges/local_session.py`, `local_server.py` | Opt-in privileged labels, capture/queue checks and actual controller/source receipts |
| `learning/data/storage.py` | Explicit storage root, mount/writability/capacity checks, no overwrite or fallback |
| `learning/data/demonstrations.py` | Bounded episode buffer, aligned observation/action pairs, archive validation |
| `learning/rollouts/collect_expert.py` | Episode-seeded collection through IPC, complete manifests and whole-episode splits |
| `learning/integrations/lerobot/demonstrations.py` | LeRobot 0.6.1 export and full official-reader round-trip verification |
| `scripts/collect_sim_expert.py`, `export_sim_demonstrations.py` | Finite collection/export entry points |

Raw collection writes **one compressed NPZ per complete episode**, not one file
per physics step. It buffers at most 200 policy frames. `manifest.json` records
the contract, seed, outcome, file checksums, runtime/controller recipe, and source
hashes. An interrupted job remains incomplete and cannot be exported as complete.
Failed complete episodes remain recorded for audit, but the BC exporter rejects
collections containing unsafe failures or timeouts. There is no silent filtering.

The final ceil(20% × episode count) seeds are held out for validation. There is
no frame-level random split; duplicate archive names/seeds are rejected. For the
two-episode smoke this is one training episode and one validation episode.

LeRobot features are `observation.state` (7 float32 joints), `action` (3 float32
physical increments), both RGB image features, language and `next.done`. Export
uses **lossless image/Parquet storage**, not MP4; temporary PNG encoding is an
offline export operation, not simulation-step I/O. The official image reader
returns CHW floats in [0,1]; pixel reconstruction verifies exact original uint8
RGB values. Physical actions and joint state are not normalized by this exporter.

Sidecars:

- `demonstration_contract.json`: full verification, physical semantics, collection
  fingerprint, runtime/model/source receipts and incompatibility with the native V0 checkpoint.
- `splits.json` and `meta/info.json`: explicit training/validation episode indices/ranges.
- `train_normalization_stats.json`: official LeRobot statistics aggregated from
  **training episodes only**. The native writer's all-episode `meta/stats.json`
  includes validation data and must not be used to normalize training inputs/labels.

Future training must explicitly select `episodes=splits['train']`; loading the
combined root without episode selection loads both splits. No Hub upload occurs.

## Dataset environment

The isolated learning environment now includes the LeRobot dataset extra;
Isaac's environment remains unchanged. Reproducible profile:

```bash
cd /home/chen-lab/isaac/lab-workspace/robot-learning
uv pip install --python ../state/venvs/learning/bin/python \
  --require-hashes -r requirements/demonstrations-py312-cu128.lock \
  --cache-dir ../state/cache/uv
```

`demonstrations.in` + `demonstrations-constraints.txt` preserve the established
inference versions. One intentional compatibility change was necessary:
`fsspec` 2026.9.0 → 2026.2.0, required by datasets 4.8.5. The original inference
lock remains unchanged; use the demonstrations lock for the extended environment.
Torch 2.10.0+cu128, Transformers 5.5.4 and LeRobot 0.6.1 are unchanged. Dataset
imports/dependency checks and cached CUDA SmolVLA inference were reverified.
HF cache location must be explicit through the learning direnv; no home-cache fallback.

## Bounded smoke commands

Terminal A:

```bash
cd /home/chen-lab/isaac/lab-workspace/isaac-tasks
export ISAAC_TASKS_FR3_USD=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data/generated_assets/fr3v21_bare_flange_live_v1/fr3v2.usda
direnv exec . python -u scripts/serve_visual_reach.py \
  --contract ../robot-learning/configs/contracts/fr3_visual_reach_v1.json \
  --socket /home/chen-lab/isaac/lab-workspace/state/run/v3-demo/bridge.sock \
  --device cpu --viz none --idle-timeout 300 --max-requests 3000 \
  --enable-expert-demonstrations
```

Wait for `V2 simulation ready`. Do **not** use the V2 short-timeout test override
for successful expert demonstrations. `--viz kit` may replace `--viz none`.

Terminal B:

```bash
cd /home/chen-lab/isaac/lab-workspace/robot-learning
direnv exec . python -u scripts/collect_sim_expert.py \
  --contract configs/contracts/fr3_visual_reach_v1.json \
  --socket /home/chen-lab/isaac/lab-workspace/state/run/v3-demo/bridge.sock \
  --output-dir /tmp/fr3-v3-raw-demo --episodes 2 --seed 123 --smoke

direnv exec . python -u scripts/export_sim_demonstrations.py \
  --source /tmp/fr3-v3-raw-demo --output-dir /tmp/fr3-v3-lerobot-demo --smoke
```

Choose new output directories on reruns; existing outputs are never overwritten.
The explicit `--smoke` option allows ≤4 episodes under `/tmp`. Persistent jobs
require a provisioned mounted, writable `LAB_DATA_ROOT`, and output paths under
it. Capacity is reserved conservatively before connecting/creating datasets.
The existing NTFS volume has only ~2.3 GB free: no bulk job was launched there.
Source trees, environments and workspace runtime state are not dataset storage.
Provision POSIX staging if exporting large image datasets on NTFS; native LeRobot
PNG encoding is not intended as a high-frequency NTFS writer.

## Verified 2026-10-14

Two seeded expert episodes succeeded: 109 + 146 = **255 frames**, with no unsafe
failures/timeouts. The final raw verification collection is `/tmp/lab-v3-raw-02`;
the final export is `/tmp/lab-v3-lerobot-05`. Every exported row was read through
LeRobot and checked for state/action equality, exact reconstructed RGB, language,
episode boundary and relative timestamp. Whole-episode split and training-only
statistics tests pass. Artifacts are temporary, not a benchmark/training corpus.
All 36 contracts/learning tests, 16 Isaac tests and five workspace checks passed.

The cached SmolVLA CUDA regression passed with finite native shape `[1,50,6]` and
exact seeded replay (difference 0); no native model actions were applied.

Next: provision collection storage and prepare a task-specific SmolVLA adaptation/
training recipe for q7, two cameras and 3D Cartesian labels. Two demonstration
episodes are not enough to claim generalization or deployment readiness.
