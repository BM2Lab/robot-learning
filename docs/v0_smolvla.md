# V0 — Contracts and standalone SmolVLA inference

Status: implemented and verified on 2026-10-13. This stage does not implement
an Isaac environment, model adaptation, task training or hardware deployment.

## Typed physical contract

`configs/contracts/fr3_visual_reach_v1.json` is the proposed single-arm V1 task
contract, validated by standard-library-only dataclasses in `contracts`:

| Type | Responsibility |
|---|---|
| RobotSpec | Robot identity/model, manipulator, ordered joints, base and TCP frames |
| CameraSpec / ObservationSpec | Named uint8 RGB HWC images and measured joint positions in radians |
| SampleContext | Episode/source step, nanosecond timestamp and explicit clock domain |
| CartesianDeltaSpec | Physical translation semantics, rate, frame/TCP and bound |
| ActionChunk | Finite bounded values, source context, expiration and acceptance checks |
| PolicyIOSpec | Versioned consistent robot/observation/action specification |

These are minimal V0 contracts, not the complete fleet/session/recording schema.
Each supports JSON round trips. Unknown schema versions, ambiguous/mismatched
frames, reordered joints, invalid units, dimensions, non-finite values, duplicate
camera keys, expired actions and episode/clock mismatches are rejected.

The proposed FR3 task observes all seven joints and two real camera streams
(overhead/wrist), plus language. Frame names are logical identifiers, not an
assertion that USD link mappings or camera extrinsics already exist. Joint names
match the converted FR3 baseline's panda_joint1..7; V1 must verify the actual asset
order, bare-flange TCP mapping and camera calibration before registration.

Actions are **three-component Cartesian translation increments in meters**,
expressed in robot-base coordinates. Each is relative to the measured TCP at its
execution step. Desired orientation holds the episode-initial orientation through
the future IK controller. Initial policy rate is 20 Hz and the reference increment
L2 norm is bounded to 5 mm. This is a command bound, not a guarantee of measured
robot speed or hardware safety. No implicit clipping is performed by the schema.
Chunks contain at most 50 reference samples, with a later runner executing only
a configured prefix. Action acceptance must share the source episode/step/clock
and occur before the deadline; scheduler, camera capture timestamps, partial
execution, reset queues and transport are V1/V2 work, not implemented here.

## Pretrained model smoke (different contract)

`learning.integrations.lerobot.smolvla.SmolVLAInference` lazily loads the pinned
LeRobot release and exact policy/companion snapshots. The full policy checkpoint
contains the VLM parameters: initialize its architecture from the pinned local
companion configuration, then **strictly** restore all saved policy weights.
The companion's separate ~2 GB VLM weights are not needed or downloaded.
Saved pre/postprocessors and normalization state are loaded from the policy
snapshot. Tokenizer loads are redirected to the pinned local companion; no
nested loader resolves a floating main revision. No upstream source is patched.

The published base checkpoint has 6-D state/actions, three camera views and a
50-step action chunk. The smoke supplies clearly synthetic RGB patterns, a
synthetic six-element state and language, then generates a native action chunk.
It also resets the model/processors and replays using the same explicit noise seed.

**Those six-dimensional outputs have checkpoint-native training semantics, not
our FR3 Cartesian-increment semantics. They are never wrapped as physical
ActionChunk objects or applied to a robot/simulator.** Dimensional padding,
discarding channels or renaming units would not solve the semantic mismatch.
Feature/action adaptation, correct dataset statistics and fine-tuning are later
milestones. Synthetic inference does not measure reaching success or language
understanding.

## Run

On the configured workstation:

```bash
cd /home/chen-lab/isaac/lab-workspace/robot-learning
direnv exec . python scripts/smoke_smolvla.py --device cuda
```

On another host, use the pinned learning environment and explicitly configure
HF_HOME or pass `--cache-dir` outside the source repository. The default is
cache-only. Add `--download` for the first run to permit exact pinned downloads
(approximately 0.91 GB policy plus companion metadata/tokenizer files). A 4 GiB
free-space preflight is required before downloading. Weights remain in the model
cache, outside Git. The script writes its JSON result to stdout only; configure
an approved artifact store explicitly if retaining results or recording datasets.

```bash
python scripts/smoke_smolvla.py --download --cache-dir /configured/model-cache --device cuda
python -m pytest packages/contracts/tests packages/learning/tests --import-mode=importlib
```

ML tests require the optional SmolVLA environment and skip when those dependencies
are absent. Contract tests require only pytest and the contracts package. Neither
unit suite downloads weights or connects to hardware. The real-model smoke is
explicit and separate from unit tests/CI.

## Verification

The first RTX 4090 run passed: strict checkpoint loading, saved processors,
finite output shape `[1, 50, 6]`, and exact seeded reset replay (maximum difference
0.0). Inference wall time was approximately 0.363 s on the first request and
0.132 s on the second; these are one synthetic run, not production throughput
guarantees or a demonstrated 20 Hz replan rate. Robot actions applied: **zero**.
All 23 unit tests and the five workspace structure checks passed.

V0 is complete. V1's camera-equipped FR3 scene/scripted expert and V2's local
process bridge have since been implemented. See
[V2 protocol and verification](v2_local_sim.md). The pretrained native action
semantics still have not been adapted to the FR3 task; V0 inference remains
separate from physical simulator command execution.
