# P3 — pretrained Isaac/LIBERO rollout test

`scripts/rollout_isaac_libero.py` runs the pinned SmolVLA checkpoint through the
P2 paused simulator endpoint. It does not change the policy, OSC gains, scene,
or image calibration. The simulator and model remain in separate environments.

Start `isaac-tasks/scripts/serve_libero.py` with its Isaac direnv, the qualified
P2 `--scene-manifest`, `--visual-reference`, private `--socket`, `--max-steps 280`,
`--idle-timeout 300`, `--device cpu --viz none`. Then run in the isolated pinned
native-LIBERO evaluation Python environment (not Isaac Python):

```bash
"$LIBERO_PYTHON" scripts/rollout_isaac_libero.py \
  --socket "$PRIVATE_SOCKET" \
  --pin configs/policies/smolvla_libero.json \
  --runtime "$LIBERO_RUNTIME" \
  --output /tmp/libero-p3-rollout --seed 42 --smoke
```

All model dependencies must already be cached; this script never downloads.
Use a new output directory. Omit `--smoke` only with an explicitly provisioned
mounted `LAB_DATA_ROOT` and an output child of that root.

The runner executes the first action of each predicted 50-action chunk, then
captures a fresh observation and replans. Flow noise uses `seed + policy_step`.
Checkpoint pre/postprocessing is applied, but the already-oriented P2 camera
images are **not flipped again**. Native normalized controller inputs are
explicitly clipped with audit; they are not relabeled as physical increments.
The model is synchronous: simulated time pauses during inference and IPC.

Artifacts:

- `rollout.mp4`: agent view on the left, wrist view on the right, 20 simulated
  frames per second; initial observation plus every post-action observation.
- `steps.json`: source capture/state, sampling seed, raw/applied first action,
  saturation, inference latency, next state and terminal event.
- `manifest.json`: immutable pins, runtime versions, runner SHA, simulator/source
  receipts, terminal status, timing and completion. Exceptions retain partial
  telemetry and a failed manifest, and propagate a nonzero exit.

`complete` means the infrastructure reached the endpoint horizon, **not** that
soup was placed in the basket. `task_success` stays null: a native-equivalent
grasp/place predicate has not been qualified. This is one fixed Object-0 fixture;
changing the inference seed does not select another scene fixture. It is neither
a LIBERO benchmark nor evidence of generalization or hardware readiness.

## Workstation result (2026-10-15)

Bounded artifacts: `/tmp/libero-p3-rollout-01` (temporary, not archival).
Used `/tmp/libero-p2-cache-04/scene.json`, reference
`/tmp/libero-p2-native-03`, inference seed 42, exported fixture 0, unchanged
checkpoint revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.

| Check | Result |
|---|---|
| Applied policy steps / simulated time | 280 / 14.0 s |
| Infrastructure terminal | Horizon; clean client/server exit |
| Mean inference latency | 223 ms (not real-time) |
| Total wall time including model loading / recording | 114 s |
| Clipped executed action components | 231 / 1,960 (11.8%) |
| Video | `rollout.mp4`, 281 frames, 512×256, 20 fps, 14.05 s |
| Automated grasp/place success | Null; predicate unqualified |
| Visual inspection | Approaches basket and remains there; no successful soup pickup/place observed |

`contact_sheet.png` shows steps 0, 40, 80, 120, 160, 200, 240, 280 in row order,
two paired camera captures per row. The video and telemetry are the primary
evidence; a small set of stills is not an automated success test.

Regression checks: 110 contracts/learning tests, 35 isaac-tasks tests and five
workspace source checks passed. No controller, renderer, dependency or historical
Franka baseline changes were made.

**P3 status:** pretrained execution/recording tested; successful task transfer
not demonstrated. Do not report a LIBERO success rate from this run. Next qualify
the task-local native success predicate. The matched native control below is now
complete; it does not by itself attribute the gap solely to rendering.
Native `In` combines object contact with containment; a distance-only
proxy is not equivalent. Remaining candidates include visual appearance/domain
shift and contacts/grasping. The P2 camera/FK checks alone cannot isolate which
causes this behavior. Further tuning is separate work.

## Matched native control

`scripts/compare_libero_native.py` runs the **same** `predict_native` method,
checkpoint processors, FP32/CUDA inference, first-action execution and explicit
noise schedule in native LIBERO. It rejects mismatched model pins, reference
hashes, seed/horizon schedules and initial-state fixtures. The actual native
task/controller receipt must match the exported P2 reference. Native environment
preprocessing orients the native images once; Isaac already supplies that image
orientation. Neither path flips a processed image again.

```bash
"$LIBERO_PYTHON" scripts/compare_libero_native.py \
  --isaac-run /tmp/libero-p3-rollout-01 \
  --reference "$NATIVE_REFERENCE/reference.json" \
  --pin configs/policies/smolvla_libero.json \
  --runtime "$LIBERO_RUNTIME" \
  --output /tmp/libero-p3-native-control --smoke
```

Result: `/tmp/libero-p3-native-control-01`, including `manifest.json`,
`steps.json`, dual-view `rollout.mp4` and an inspected `contact_sheet.png`.

| Measure | Isaac/PhysX port | Native LIBERO/MuJoCo control |
|---|---:|---:|
| Exported fixture / noise seed | 0 / 42 + step | 0 / 42 + step |
| Policy steps executed | 280 | 239 |
| Outcome | Horizon; no pickup/place observed | Native success predicate true; pickup/place visible |
| Automated task success | Unqualified (null) | True |
| Simulated duration | 14.00 s | 11.95 s |
| Mean inference | 223 ms | 222 ms |
| Clipped action components | 231 / 1,960 (11.8%) | 175 / 1,673 (10.5%) |
| Video frames including reset | 281 | 240 |

The native reset observation exactly matches the cached exported native state.
Its maximum difference from the Isaac initial 8D observation is `1.44735e-4`
(mixed state units), consistent with the already-documented cached/fresh reset
residual. First raw actions differ by up to `0.08543` in normalized controller
input units. Images differ substantially in material/texture appearance despite
camera geometry matching. These first-action differences occur **before** the
first policy action is executed, so subsequent contact dynamics cannot be their
cause; the differing image/state observations remain confounded.

**Conclusion:** the explicit-noise inference/processor loop can solve this fixed
fixture in native LIBERO. It is not universally broken. The Isaac run demonstrates
a transfer gap, not successful zero-shot task transfer. One paired episode cannot
estimate success rates or rule out stochastic sensitivity. Inspect material/UV
and appearance parity next (especially the basket and object textures), while
keeping controller parameters fixed; then separate observation effects from
physics using fixed-observation inference and action replay. No rendering,
controller or contact tuning was performed as part of this comparison.

Verification after adding the comparison guard tests: 117 contracts/learning
tests passed. The original 35 simulator tests remain unchanged and were not
rerun for this learning-only comparison.

## Frozen-input image/state intervention

`scripts/probe_libero_visual_inputs.py` isolates visual inputs **without physics**.
It runs all eight combinations of native/Isaac state, agent image and wrist image,
with noise seeds 42, 43 and 44. Model/processor history resets before every
condition. State and instruction stay fixed within each image comparison; noise
is paired by seed. Images are the lossless, already-oriented reset PNGs from the
qualified P2 capture, **not compressed frames extracted from rollout videos**.

```bash
"$LIBERO_PYTHON" scripts/probe_libero_visual_inputs.py \
  --images "$P2_VISUAL_CHECK_DIRECTORY" \
  --reference "$NATIVE_REFERENCE/reference.json" \
  --isaac-run /tmp/libero-p3-rollout-01 \
  --pin configs/policies/smolvla_libero.json --runtime "$LIBERO_RUNTIME" \
  --output /tmp/libero-p3-frozen-input --smoke
```

Result: `/tmp/libero-p3-frozen-input-01/report.json`. All 24 chunks were finite;
an identical-input repeat reproduced the entire native chunk exactly (max error
zero). At the **native fixed state**, mean differences over three noise seeds are:

| Agent image | Wrist image | First-action max absolute difference from native | Full-chunk RMS difference |
|---|---|---:|---:|
| Native | Native | 0 | 0 |
| Isaac | Native | 0.09487 | 0.06547 |
| Native | Isaac | 0.17163 | 0.09583 |
| Isaac | Isaac | 0.10915 | 0.11348 |

All values are unnormalized **native controller input units**, not meters/radians.
For comparison, replacing **only the state** with the slightly different Isaac
reset state gives mean first-action max difference `0.00656` and chunk RMS
`0.04409`. The state effect is small for the first action but not negligible over
the predicted chunk. The image effects interact; they are not additive.

For seed 42, native state and fixed instruction/noise:

| Agent image | Wrist image | First world-Y controller input |
|---|---|---:|
| Native | Native | -0.07449 |
| Isaac | Native | -0.14112 |
| Native | Isaac | +0.07389 |
| Isaac | Isaac | +0.01517 |

The soup is at world Y approximately -0.240 m, the basket at +0.252 m, and the
reset TCP at -0.006 m. Thus the wrist-only swap reverses the initial Y component
from the can's side toward the basket's side **with no dynamics involved**. This
is a directional component, not proof the complete action reaches either object.
Across seeds, the wrist swap shifts Y positive relative to its paired native
baseline, but does not reverse the sign for every seed.

**Interpretation:** visual inputs demonstrably cause a substantial initial-action
change, with wrist appearance a leading suspect. The test supports a visual
contribution to the apparent skipped pickup, not a literal task-sequencer bug.
It does not isolate materials from all other pixel differences, establish what
the model recognizes, or prove that rendering repair alone fixes the full task.
These P2 PNGs are separate captures: their seed-42 first actions differ from the
recorded P3 rollout first actions by up to 0.00536 (native) and 0.00733 (Isaac).
Therefore they are a controlled frozen-input probe, not exact replay of the
original two rollout observations.

Next inspect/repair wrist-view texture/material parity with the native reference,
keeping model, camera mapping and controller fixed, then rerun the frozen probe
and full rollout. No scene/controller changes were made in this test. All 118
contracts/learning tests passed after adding factorial-coverage verification.

## Render-only texture correction experiment

The material audit found forced sRGB decoding of explicitly linear native
textures, and a texture-origin mismatch: imported mesh UVs are preserved while
MuJoCo uploads the source image rows directly. The correction converts 2D texture
V coordinates with `(u, 1-v)` and preserves explicit linear sampling (`raw` in
USD). This changes material sampling, **not** the camera-image orientation.
Both corrections were tested together; their separate contributions have not
been ablated. Cube/procedural texture projection and lighting remain approximate.

All historical caches remain untouched. New artifacts:

- Scene: `/tmp/libero-p3-textures-cache-01/scene.json`.
- Qualified lossless paired renders: `/tmp/libero-p3-textures-visual-01`.
- Repeated 24-condition inference probe: `/tmp/libero-p3-textures-frozen-01`.
- Unchanged-model rollout: `/tmp/libero-p3-textures-rollout-01`.

The corrected paired renders show coherent can/basket texture detail. P2
state/camera/IPC checks passed, as did 37 simulator tests and five workspace
checks. A direct old/new USD comparison found 2,425 physical/geometry/transform
attributes unchanged. Model, controller, dynamics, camera poses and post-capture
image orientation were not changed.

Frozen-input mean **full-chunk RMS difference** from native over the same three
noise seeds, with native state fixed:

| Substituted Isaac views | Before | Corrected textures |
|---|---:|---:|
| Agent only | 0.06547 | 0.05033 |
| Wrist only | 0.09583 | 0.03936 |
| Both | 0.11348 | 0.05164 |

The both-view full-chunk difference falls about 54%, but this does **not** mean
task success improves by 54%. Mean first-action max difference for both views
actually increases from 0.10915 to 0.18440. Seed-42 first Y with both corrected
Isaac images remains basket-directed (+0.09809 at native fixed state). Identical
frozen-input repeats are still exact. Appearance repair improves one diagnostic
metric without consistently improving the initial action.

The full corrected-texture rollout completed 280 steps (281 video frames) with
mean inference 231 ms and 74 clipped action components out of 1,960 (3.8%).
**Visual outcome remains unsuccessful:** the gripper again goes directly into
the basket instead of picking up the soup, then remains in/around the basket.
The recorded contact sheet at steps 0/40/80/120/160/200/240/280 confirms this
behavior; `rollout.mp4` preserves the full episode. Automated success remains
null because the Isaac task predicate is still unqualified. All 118 learning
tests pass as well as the 37 simulator tests.

**Conclusion:** the identified texture errors are corrected and visual/chunk
agreement improves, but the apparent skipped pickup persists. Do not present
this as a successful sim transfer or a complete diagnosis. Remaining appearance
differences include illumination/contrast, wrist-visible robot materials and
room/cube projection. A useful next isolation is fixed-state native rendering of
Isaac robot/object poses (or a bounded matched-state observation sequence), to
separate remaining observation shift from closed-loop dynamics without tuning
the policy or controller. Another arbitrary lighting change would not by itself
isolate the cause.

## Moving matched-pose observation test

Three reusable scripts now separate observation shift from Isaac dynamics:

1. `robot-learning/scripts/export_libero_pose_sequence.py` replays the saved
   native successful actions and exports measured robot joints, all seven object
   poses, fresh native state and lossless RGB at steps 0, 20, 40, 80, 120, 160,
   200 and 239. Native replay again reports task success.
2. `isaac-tasks/scripts/render_libero_pose_sequence.py` writes those exact poses
   into the corrected-texture scene and renders them **without stepping Isaac
   physics** after initialization. It checks native versus Isaac state at every
   pose with the existing `2e-5` tolerance.
3. `robot-learning/scripts/probe_libero_pose_sequence.py` compares native and
   Isaac image predictions at each pose, using **the identical fresh native
   state, instruction and flow-noise seed `42 + native_step`** for both. History
   resets between queries. This is not a closed-loop manipulation success test.

The first capture attempt correctly failed strict state parity: MuJoCo's cached
Cartesian state was one integrator substep behind its exported joint positions
(max mismatch 0.000441). Explicit `sim.forward()` before native snapshot capture
resolved it. No tolerance was relaxed. The qualified pair's maximum 8D state
error is `4.77e-7`, over all eight poses. This identifies a cache-time alignment
issue in the diagnostic, not evidence of a controller failure.

Artifacts (bounded temporary checks):

- Native poses/images: `/tmp/libero-p3-pose-native-03/poses.json`.
- Isaac images/parity: `/tmp/libero-p3-pose-isaac-02/report.json`.
- Paired contact sheet: `/tmp/libero-p3-pose-isaac-02/paired_contact_sheet.png`.
- Inference comparison: `/tmp/libero-p3-pose-probe-01/report.json`.

The contact sheet has rows for steps 0/40/120/200 and columns **native agent,
Isaac agent, native wrist, Isaac wrist**. Geometry/camera alignment is close;
illumination, contrast, robot/gripper shading and room appearance still differ.

| Native step | Full-chunk RMS difference | First translation cosine | Native / Isaac first gripper input |
|---|---:|---:|---:|
| 0 | 0.0439 | -0.477 | -1.026 / -1.023 |
| 20 | 0.3726 | 0.997 | -0.989 / -0.996 |
| 40 | 0.2623 | 0.986 | -1.018 / -0.998 |
| 80 | 0.4069 | 0.259 | -1.001 / -1.002 |
| 120 | 0.2968 | 0.938 | +1.021 / -1.007 |
| 160 | 0.0257 | 0.997 | +1.001 / +1.010 |
| 200 | 0.4477 | 0.754 | -0.995 / +1.016 |
| 239 | 0.0243 | 0.991 | -1.006 / -1.003 |

Values are raw native controller inputs before clipping. Translation cosine
compares the first predicted XYZ vectors (1 aligned, 0 perpendicular, -1
opposed), not measured Cartesian velocities. Chunk RMS includes all 50 predicted
actions; only the first would be executed in the actual rollout.

**Finding:** residual image differences alone can alter motion direction **and
gripper intent** even at matched moving poses. At reset, native XYZ is
`[0.0651,-0.0658,0.0072]` while Isaac XYZ is `[0.0314,0.1045,-0.0002]`—toward
opposite Y sides, as in the failed rollout. At steps 120 and 200 the gripper
sign differs. At other poses (160/239), predictions agree much better. Thus the
visual transfer gap is state-dependent, not merely a constant action offset.
These differences cannot be attributed to differing Isaac dynamics in this test,
since no such dynamics were run. This does **not** prove every full-rollout
failure is visual or that contact physics already transfers.

Next prioritize a controlled native-light/robot-material appearance audit,
especially wrist-visible geometry, then repeat this matched-pose test before
another rollout. Do not change the controller or train around the rendering gap
yet. Offline regressions remain 118 learning/contracts and 37 simulator tests.

Example commands (use each repository's isolated Python; Isaac requires direnv):

```bash
# robot-learning
"$LIBERO_PYTHON" scripts/export_libero_pose_sequence.py \
  --native-run "$NATIVE_CONTROL_RUN" --reference "$NATIVE_REFERENCE/reference.json" \
  --pin configs/policies/smolvla_libero.json --runtime "$LIBERO_RUNTIME" \
  --output /tmp/new-native-poses --smoke

# isaac-tasks
direnv exec . "$ISAAC_PYTHON" scripts/render_libero_pose_sequence.py \
  --sequence /tmp/new-native-poses --scene-manifest "$CORRECTED_CACHE/scene.json" \
  --visual-reference "$NATIVE_REFERENCE" --output /tmp/new-isaac-poses \
  --device cpu --viz none

# robot-learning
"$LIBERO_PYTHON" scripts/probe_libero_pose_sequence.py \
  --sequence /tmp/new-native-poses --renders /tmp/new-isaac-poses \
  --pin configs/policies/smolvla_libero.json --runtime "$LIBERO_RUNTIME" \
  --output /tmp/new-pose-probe --smoke
```

## Lighting/material audit (2026-10-16)

The 50 named robot materials checked in the converted USD have diffuse RGB
values matching their exported MJCF definitions (maximum allowed error `1e-6`,
zero mismatches). Finger visuals without named materials retain their native
display-color values. This does not establish matching BRDF/specular response:
MuJoCo's Phong model and RTX's USD Preview Surface remain different.

Concrete lighting mismatch: no light prims were found in the converted room
USD. The Isaac profile therefore uses its 300-intensity uniform dome, while the
native compiled model contains two lights at `(1,1,4)` and `(-3,-3,4)` with
diffuse `(0.8,0.8,0.8)`, constant attenuation `(1,0,0)` and shadows disabled,
plus an active view-relative headlight (ambient 0.1, diffuse 0.4, specular 0.5).

Added an **opt-in diagnostic only** in isaac-tasks:
`assets/panda/libero_lighting_audit.py`, selected by
`scripts/render_libero_pose_sequence.py --lighting-audit`. It approximates the
constant-attenuation source directions using distant lights aimed at the
workspace origin, uses a weaker dome as ambient fill, and renders each view
separately with its own view-aligned headlight. Intensity scale 1000 is explicit
and recorded; it is not a unit conversion from native GL to physical USD light
units or a calibrated photometric match. Native per-light Phong specular response
is not reproduced. A renderer warm-up, explicit USD shadow API and fresh camera
readback are used before final capture. Default scene lighting is unchanged.

Final artifacts: `/tmp/libero-p3-lighting-poses-03` (paired renders and report),
`/tmp/libero-p3-lighting-probe-02` (paired inference). The former's
`wrist_comparison.png` has columns **native / dome baseline / lighting audit**,
rows at steps 0/120/200. Earlier `-01`/`-02` render captures are development
attempts and are not the final audit profile.

| Diagnostic over eight matched poses | Dome baseline | Lighting audit |
|---|---:|---:|
| Agent-view mean absolute pixel error (0–255) | 35.25 | 69.65 |
| Wrist-view mean absolute pixel error (0–255) | 40.12 | 74.24 |
| Mean full-chunk RMS action difference | 0.2350 | 0.2151 |
| Reset first-translation cosine vs native | -0.477 | 0.060 |
| Gripper signs match native at steps 120 and 200? | No | No |
| Maximum 8D state error | 4.77e-7 | 4.77e-7 |

**Decision:** do not promote this lighting approximation. Pixel agreement
worsens substantially, and the key grasp/release disagreements persist despite
an 8.5% decrease in average action-chunk difference. No new closed-loop success
claim or full rollout was made for this diagnostic. Policy, controller, physics,
camera poses, historical caches and the default lighting remain unchanged.

The next rendering step should calibrate the photometric pipeline (exposure,
display/color transform and BRDF/specular response) against fixed native RGB,
using explicit image-space metrics and a held-out pose subset—not select lights
by policy success or repeatedly guess intensity settings. Native GL light values
cannot simply be copied into RTX intensities. Offline regressions: 118
learning/contracts tests and 38 simulator tests pass.
