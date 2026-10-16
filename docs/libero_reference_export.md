# P0 native LIBERO reference export

This is the native oracle for the forthcoming Isaac port, **not** a benchmark,
policy rollout, converted asset bundle or hardware deployment. It does not load
SmolVLA weights. Keep using the isolated, pinned LIBERO evaluation environment,
not Isaac's Python. Existing model-pin runtime versions and immutable asset
revision are checked before any native simulator import; no downloads are made.

From robot-learning, activate its direnv and explicitly choose LIBERO Python:

```bash
direnv exec . "$LIBERO_PYTHON" scripts/export_libero_reference.py \
  --policy-pin configs/policies/smolvla_libero.json \
  --runtime-dir "$LIBERO_RUNTIME_DIR" \
  --output /tmp/libero-p0-reference
```

Set `LIBERO_PYTHON` to the qualified isolated interpreter, and
`LIBERO_RUNTIME_DIR` to the existing qualified cache documented in
[native LIBERO](native_libero.md). Output parent must already exist, have at
least 100 MiB free, and the output itself must not exist. A failed partial export
is not complete and will not be overwritten. `/tmp` is appropriate for this
bounded development reference; persistent experiments need an explicit mounted
artifact root. The complete export is approximately 140 KiB, not an asset copy.

Fixed reference: `libero_object`, task 0 (soup in basket), seed 42,
initial-state index 0. Outputs:

- `model.xml`: actual native MuJoCo XML; mesh references still require the pinned
  asset cache, so do not treat this as a standalone transferred scene.
- `reference.json`: runtime/asset/controller/source receipts, selected initial
  state and digest, model mass/inertia/COM and joint metadata, nine native OSC
  oracle cases with expected goals, null-space matrices and raw/clipped torques.
  P1 additionally records settled body poses/joint states, passive parameters
  and three FK/Jacobian/mass/bias probes, including moving joints. These are
  same-state references, not trajectories; regenerate older P0-only exports
  before building an Isaac scene. The extended export remains below 200 KiB.
  P2 adds camera OpenGL extrinsics/fovy, the native orientation-body FK chain,
  cached reset and fresh same-state policy vectors, plus two processed PNG views.
  PNG hashes are recorded; these additional files increase the total export size.
- `receipt.json`: SHA256 of the reference file. Only a complete valid JSON
  reference plus matching receipt should be consumed.

Cases start from an actual reset snapshot; moving, singular, 180-degree and
coupled-mode cases deliberately modify inputs. The exporter calls upstream
robosuite `set_goal` / `run_controller` with acquisition overridden, rather than
reimplementing the expected algebra. They are **not physical trajectories**.
The controller initial-joint anchor is captured independently from measured
post-settling joints. The seventh native action is gripper control, not an arm
rotation component. P1's Isaac scene qualification covers arm dynamics, gripper
open/close and support contact. P2 qualifies camera geometry, state semantics and
local IPC; image rendering is not pixel-identical. Actual grasp/place success
remains separate work. See isaac-tasks' `docs/p2_libero_visual.md` and
[learning-side P2 bridge](isaac_libero_bridge.md).

In isaac-tasks, compare without loading MuJoCo or Kit:

```bash
python scripts/validate_libero_osc.py --reference /tmp/libero-p0-reference/reference.json
```

That repository retains a portable lightweight subset as
`tests/fixtures/libero_osc_reference.json` for regression tests. Full reference
generation was verified on 2026-10-15 at `/tmp/lab-libero-p0-reference-04`;
earlier partial development outputs are not qualification evidence. All nine
float64 cases pass CPU and CUDA comparison. Numerical limitations, mathematical
contract and the next scene-validation gate are documented in isaac-tasks'
`docs/p0_libero_osc.md`. No historical franka-rl code is changed or imported.
