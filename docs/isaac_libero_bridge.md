# P2 — native LIBERO features from Isaac Lab

This optional bridge uses `contracts.libero.LiberoIOSpec` and a separate simulator
process. It is **not** the FR3 visual-reaching contract or a hardware interface.
No simulator dependencies are installed in the learning environment. P2's checks
send only neutral actions. P3's bounded pretrained rollout and observed failure
to demonstrate task transfer are documented in [p3_isaac_libero_rollout.md](p3_isaac_libero_rollout.md).

`learning.integrations.isaaclab.libero` contains:

- `LiberoSimulatorClient`: exact native contract handshake over existing private
  local RPC, with backend/orientation checks.
- `native_features(packet)`: native 8D state, language and decoded named RGB HWC.
  Pixels already match LIBERO **environment-processor output orientation**.
  Never run that environment image flip again. Checkpoint policy pre/postprocessing
  is still required; `SmolVLAInference.predict_native` provides it for future P3.
- `bounded_chunk(packet, raw_actions, execute_steps=1)`: explicitly clips native
  controller inputs to [-1,1], reports saturation, and binds execution to the
  exact capture with a simulation-clock deadline. It never converts to FR3 meters.

Position is native grip-site/world position; orientation is the native
`robot0_right_hand` body quaternion converted to rotation vector **without**
quaternion sign canonicalization; finger positions retain signed meter units.
This mixed native frame convention is deliberate. Two RGB views are 256x256.
See isaac-tasks' `docs/p2_libero_visual.md` for camera mapping and visual gaps.

## Run the finite IPC smoke

Prepare/qualify a P2 scene and start `scripts/serve_libero.py` in isaac-tasks,
using its direnv and Isaac Python. Use an explicit private socket directory
on a local filesystem, mode 0700. For the bounded terminal test:

```bash
# In isaac-tasks; explicit paths must already be set.
direnv exec . "$ISAAC_PYTHON" scripts/serve_libero.py \
  --scene-manifest "$SCENE_MANIFEST" --visual-reference "$NATIVE_REFERENCE" \
  --socket "$PRIVATE_SOCKET" --max-steps 3 --device cpu --viz none
```

In robot-learning's separate terminal/environment:

```bash
direnv exec . "$LEARNING_PYTHON" scripts/check_libero_bridge.py \
  --socket "$PRIVATE_SOCKET" --output-dir /tmp/libero-p2-client
```

The client records initial images, typed receipt, neutral-action audit, reset
invalidation and terminal capture, then closes the server. Outputs must be new.
The smoke is bounded to at most three steps. Simulation pauses during IPC or
inference; this is synchronous, not real-time control. A horizon has `success:
null` because P2 has no native task-success predicate. Do not count it as failure
or success of SmolVLA. Reset always replays the exported fixed fixture; a different
seed does not select a different native initial-state row.

Verified 2026-10-15: actual socket smoke at `/tmp/libero-p2-client-01`, three
neutral steps and clean close, **zero model actions**. State/camera checks and
paired native/Isaac renders are at `/tmp/libero-p2-visual-final`. Camera geometry
matches; images still differ in lighting/shading/texture projection, so successful
pretrained policy transfer is not yet demonstrated. Historical franka-rl stays
unchanged.
