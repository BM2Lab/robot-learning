# D0 — native openpi DROID joint-position contract

`contracts.droid` declares the I/O of the pretrained `pi05_droid_jointpos`
checkpoint for a zero-shot FR3 + Franka Hand deployment. It is a distinct
native contract, like `contracts.libero`; it is **not** the FR3
`CartesianDeltaSpec` and does not add any hardware command path. No policy client,
runner, network service or runtime integration is implemented yet.

## Upstream pins

| Item | Pin |
|---|---|
| openpi | `215abfb217dbac7d5f1273282331b9b1866c0479`, config `pi05_droid_jointpos_polaris` |
| Checkpoint | `gs://openpi-assets-simeval/pi05_droid_jointpos` |
| Norm stats | `assets/droid/norm_stats.json`, SHA-256 in `configs/policies/pi05_droid_jointpos.json` |
| Reference client | Isaac Lab Arena 0.3.1 `isaaclab_arena_openpi/policy/droid_adapter.py`, embodiment `droid_abs_joint_pos` |

## Semantics

- **State** (8): seven measured joint positions in radians, `panda_joint1..7`
  order, then gripper closure in [0, 1] (0 open, 1 closed). openpi concatenates
  `observation/joint_position` and `observation/gripper_position` in that order.
- **Images**: `observation/exterior_image_1_left` and `observation/wrist_image_left`,
  uint8 RGB HWC, 224x224 **after** openpi `resize_with_pad`. Raw capture
  geometry, drivers and exposure belong to the sensors layer.
- **Actions**: chunks of 15 rows at 15 Hz. Each row is an **absolute** joint
  target in radians (the server applies `AbsoluteActions` to its trained
  deltas) plus gripper closure. Do not clip joints to [-1, 1]; that is
  upstream's joint-velocity example. `open_loop_horizon` rows are executed
  per inference (config: 8, openpi's DROID real-robot default; Arena uses 15).
- **Deviation bound**: each joint target must stay within
  `max_deviation_from_source_rad` of the joint state captured with the source
  observation, which the chunk carries. Out-of-bound chunks are rejected, never
  clipped. The configured 0.6 rad sits just outside the training q01/q99
  (about ±0.53 rad). It is a semantic sanity check, **not** a safety guard:
  joint, velocity and workspace limits remain franka-runtime authority.
- **Gripper**: DROID was trained with a Robotiq 2F-85. `FrankaHandClosureMap`
  declares closure = 1 − width/`open_width_m` (clipped) and a binary command
  that closes above 0.5, as upstream does. Grasp force/speed and execution are
  runtime concerns. `open_width_m` 0.08 is nominal; replace it with the homed
  width of the commissioned hand.
- **Timing**: acceptance requires the same episode and clock domain, a current
  step at or after the source step, and a current time inside
  [capture, deadline). Cross-host clock mapping between the inference machine
  and the real-time host is not defined yet and belongs to a sessions contract.

`robot_id` `fr3_01` in `configs/contracts/droid_fr3_jointpos_v1.json` is a
placeholder until franka-runtime commissioning records the unit identity.

## Known transfer gaps

The checkpoint saw Panda + Robotiq data with DROID camera placement. FR3 joint
limits, the Franka Hand, a webcam exterior view and a RealSense D435 wrist view
are all out of distribution. Zero-shot success is unqualified.
