# robot-learning contributor instructions

## Ownership and dependency boundaries

This repository contains two separately installable distributions/imports:
`packages/contracts` (`contracts`) and `packages/learning` (`learning`).
Keep both in src layout. The repository root is not a third Python distribution.

`contracts` owns versioned robot, action, observation, policy, task, session and
episode semantics. It must not import or depend on Isaac, ROS, Franky/libfranka,
PyTorch, Transformers or LeRobot. Keep dependencies lightweight and serialization
portable. Specify joint order, units, frames, quaternion conventions, timestamps,
validity and sensor masks; reject ambiguous inputs rather than inventing defaults.

`learning` owns orchestration, recording/export, policy interfaces, learning
recipes and evaluation. Keep backend/ML integrations optional and lazily loaded.
Reuse pinned upstream libraries through adapters; do not silently fork them.
Do not implement vendor drivers or native servo loops in this repository.
Dependencies must remain acyclic: runtimes and simulation may depend on `contracts`,
but must not need the full `learning` distribution just to read a schema.

## Development and verification

Use a project-isolated environment, separate from simulator and hardware native
environments. Do not assume a sibling checkout, user home path, parent `AGENTS.md`
or a particular host layout. Use configurable paths and declared package dependencies.
Record compatible Python/library versions and code revisions before integrations.

Run package tests after changes:

```bash
python -m pytest packages/contracts/tests packages/learning/tests --import-mode=importlib
```

Test serialization/feature order, semantic version compatibility, optional import
isolation and dataset round trips as those features are added. Schema semantics
changes require versioning/migration, not just renaming fields.

Keep datasets, checkpoints, videos and raw logs outside Git. Require an explicitly
configured artifact location; validate availability/capacity and do not silently
fall back to the checkout. Buffer high-rate telemetry off the control hot path.
Store manifests, configuration, lightweight fixtures and deterministic splits in Git.

## Safety and current status

V0 standalone inference/contracts and V2 local simulation client/mock rollout
are implemented; see `docs/v0_smolvla.md` and `docs/v2_local_sim.md`.
V3 expert archives/LeRobot export are implemented; see `docs/v3_demonstrations.md`.
Native LIBERO L0–L2 uses an isolated optional MuJoCo integration; see
`docs/native_libero.md`. Preserve its native 8D EE/gripper state and 7D OSC action
semantics; do not adapt them by padding/relabeling the FR3 Cartesian contract.
Reuse upstream preprocessing/rollout and immutable model/asset pins. New package
versions require requalification, including the isolated hf-libero asset-cache hook.
L3 adds `evaluation/libero_suite.py`; see `docs/native_libero_suite.md`. Keep task
selection and seed/fixture pairs explicit and save first-episode videos regardless
of outcome. Never aggregate incomplete jobs into an apparently complete benchmark
score; preserve infrastructure-failure versus policy-failure accounting.
Preserve observation-before-action alignment, complete episode boundaries, held-out
episode seeds and training-only normalization statistics. Never substitute reset
observations for terminal samples or place target coordinates in policy features.
Other placeholders do not imply working integrations. Pretrained native VLA
actions are not task-compatible and must not be relabeled as physical commands.
Default tests/CI use mocks or simulation, never live robots. A learning supervisor
cannot override a native runtime's stop/fault authority. Hardware connection,
actuation, calibration and recovery require explicit separate authorization and
the selected runtime's qualification procedure.

D0 adds `contracts.droid` for the native openpi pi05_droid_jointpos I/O on FR3 +
Franka Hand; see `docs/droid_jointpos.md`. Actions are absolute joint targets plus
gripper closure, not velocities or FR3 Cartesian deltas. Reject, never clip,
out-of-bound joint targets; the contract bound is not a runtime safety guard.
D1 adds the openpi DROID client and camera routing in
`learning.integrations.openpi`; see `docs/openpi_droid_client.md`. It has no robot
path. Use the separate numpy<2 `openpi-droid` profile; learning must not import
sensors. Chunks contain only executable open-loop rows; record `ChunkRejected`,
never clip joints. Clients require the pinned server identity.
D2 serves openpi standalone: upstream checkout at the pinned commit in a configured
path outside this repo, openpi's own frozen lock, verified cache; see
`docs/openpi_server.md`. Never vendor/fork openpi or depend on a downstream project
(e.g. USRobot) for baselines. The launcher must not download; pin every artifact.

P2's optional Isaac/LIBERO bridge uses the distinct lightweight
`contracts.libero.LiberoIOSpec`, not the FR3 joint-state/Cartesian-delta schema.
See `docs/isaac_libero_bridge.md`. Preserve mixed native grip-site position and
right-hand-body orientation, original FK quaternion branch, and already-processed
image orientation. Do not run the LIBERO environment image flip twice. Clip raw
controller inputs explicitly with saturation audit and bind chunks to capture
identity/deadline. P2's neutral IPC smoke is not pretrained rollout or a success
benchmark; its terminal success is null until a task predicate is qualified.
