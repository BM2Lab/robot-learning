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
Preserve observation-before-action alignment, complete episode boundaries, held-out
episode seeds and training-only normalization statistics. Never substitute reset
observations for terminal samples or place target coordinates in policy features.
Other placeholders do not imply working integrations. Pretrained native VLA
actions are not task-compatible and must not be relabeled as physical commands.
Default tests/CI use mocks or simulation, never live robots. A learning supervisor
cannot override a native runtime's stop/fault authority. Hardware connection,
actuation, calibration and recovery require explicit separate authorization and
the selected runtime's qualification procedure.
