# robot-learning

Two separately installable src-layout packages live in this Git repository:
- packages/contracts: distribution and import contracts. No ML, Isaac, ROS,
  Franky or LeRobot dependency.
- packages/learning: distribution and import learning. Owns orchestration,
  recording, policy adapters, training recipes and evaluation; depends on contracts.

These short package names are intentional. Use isolated environments to avoid
unrelated packages with the same names. Do not install the full learning package
on a hardware host just to obtain contracts. The optional learning smolvla extra
pins upstream LeRobot 0.6.1; model revisions and the Linux/Python 3.12 CUDA lock are
documented in [docs/learning_environment.md](docs/learning_environment.md).
No LeRobot source or model weights are vendored in this repository.
See ../README.md for editable installation and source-only validation.

V0 is implemented: dependency-light typed contracts and a standalone pretrained
SmolVLA inference smoke, with no robot/simulator action mapping. See
[docs/v0_smolvla.md](docs/v0_smolvla.md) for the contracts, tests and smoke command.

V2 is implemented: a typed local simulator client and finite mock action-chunk
verification across isolated environments. See [V2 protocol and commands](docs/v2_local_sim.md).
This does not map pretrained SmolVLA native actions to the FR3 task.
