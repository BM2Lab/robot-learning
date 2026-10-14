# Pinned SmolVLA inference environment

Prepared 2026-10-13. Keep ML inference separate from the pinned Isaac runtime and
from physical-control environments. Adjacent repositories need not share a venv.
The interface between processes will be versioned observations/actions, not
cross-environment Python imports.

## Baseline

| Component | Pin |
|---|---|
| Python | 3.12 (workstation interpreter 3.12.12) |
| LeRobot | 0.6.1, upstream tag v0.6.1 |
| Upstream source | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| Policy | lerobot/smolvla_base |
| Policy revision | 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 |
| VLM/tokenizer | HuggingFaceTB/SmolVLM2-500M-Video-Instruct |
| VLM/tokenizer revision | 7b375e1b73b11138ff12fe22c8f2822d8fe03467 |
| PyTorch / torchvision | 2.10.0+cu128 / 0.25.0+cu128 |
| Transformers | 5.5.4 (resolved lock) |
| NumPy | 2.2.6 (resolved lock) |

`requirements/learning.in` selects the inference baseline; the adjacent
`learning-py312-cu128.lock` pins all resolved packages and hashes. It targets
Linux x86_64/Python 3.12, not every host. CUDA libraries are bundled wheel
dependencies; no system CUDA toolkit or driver update is required by installation.
Only the SmolVLA extra is selected: no hardware, dataset/training or Arena extras.
Add and re-lock those deliberately when their milestones arrive.

The release's Transformers requirement differs from the existing Isaac stack;
do not merge this lock into the simulator environment. Install upstream LeRobot,
not a fork. Configuration in `configs/policies/smolvla_base.json` pins both model
repositories. Future loaders must honor those revisions, including nested
VLM/tokenizer loads: a top-level policy revision alone is insufficient.

## Reproduce in an independent environment

From the repository root, set `LEARNING_VENV` and `LEARNING_CACHE` to local ext4
locations. On this workstation the intended environment is workspace
`state/venvs/learning`, activated through the repository `.envrc`.

```bash
uv venv --python python3.12 --seed "$LEARNING_VENV"
uv pip sync --python "$LEARNING_VENV/bin/python" \
  --require-hashes --index-strategy unsafe-best-match \
  --cache-dir "$LEARNING_CACHE" requirements/learning-py312-cu128.lock
uv pip install --python "$LEARNING_VENV/bin/python" --no-deps \
  --cache-dir "$LEARNING_CACHE" -e packages/contracts -e packages/learning
uv --cache-dir "$LEARNING_CACHE" pip check --python "$LEARNING_VENV/bin/python"
"$LEARNING_VENV/bin/python" scripts/verify_learning_env.py --check-cuda
```

Only two official indexes are used: PyPI and the PyTorch CUDA 12.8 wheel index.
`unsafe-best-match` allows resolution across both because the PyTorch index has
stale copies of unrelated packages (e.g. requests). Exact hashes bound the
installation; do not add untrusted extra indexes. Hashes protect artifact identity,
not an assertion that every upstream package has been security-audited.

Inspect then authorize `.envrc` with `direnv allow` for normal shell usage.
`HF_HOME` is configured there to a local model cache, outside Git. GPU checks
must run on a GPU-visible host, not a restricted container lacking device access.

## Current verification boundary

The verifier imports LeRobot/SmolVLA and local contracts/learning, checks isolation
from Isaac, and optionally exercises CUDA and the pinned policy configuration.
`--check-model-config` downloads only config.json, not model weights. Full model
loading, saved tokenizer/processor execution and synthetic action inference are
now verified by the separate V0 smoke script. FR3 adaptation remains a later gate.
The base checkpoint declares 6-D state/actions and three cameras;
it is not an FR3-ready policy or a hardware deployment approval.

Workstation verification on 2026-10-13 passed: hash-locked dependency installation,
dependency consistency (81 installed distributions including editable local
packages and bootstrap pip), SmolVLA imports, CUDA matrix calculation on an RTX
4090, pinned configuration parsing/hash validation, and both local package tests.
The existing Isaac environment still reports Transformers 4.57.6, Isaac Lab
6.1.14 and Isaac Sim 6.0.1.0; its packages were not modified.

V0 model/inference details and commands are in [v0_smolvla.md](v0_smolvla.md).

References: [release](https://github.com/huggingface/lerobot/releases/tag/v0.6.1),
[release dependencies](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml),
[policy snapshot](https://huggingface.co/lerobot/smolvla_base/tree/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6).
