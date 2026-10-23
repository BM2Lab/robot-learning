# D2 — standalone pinned openpi policy server

robot-learning serves openpi baselines from its own pinned setup: an upstream
openpi checkout at the pin's commit in a configured path **outside this
repository**, an environment built from openpi's own `uv.lock`, and a verified
checkpoint cache. Nothing is vendored or forked; upstream is used as a library.
No downstream project (e.g. USRobot) is required.

## Pieces

| Item | Where |
|---|---|
| Policy pin | `configs/policies/pi05_droid_jointpos.json`: openpi commit, train config, checkpoint URL plus tree digest, train-config assets, tokenizer |
| Verification (stdlib) | `learning.integrations.openpi.checkpoint`: openpi cache path mapping, `tree_digest`, `verify_artifacts`, `server_identity` |
| Fetch | `scripts/fetch_openpi_checkpoint.py`: downloads with openpi's `maybe_download`, then verifies |
| Serve | `scripts/serve_openpi_policy.py`: verifies, warms up, serves with identity metadata |

`tree_sha256` hashes the sorted list of relative paths and file SHA-256s; file
order and timestamps do not matter, symlinks are rejected. Building the train
config makes openpi load the config's own norm stats
(`gs://openpi-assets/.../pi05_droid_jointpos_polaris/assets/droid`); they are
pinned as `config_assets` so the launcher never downloads. They are byte-identical
to the checkpoint's norm stats, which take precedence for inference.

## Paths

All locations are explicit. The workstation follows this repository's `.envrc`
convention, `<workspace>/state/`; other hosts choose their own:

| Variable | Workstation example | Content |
|---|---|---|
| `OPENPI_CHECKOUT` | `/shared/projects/state/src/openpi` | upstream clone, detached at the pinned commit |
| `OPENPI_SERVER_VENV` | `/shared/projects/state/venvs/openpi-server` | openpi's locked environment (Python 3.11, JAX) |
| `OPENPI_DATA_HOME` | `/shared/projects/state/cache/openpi` | verified checkpoint, config assets, tokenizer (~12 GB) |
| `UV_CACHE_DIR` | `/shared/projects/state/cache/uv` | package cache (~8 GB, may be pruned) |

Environments store absolute paths: rebuild rather than move them.

## Build and fetch

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_CHECKOUT"
git -C "$OPENPI_CHECKOUT" checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479

# openpi's own frozen lock; submodules (aloha, libero) are not needed for serving
(cd "$OPENPI_CHECKOUT" && UV_PROJECT_ENVIRONMENT="$OPENPI_SERVER_VENV" UV_LINK_MODE=copy \
  GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen)

# the stdlib verifier and contracts, without dependencies
uv pip install --python "$OPENPI_SERVER_VENV/bin/python" --no-deps \
  -e packages/contracts -e packages/learning

mkdir -p "$OPENPI_DATA_HOME"
"$OPENPI_SERVER_VENV/bin/python" scripts/fetch_openpi_checkpoint.py \
  --policy-pin configs/policies/pi05_droid_jointpos.json --data-home "$OPENPI_DATA_HOME"
```

The fetch prints `"verified": true` or fails on any tree, norm-stats, config-asset
or tokenizer mismatch. `--report-only` prints found digests for recording a new pin.

## Serve

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 "$OPENPI_SERVER_VENV/bin/python" scripts/serve_openpi_policy.py \
  --policy-pin configs/policies/pi05_droid_jointpos.json --data-home "$OPENPI_DATA_HOME" --port 8000
```

Before listening, the launcher:

1. requires the imported openpi to come from a clean checkout at the pinned commit,
2. verifies every cached artifact (it never downloads),
3. loads the policy through upstream `policy_config.create_trained_policy`,
4. runs one zero-input warmup inference (the first call JIT-compiles, ~8 s on an
   RTX 5090), so the first real chunk can meet its deadline,
5. serves upstream `WebsocketPolicyServer` with `server_identity(pin)` as metadata.

It binds `127.0.0.1` unless `--host` is given (upstream `serve_policy.py` always
binds `0.0.0.0`). The D1 client refuses a server whose metadata differs from the
pin; upstream `serve_policy.py` (empty metadata) is admitted only with
`allow_unverified_server`, recorded as unverified. Stop the server with Ctrl-C to
free the GPU. One server uses about half the GPU memory with the fraction above.

## Verified 2026-10-23 (workstation, clean build)

- Fresh clone at `215abfb`; `uv sync --frozen` produced JAX 0.5.3 with the RTX 5090
  visible.
- Fresh download (about 18 min at ~12 MB/s) reproduced the pinned tree digest:
  26 files, 12,435,136,033 bytes, matching the digest first computed from a
  separately downloaded copy, plus the norm-stats, config-asset and tokenizer hashes.
- Launcher verification passed, made no downloads, warmed up in 8.2 s and served
  on 127.0.0.1 only.
- D1 live smoke against it: server identity verified, 50/50 chunks accepted,
  capture → chunk 64–98 ms (median 81 ms), policy `infer_ms` median 47 ms.
- Without warmup, the first chunk took 8.4 s and correctly failed acceptance as expired.
