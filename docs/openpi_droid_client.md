# D1 — openpi DROID client and live camera routing

D1 connects live cameras to an upstream openpi policy server and turns its replies
into contract-checked `DroidActionChunk`s (see [D0](droid_jointpos.md)). There is
**no robot path**: the smoke uses a static joint fixture, and nothing is sent to
franka-runtime. Robot state, chunk scheduling and stopping remain runtime work.

## Ownership

| Owner | Implementation |
|---|---|
| openpi (upstream, own env) | `scripts/serve_policy.py`, model, normalization, `resize_with_pad` |
| BM2Lab sensors (own repo) | RealSense drivers, `LatestFrameReader`, per-device configs |
| `learning.integrations.openpi.cameras` | `DroidCameraRouter`, `droid_packet`: frame routing to a validated `ObservationPacket` |
| `learning.integrations.openpi.droid` | `OpenPIDroidClient`, `request`, `bounded_chunk`, `ChunkRejected` |
| Entry point | `scripts/smoke_openpi_droid.py` |

`learning` does not import sensors: the router accepts any reader whose
`latest(max_age_ns)` returns a frame with `image`, `optical_frame`, `sequence`,
`capture_ns` and `sensor_monotonic_ns`. Package import loads neither openpi-client
nor numpy; both are imported inside functions.

## Semantics

- **Routing**: each contract camera key gets exactly one reader. A frame must carry
  the contract's optical frame (`exterior_optical`, `wrist_optical`), be younger
  than `max_age_ns`, and the two captures must be within `max_skew_ns`. Swapped,
  stale or skewed cameras raise; no frame is repeated or substituted.
- **Packet**: measured joints + Franka Hand width mapped to closure, images resized
  with upstream openpi `resize_with_pad` (16:9 → 224x224 letterbox, not stretch).
  The capture context's timestamp is the newest camera capture (CLOCK_MONOTONIC).
- **Request**: openpi DROID keys; joints and closure split from the packet state.
- **Reply**: a finite `(15, 8)` array is required. Only the first
  `open_loop_horizon` rows (8) form the chunk, because only those can execute.
  Gripper closure is clipped to [0, 1] and counted; joint targets are never
  clipped, so a row beyond the deviation bound raises `ChunkRejected` carrying the
  audit. The audit reports deviation over executed rows and the full chunk.
- **Deadline**: `capture + open_loop_horizon / 15 Hz` (533 ms). Late chunks fail
  `validate_for` at acceptance; the client itself has no inference watchdog
  (upstream `infer` blocks), so the runtime's own command timeout stays authoritative.
- **Connect**: a TCP probe fails fast if nothing listens; upstream's client would
  otherwise retry forever. The server's metadata must equal `server_identity(pin)`
  (openpi commit, train config, checkpoint URL and tree digest), which the
  [D2 launcher](openpi_server.md) publishes after verifying its artifacts. Upstream
  `serve_policy.py` publishes `{}` and is refused unless `allow_unverified_server`
  (`--allow-unverified-server`) is set; the receipt then records `server_verified: false`.

## Environment (`openpi-droid` profile)

openpi-client pins `numpy<2`, so this is a separate venv; never sync it into the
learning (numpy 2) or LIBERO profiles. Client dependencies (websockets, msgpack,
dm-tree, pillow, typing-extensions, numpy) are pinned to openpi 215abfb's `uv.lock`;
newer websockets deprecate the client's `connect()` use. typing-extensions is
imported by openpi-client but not declared upstream. Hash mode cannot cover the
git dependency, so install hashed packages first, then openpi-client at the pinned
commit without dependencies:

```bash
uv venv --python python3.12 "$OPENPI_DROID_VENV"
grep -v '^openpi-client @' requirements/openpi-droid-py312.lock > "$OPENPI_DROID_VENV/hashed.txt"
uv pip sync --python "$OPENPI_DROID_VENV/bin/python" --require-hashes "$OPENPI_DROID_VENV/hashed.txt"
uv pip install --python "$OPENPI_DROID_VENV/bin/python" --no-deps \
  "openpi-client @ git+https://github.com/Physical-Intelligence/openpi@215abfb217dbac7d5f1273282331b9b1866c0479#subdirectory=packages/openpi-client" \
  -e packages/contracts -e packages/learning -e "$SENSORS_CHECKOUT"
uv pip check --python "$OPENPI_DROID_VENV/bin/python"
```

## Live smoke (no robot)

Start the pinned server as described in [D2](openpi_server.md) (its own openpi
environment, verified cache, warmup, 127.0.0.1), then run the bounded client in
the openpi-droid venv. Stop the server afterwards to free the GPU.

```bash
# output dir must be new
"$OPENPI_DROID_VENV/bin/python" scripts/smoke_openpi_droid.py \
  --contract configs/contracts/droid_fr3_jointpos_v1.json \
  --policy-pin configs/policies/pi05_droid_jointpos.json \
  --exterior-camera "$SENSORS_CHECKOUT/configs/cameras/exterior_d435.json" \
  --wrist-camera "$SENSORS_CHECKOUT/configs/cameras/wrist_d435i.json" \
  --host 127.0.0.1 --port 8000 --instruction "put the cube in the bowl" \
  --steps 50 --output-dir "$RUN_DIR"
```

It writes `receipt.json` (camera receipts, policy pin, server metadata, summary),
`steps.json` (per-step camera/policy audits, acceptance, executed rows) and the
first step's 224x224 policy inputs. Contract rejections are recorded, not raised.

## Verified 2026-10-23 (RTX 5090 workstation, both cameras on USB 3.2)

Against the standalone D2 server (clean checkout, freshly downloaded checkpoint):
server identity verified; 50/50 chunks accepted; capture → chunk 64–98 ms
(median 81 ms); executed-row deviation up to 0.53 rad. In one 50-step run, one
chunk was rejected for 0.68 rad within the 8 executed rows; with the frozen
fixture this is an expected contract rejection, recorded rather than clipped.

Earlier run against upstream `serve_policy.py` (unverified identity):

- 50/50 chunks accepted. Camera capture → validated chunk 65–96 ms (median 79 ms);
  policy `infer_ms` median 47.5 ms. No gripper clipping.
- Joint targets from a static fixture moved up to 0.50 rad within the 8 executed
  rows and 0.81 rad over the full 15-row chunk. Before D1 truncated chunks to the
  executable rows, a chunk was rejected for full-chunk deviation above 0.6 rad.
  With a frozen fixture the model keeps extrapolating motion; a moving robot's
  measured state should keep deviations smaller, which must be checked in shadow mode.
- Row 0 always lands next to the source state, confirming absolute joint targets.
- Policy inputs were upright, RGB, letterboxed; both views looked dim/green under
  auto white balance in office lighting. Cameras were not yet mounted.
- 72 contracts/D1/D2-verifier tests pass in the openpi-droid venv (including a
  real openpi-client websocket round trip and identity refusal); sensors' 22 pass
  there too. 142 pass / 5 skip in the numpy-2 learning env, where openpi-client
  tests skip.

## Not implemented

Robot state source, franka-runtime transport/session, chunk execution, recording of
deployment episodes, asynchronous inference, and camera extrinsics/mounting.
