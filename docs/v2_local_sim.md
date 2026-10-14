# V2 — Local simulator/learning process bridge

V2 connects the existing camera-equipped FR3 task to the **separate learning
Python environment**. It verifies a bounded mock Cartesian policy, not a
pretrained VLA reaching policy. No hardware runtime, dataset, training job or
additional dependency installation is involved.

## Ownership and boundaries

| Owner | Implementation |
|---|---|
| `contracts` | `samples.py`: portable observation/RGB capture schemas; `wire.py`: bounded JSON framing codec |
| `isaac-tasks` | `bridges/action_queue.py`: authoritative chunk execution; `local_session.py`: episode/step identity and Gym resets; `local_server.py`: private Unix socket endpoint |
| `learning` | `integrations/isaaclab/local_client.py`: typed client; `policies/mock_cartesian.py`: bounded test policy; `rollouts/local_smoke.py`: finite acceptance tests |
| Entry points | Simulator `scripts/serve_visual_reach.py`; learning `scripts/run_local_sim_smoke.py` |

The simulator imports **contracts, not learning**. The client/mock rollout uses
only standard-library Python and contracts; it imports neither Isaac nor the ML
stack. Existing SmolVLA inference remains independent. Its native 6D action
output cannot be sent to the FR3 3D physical action contract.

## Session semantics

1. `hello` requires exact typed policy I/O contract equality and canonical SHA-256.
   The response records runtime versions, model provenance, timestep and session ID.
   The fingerprint hashes the normalized typed contract, not raw JSON file bytes.
2. `observe` returns seven measured joints, language and two losslessly encoded
   uint8 HWC RGB images. Each image carries its logical optical frame, frame
   counter and the same capture context as state. No target coordinates or
   privileged expert state are sent. Terminal metrics are separate from policy input.
3. `submit` accepts at most one pending action chunk. It must reference the exact
   current capture context and match action units, bounds, frames and chunk size.
4. `advance` consumes **one** queued physical action and steps the simulator once
   (20 Hz policy period / ten physics substeps). Every row is applied relative to
   the current measured TCP at that execution step, not an accumulated reference.
5. Each row is checked again for episode identity, consecutive step index,
   simulation-clock compatibility and deadline. Expired/invalid queued execution
   clears the queue; stale RPC requests never advance simulation.
6. Explicit `reset` and Gym's automatic terminal reset create a new episode ID
   and discard all pending rows. Auto-reset responses contain old-episode terminal
   metrics and a **new-episode** observation. Reset does not restart global time.
7. `close`, disconnect, idle timeout or fatal physics failure clears the queue and
   closes the simulation session. A disconnected session cannot be resumed.

Capture context is `(session UUID + episode counter, episode step, timestamp_ns,
clock='simulation')`. Timestamp is an integer global policy tick × 50,000,000 ns.
Action deadlines use **simulation time**; socket deadlines use wall time. An
empty queue pauses simulation—it does not invent a zero action or continue physics.
There is no interpolation, temporal ensemble or low-level smoother in this bridge.

This is a **synchronous, single-environment** milestone. Inference and serialization
pause physics; 20 Hz is a simulated rate, not a wall-clock latency guarantee.
Base64 RGB copies occur at the process boundary, not the physics servo loop.
Shared-memory images, asynchronous inference, recording/export and wall-clock
deployment semantics remain later work.

## Transport

One local Unix-domain socket; no TCP listener, network API or pickle. Frames use
a four-byte big-endian length followed by UTF-8 JSON, capped at 2 MiB. Images are
raw uint8 RGB bytes encoded in base64; geometry/byte counts are validated. NaN,
duplicate JSON keys, oversized/truncated frames and duplicate request IDs are rejected.

The socket parent must be owned by the user with mode 0700; the socket uses 0600.
Choose a short path on a **POSIX/ext4 runtime filesystem**, not NTFS, source or
dataset storage. Existing endpoints are never overwritten. Only the endpoint
created by this process is removed on shutdown. This is a same-user local trust
boundary, not authentication against processes running as that same user.

## Workstation commands (two terminals)

Terminal A starts Isaac using its own direnv settings:

```bash
cd /home/chen-lab/isaac/lab-workspace/isaac-tasks
export ISAAC_TASKS_FR3_USD=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data/generated_assets/fr3v21_bare_flange_live_v1/fr3v2.usda

direnv exec . python -u scripts/serve_visual_reach.py \
  --contract ../robot-learning/configs/contracts/fr3_visual_reach_v1.json \
  --socket /home/chen-lab/isaac/lab-workspace/state/run/v2-demo/bridge.sock \
  --device cpu --viz none --idle-timeout 300 \
  --test-episode-length-s 0.2
```

Wait for `V2 simulation ready`. Terminal B uses the isolated learning environment:

```bash
cd /home/chen-lab/isaac/lab-workspace/robot-learning
direnv exec . python -u scripts/run_local_sim_smoke.py \
  --contract configs/contracts/fr3_visual_reach_v1.json \
  --socket /home/chen-lab/isaac/lab-workspace/state/run/v2-demo/bridge.sock
```

The short episode override deliberately induces a timeout so automatic queue
clearing can be checked quickly. That timeout is **expected**, not a reaching
failure of a trained model. Remove the override for the normal ten-second task.
Use `--viz kit` in Terminal A for a visible session; simulation only advances
when the client requests a step. The finite client closes the server normally.
If connection times out, start a new server; do not reuse an orphan endpoint.

The mock sends two small alternating Z increments (±1 mm); it is a transport
test, **not** a goal-reaching expert. Default smoke reports print to stdout; no
dataset or image archive is silently created. Runtime socket files live in state,
not in the repositories. Paths/interpreters are host setup examples, not Python
constants; another host supplies its compatible environments, asset and contract.

## Verified 2026-10-14

- 32 contracts/learning tests and 15 Isaac task tests passed; all five workspace
  source checks passed.
- Two real isolated Python processes exchanged q7 and both 256×256 RGB images.
- Six mock policy steps with a forced automatic timeout/reset completed; manual
  reset also discarded a partially consumed chunk.
- Old-episode chunk, stale step, duplicate request and expired execution were
  rejected without unintended stepping. Empty queue stayed paused.
- Camera frame counters advanced; canonical contract fingerprint and runtime/model
  receipts were returned. The learning process imported no Isaac/ML modules.
- Clean close removed the socket. A separate idle-timeout test exited nonzero and
  removed its endpoint.

Next gate: demo recording/LeRobot export and task-compatible policy adaptation.
Provision usable `LAB_DATA_ROOT` storage before collecting datasets; the existing
bulk volume is not assumed to have enough capacity.
