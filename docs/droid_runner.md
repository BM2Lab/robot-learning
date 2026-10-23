# D5 — DROID runner and franka-runtime client

D5 closes the loop on the inference host:
- measured robot state from a franka-runtime DROID service
- live cameras
- the pinned openpi policy server
- chunk submission
- a recorded episode

It talks to robots only through franka-runtime. Against the D4 mock
(`franka-droid-mock`) this is the loopback test. Against a real runtime it is
subject to that runtime's commissioning and authorization.

## Pieces

| Module | Role |
|---|---|
| `learning.integrations.franka_runtime.client` | `DroidRuntimeClient`, which speaks franka-runtime's DROID protocol using contracts only (never imports `franka_runtime`). Open sequence: `hello` (contract and robot must match) → clock probes → `open` (verified with `DroidSessionReady.validate_for`) → `clock_register` (the runtime must derive the same estimate) → `start`. A background heartbeat sends probes every 0.25 s. `submit` advances the sequence only on acceptance. |
| `learning.rollouts.droid_runner` | `DroidRunner` (episode loop) and `DroidEpisodeRecorder` (JSON lines + summary) |
| `scripts/run_droid_runtime.py` | one recorded episode. The token comes from a `chmod 600` file. Output goes through `validate_artifact_path` (`--smoke` for /tmp, otherwise under `LAB_DATA_ROOT`). |

## Loop

1. Poll receipts. A latched runtime fault ends the episode, with no release and no retry.
2. Observe:
   - **state:** the latest `FrankaStateSample`; it must be `valid`, no older than 100 ms after clock conversion, and within 100 ms of the camera capture
   - **cameras:** both frames, routed as in D1
   - **packet:** measured q and Franka Hand width → closure
3. Infer. A contract rejection (`ChunkRejected`) is recorded and nothing is submitted.
4. Submit. A runtime rejection is recorded, and the next attempt reuses the sequence
   number. More than `--max-consecutive-rejections` in a row ends the episode.
5. Wait for this submission's `applied` receipt. The runtime drains receipts in
   issue order and accepts a sequence once, so only receipts after its `accepted`
   one count; earlier rejections of the same sequence number are ignored.
6. Replan `--replan-after-rows` rows (default 6 of 8) after the chunk started, so
   the next chunk replaces the remaining rows before they run out.
7. At the step budget, let the last rows finish, release, and wait for `complete`.

Each `step` record holds:
- the observation packet (including its 224² images)
- the state sample, camera timing and state-to-camera skew
- the policy audit, chunk, envelope and receipt

Separately the episode records the session (`hello`, `DroidSessionReady`, the clock
estimate, the policy pin identity, camera receipts), every drained receipt, any
fault, and the summary. About 0.4 MB per step.

## Loopback verified 2026-10-23 (inference workstation, no robot)

| Component | Detail |
|---|---|
| Policy server | D2 launcher: verified identity, warmup 8.4 s |
| Runtime | franka-runtime D4 `franka-droid-mock` |
| Cameras | both live RealSense cameras |
| Runner | the openpi-droid venv |

Results:
- The full session ran. Five chunks were applied, and the mock arm followed the policy
  for up to 0.25 rad per joint, so the policy saw its own actions through the cameras.
  Policy round trip was about 63 ms; state-to-camera skew was at most 30 ms.
- Twelve submissions were rejected. Eleven failed the runtime's speed check from the
  **measured** q to row 0: the policy's first target led the measured state by up to
  0.25 rad, which is 3.8 rad/s over one 66.7 ms row against the mock's 1.0 rad/s.
  The twelfth was refused because the fault had already latched.
- With no accepted chunk for longer than the horizon plus the 200 ms hold, the
  runtime latched `native_fault:101` (underrun) and stopped. The runner ended
  `runtime_fault` without releasing. Both behaved as designed.
- A first loopback found a runner bug: earlier rejections of a reused sequence number
  were read as rejection of its later accepted submission. It was fixed, and the
  unit-test fake now emits rejected receipts the way the real authority does.

Tests:
- `test_droid_runner.py`: runner logic with fakes and an injected clock.
- `test_franka_runtime_client.py`: end-to-end against the real mock. Runs only when
  `FRANKA_RUNTIME_CHECKOUT` and `FRANKA_DROID_MOCK_BIN` are set; otherwise skipped,
  because there is no sibling-checkout assumption.

## Row-0 semantics: setpoint lead (decided 2026-10-23)

DROID's joint-position actions were setpoints for a joint-impedance controller, so they
can lead the measured state. The first loopback used franka-runtime's original strict
rule (measured q → row 0 within one row at the velocity limit) and most chunks were
rejected. franka-runtime now uses **setpoint lead**:
- row 0 may lead the measured q by at most `max_target_lead_rad`;
- consecutive rows stay under the velocity limit;
- the motion generator limits velocity and acceleration toward the setpoint (the mock
  now, Franky in D6).

Loopback in setpoint mode, with the mock starting at the robot's observed home pose:

| Result | Value |
|---|---|
| Steps | 30 |
| Chunks applied | 29 |
| Rejected | 1 (joint 6 lead 0.316 rad > mock 0.3) |
| Row-0 lead | median 0.11 rad, maximum 0.30 rad |
| Ending | clean release, no fault |

The mock arm drifted far (+1.9 rad joint 2, +2.0 rad joint 4), because the cameras do
not see the mock arm and the scene had no task objects. **Workspace and self-collision
checks in franka-runtime are required before evaluating motion in shadow mode or
moving the robot.**
