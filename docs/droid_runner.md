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
| `scripts/run_droid_runtime.py` | one recorded episode. The token comes from a `chmod 600` file. Output goes through `validate_artifact_path` (`--smoke` for /tmp, otherwise under `LAB_DATA_ROOT`; on a single-disk host set `LAB_DATA_ROOT_ON_SYSTEM_DISK=1`). |

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

Startup: the runner waits up to 5 s after `start` for the first `valid` state sample,
because native feedback is invalid until the first native callback, and never acts on an
invalid sample. Each episode's `session` event records the robot-learning and sensors git
revisions (commit and dirty flag) next to the runtime versions, policy pin and camera
receipts. `scripts/summarize_droid_episode.py <episode_dir> [--candidate-limits <physical
config>]` reports:
- clock uncertainty and startup wait
- state-to-camera skew
- policy and capture-to-receipt latency
- accepted → applied delay
- row-0 lead and row-to-row speed
- rejection causes, the fault and the outcome
- how many accepted chunks the physical candidate's acceptance bounds would have rejected

## Shadow mode (D7) and replanning

`--replan-after-rows` now defaults to 5 of 8 rows. The cross-host mock measured capture →
receipt at p95 110 ms, which leaves about 90 ms of slack (6 rows left about 18 ms).

`--shadow` runs the full pipeline against a runtime that evaluates but never executes:

- **The runtime's `hello` must report a backend name starting with `shadow`.** The runner
  refuses a mismatch in either direction: `--shadow` against a backend that can move the
  robot, or no `--shadow` against a shadow backend.
- **The shadow runtime never sends `applied`.** That is franka-runtime's
  `SessionAuthority(expect_application=False)`: offline shadow never confirms
  application. The runner therefore times replanning from the `accepted` receipt and
  records `chunks_accepted` and `"shadow": true`.
- **What the runner expects from a franka-runtime shadow backend (to be built on the RT
  host; this is D7 and needs authorization):**
  - **State:** read-only FCI state is streamed as `FrankaStateSample`, with the native
    validity rules.
  - **Checks:** every acceptance and workspace check runs against the real measured q.
  - **Execution:** the executor runs virtually, so would-be targets, holds and underruns
    are logged; no command or gripper action is sent.
- **Faults:** the runner now polls receipts during the replan wait, so a fault latched
  mid-chunk ends the episode promptly.

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
- toward the setpoint, the mock rate- and acceleration-limits its kinematic arm. The D6
  native candidate instead runs continuous 1 kHz hybrid joint impedance (DROID FR3 gains)
  with no trajectory generator; measured-speed and torque guards fault and stop. Mock
  acceptance rates therefore do not predict physical ones: compare against the
  physical candidate's bounds with `scripts/summarize_droid_episode.py --candidate-limits`.

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
