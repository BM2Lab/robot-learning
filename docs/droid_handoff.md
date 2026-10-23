# DROID zero-shot deployment — handoff and roadmap

Status as of 2026-10-23. Goal: run the pretrained openpi `pi05_droid_jointpos`
policy zero-shot on a Franka FR3 with a Franka Hand, using robot-learning for
the policy side and franka-runtime for the robot side. Read this first; the
runtime contract proposal is [droid_runtime_interface.md](droid_runtime_interface.md).

## Architecture

```
Inference host (RTX 5090, cameras)                                      RT host (RT kernel, FCI)
  sensors: D435 exterior + D435i wrist ─┐
  openpi server (pinned, 127.0.0.1) ◄───┤ learning runner ◄── dedicated ethernet ──► franka-runtime
                                        └─ contracts.droid chunks                    executor, gripper,
                                                                                    guards, stop authority
```

| Repo | Owns | Depends on |
|---|---|---|
| robot-learning `contracts` | versioned I/O semantics (stdlib only) | — |
| robot-learning `learning` | openpi client, camera routing, runner, recording, evaluation | contracts |
| sensors (BM2Lab, new, not yet on GitHub) | camera drivers, capture timing | contracts |
| franka-runtime | robot session, executor, gripper, guards, stop/fault authority | contracts only |
| openpi (upstream) | model, server, preprocessing | used as a library at a pinned commit |
| USRobot | ultrasound project, strictly downstream | robot-learning (migration pending) |

The policy server binds localhost on the inference host. Only the runner talks to
the RT host. The runtime never imports `learning`; the learning side never
overrides runtime stops.

## Done

| Step | What | Doc |
|---|---|---|
| D0 | `contracts.droid`: native DROID I/O, absolute joint targets + gripper closure, Franka Hand closure map, deviation bound | [droid_jointpos.md](droid_jointpos.md) |
| D1 | openpi client, camera routing, live smoke with no robot | [openpi_droid_client.md](openpi_droid_client.md) |
| D2 | standalone pinned openpi server: verified checkpoint, identity metadata, warmup | [openpi_server.md](openpi_server.md) |
| D3 | `contracts` 0.0.2: sessions (franka-runtime-compatible), FR3 state sample, cross-host clock offset, DROID session/chunk binding | [droid_runtime_interface.md](droid_runtime_interface.md#contracts-d3-implemented-in-contracts-002) |
| sensors | RealSense color driver, latest-frame reader, live preview | sensors `docs/realsense.md` |

Measured on the inference host: capture → validated chunk 64–98 ms, model about
47 ms, 50/50 chunks accepted against a verified server. All of this ran with a
static joint fixture. **No robot has been connected or commanded.**

## Next steps and gates

Steps marked **gate** need explicit operator authorization and follow
franka-runtime's `docs/hardware-qualification.md`. Nothing before D7 touches hardware.

| Step | Work | Where | Done when |
|---|---|---|---|
| D4 | Network service, 7-joint position chunk executor, Franka Hand backend, all against mock backends | franka-runtime (can be developed on any machine) | mock tests for every rule in the interface spec |
| D5 | Runner: state + cameras → policy → chunk submit; episode recording | robot-learning `learning` | loopback on the inference host against the D4 mock service |
| D6 | Native integration: pinned Franky/libfranka backend, 1 kHz timing, jitter, link latency, clock offset | franka-runtime **on the RT host** | timing report; watchdog margins measured |
| D7 **gate** | Shadow mode: read-only robot state, full pipeline, commands logged but not applied | both hosts | shadow run report: deviations, would-be velocities, rejections |
| D8 **gate** | Hold/stop and gripper commissioning per unit | RT host | franka-runtime qualification evidence |
| D9 **gate** | Slow, workspace-limited zero-shot trials, every episode recorded | both hosts | recorded episodes with outcome accounting |

Develop D4–D5 off the RT host and test with mocks. Deploy pinned commits to the RT
host. Do not edit code there, and keep IDEs, agents and GPU jobs off it while it controls the robot.

## Environments on the inference workstation

| Path | Content | Rebuild from |
|---|---|---|
| `/shared/projects/state/venvs/openpi-droid` | client: contracts, learning, sensors, openpi-client, pyrealsense2 (numpy 1.26) | [openpi_droid_client.md](openpi_droid_client.md#environment-openpi-droid-profile) |
| `/shared/projects/state/venvs/openpi-server` | openpi at 215abfb, JAX 0.5.3 | [openpi_server.md](openpi_server.md#build-and-fetch) |
| `/shared/projects/state/src/openpi` | upstream checkout, detached at the pinned commit | same |
| `/shared/projects/state/cache/openpi` | verified checkpoint, config assets, tokenizer | `scripts/fetch_openpi_checkpoint.py` |

These paths are host examples. `.envrc` still assumes a `../direnv/python.envrc`
that does not exist under `/shared/projects`; set the variables explicitly
instead. The SmolVLA/LIBERO `learning` environments are separate (numpy 2) and
are not needed for DROID.

Smoke the current pipeline (two terminals): start the server per
[openpi_server.md](openpi_server.md#serve), then run `scripts/smoke_openpi_droid.py`
per [openpi_droid_client.md](openpi_droid_client.md#live-smoke-no-robot). Close
RealSense Viewer first; a camera can be opened by only one process.

## Hardware facts

| Item | Value |
|---|---|
| Robot | FR3 with Franka Hand. Unit identity, firmware and limits are not yet recorded (`robot_id` `fr3_01` is a placeholder). |
| Exterior camera | RealSense D435, librealsense serial `021222073194`, firmware 5.17.3.10 |
| Wrist camera | RealSense D435i, serial `153222071583`, firmware 5.17.3.10; not yet mounted |
| Camera mode | 640x360@30 RGB (16:9, matches DROID's aspect before letterboxing) |
| Inference GPU | RTX 5090, 32 GB; one openpi server uses about half |

## Known issues and open decisions

1. **Contracts versioning.** `contracts` is now 0.0.2 (`learning` 0.0.2 and sensors
   pin it). It implements `contracts.sessions` compatibly with franka-runtime's
   authority. franka-runtime on GitHub still pins `contracts==0.0.3` and also imports
   `contracts.motion` and `contracts.franka_policy` (PPO/M4), which do not exist. D4
   must repin franka-runtime to 0.0.2 and either implement those two modules in
   `contracts` (a later version) or move them into franka-runtime. isaac-tasks
   similarly pins 0.0.1 while importing newer modules. 0.0.2 was not labelled 0.0.3
   so that it does not falsely claim to satisfy that pin.
2. **Cross-host clocks.** D3 defines the probe and estimate contracts and the
   conservative deadline conversion. The probe exchange, its rate and the uncertainty
   bound in practice are D4/D6 work (measure on the real link).
3. **Deviation bound.** 0.6 rad over 8 executed rows. A frozen fixture produced
   up to 0.68 rad (one rejection in 50). Re-evaluate with moving state in D7.
4. **Embodiment gap.** DROID is Panda + Robotiq 2F-85 with ZED cameras. FR3
   limits, the Franka Hand and the camera placement are out of distribution.
   Measure the hand's homed width (`open_width_m`, currently nominal 0.08).
5. **Camera image quality.** Auto white balance gave dim, green images in office
   lighting. Tune exposure and white balance after mounting.
6. **Placement of Isaac Lab Arena baselines** is undecided; they stay in USRobot for now.
7. **USRobot migration** to robot-learning's server and client is deferred.
8. **sensors** exists only locally (`/shared/projects/sensors`), and its GitHub
   visibility is undecided. robot-learning's client environment installs it from a checkout path.
9. **Policy server on the network.** It binds 127.0.0.1. Only the runner should
   cross the robot link; decide authentication for the runtime service (D3/D4).
