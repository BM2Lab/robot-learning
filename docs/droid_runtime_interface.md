# DROID runtime interface — franka-runtime

**Status: the message contracts are implemented (D3, `contracts` 0.0.2) and the
runtime side runs against mocks (D4, franka-runtime 0.0.4 `franka_runtime.droid`,
see its `docs/droid-runtime.md`). D6 adds an offline-tested native candidate on the RT
host (franka-runtime `docs/droid-inference-workstation-handoff.md`); it is not
qualified for motion.** Where D4 refined this
proposal, the refinement is marked **D4**. This specifies what the learning side will send to
and expect from franka-runtime for zero-shot `pi05_droid_jointpos`. See
[droid_handoff.md](droid_handoff.md) for the roadmap.
Where it conflicts with franka-runtime's AGENTS.md or qualification procedure,
those win. Native timing, guards and stopping are franka-runtime's design.

## Roles

- **franka-runtime** has final authority. It owns the lease, execution, the
  gripper, all physical guards, faults and stopping. It may reject or stop at any
  time, and the learning side cannot override, resume or reset a latched fault.
- **learning runner** owns observation capture, inference and chunk submission.
  It reads robot state from the runtime and never commands the robot any other way.
- Reuse franka-runtime's existing `SessionAuthority` model: lease
  acquire/start/release, monotonically sequenced command envelopes,
  accepted/applied/rejected receipts, latched faults, and `finish_stop` evidence.

## What the policy produces (fixed by D0)

From `contracts.droid.DroidActionChunk`:

- Up to 8 rows (`open_loop_horizon`) at 15 Hz, a 66.7 ms row period.
- Each row is **absolute** FR3 joint targets `panda_joint1..7` in radians, then a gripper closure in [0, 1].
- The chunk carries its source capture context (episode, step, timestamp, clock),
  the joint state used for inference, and `valid_until_ns`.
- The learning side has already enforced finiteness, closure range and
  `|target − source state| ≤ max_deviation_from_source_rad` (0.6) on every row.

These are **targets, not velocities or deltas**. They must never be reinterpreted.

## Required runtime behavior

### Session

1. **Open.** The runner sends the exact `DroidIOSpec` and its canonical SHA-256;
   the runtime accepts only a configured, allow-listed contract ID and fingerprint.
   The reply includes the unit identity, the runtime and native artifact versions, the
   limits in force and the clock information below.
2. **Lease.** One lease at a time. Submissions outside an active lease are rejected.
3. **Release or stop.** Releasing, disconnecting, missing heartbeats or any fault
   discards queued rows and leads to the runtime's own stop. Sessions are never resumed.

### State stream (runtime → runner)

At ≥ 50 Hz with a bounded queue (latest-wins on the runner side), one sample contains:

- measured `q` (7, rad, `panda_joint1..7` order) and `dq` (7, rad/s)
- Franka Hand measured width (m) and a moving/grasped flag
- runtime timestamp (its CLOCK_MONOTONIC, ns), sample sequence number, validity flags
- active lease, the chunk and row currently executing, and the fault state

The runner builds observations from the **latest** sample. Its capture context
records both the camera time and the state sample's runtime timestamp.

### Chunk submission and execution

1. **Accept or reject at submission.** The runtime repeats every contract check and
   adds its own:
   - FR3 joint limits with margin
   - per-row implied velocity, `|q_{k+1} − q_k| / 66.7 ms`, under configured limits
   - the first row against the **current** measured `q` (not only the source state).
     **D5/D4 decision:** setpoint lead. Row 0 may lead the measured q by at most
     `max_target_lead_rad`. **D6:** there is no trajectory generator. The native 1 kHz
     hybrid joint-impedance controller pulls toward the setpoint, with explicit Franky slew
     and libfranka filtering/rate limiting; measured-speed and raw-torque guards fault and stop
   - workspace and self-collision checks if available
   - a deadline that is still in the future after clock conversion

   Any failure rejects the whole chunk with a reason in the receipt. Nothing is clipped.
   **D4:** the joint-range, implied-speed and measured-q checks run at submit and again at
   dispatch. Workspace and self-collision checks are not implemented. A rejected command
   does not consume its sequence number, so the runner resubmits with the same sequence.
2. **Replacement.** One pending chunk at most. A newly accepted chunk replaces the
   rows not yet started of the current one at the next row boundary. It never
   interleaves with or rewinds the current chunk.
3. **Execution.** The executor switches row targets at 15 Hz. **D6 (native candidate):**
   continuous 1 kHz hybrid joint impedance on all seven joints,
   τ = (K_q + JᵀK_xJ)(q_d − q) − (D_q + JᵀD_xJ) q̇, plus Coriolis (gravity from firmware),
   with a desired velocity of zero and no stop at each row. Gains are the pinned DROID FR3
   defaults. Franky slew and libfranka filtering/rate limiting are explicit. There is
   **no** Ruckig trajectory generator and no promised acceleration/jerk-limited
   interpolation. The runtime bounds motion with a measured joint-speed guard and a
   raw-torque guard (both fault and stop, they do not clamp), plus the row checks above.
   The PPO route (6 velocities, J7 held) is not used.
4. **Underrun.** When rows run out before a new chunk arrives, hold the last target
   for at most a configured `hold_timeout_ms`, then perform the runtime's own stop.
   The start of a hold, its duration and any stop are reported in receipts.
   **D4:** the hold target is the last row whose start time passed. A hold timeout
   latches fault `native_fault:101`; a heartbeat timeout latches 102; a client
   disconnect while running latches 103.
5. **Expiry.** No row starts after `valid_until_ns`, converted to runtime time with
   its uncertainty subtracted. Expiry is treated as an underrun.
6. **Receipts.** Each chunk gets accepted or rejected (with reason). **D4:** "applied"
   is per chunk, issued when its first row starts within the deadline, with the
   native tick and runtime time. A chunk replaced or expired before starting is
   rejected as superseded once a later chunk applies. Per-row progress and holds are
   reported in the state stream (`executing_sequence`/`executing_row`, `valid`), and
   stops through the latched fault, not as separate receipts.

### Gripper

- Command only on closure transitions across 0.5: "close" (grasp with the
  configured force and speed) or "open" (move to the configured open width).
  Franka Hand motions are slow and blocking, so they run asynchronously from arm
  control and never stall the arm executor.
- Report the measured width continuously. The runner maps it to closure with
  `FrankaHandClosureMap` (`1 − width / open_width_m`). Commissioning must record
  the homed `open_width_m`.
- The gripper force, speed and the "grasped" criterion are runtime configuration
  recorded in the session reply.

## Timing

| Quantity | Value or proposal |
|---|---|
| Policy rate | 15 Hz, 8 executed rows per chunk (533 ms) |
| Inference latency, capture → chunk | 64–98 ms measured on the inference host, no network |
| Inference mode (first version) | synchronous: capture, infer, submit; submit the next chunk before the current one finishes |
| Runtime command watchdog | the existing PPO configs use 50 ms against 50 Hz commands. Rows arrive in 533 ms chunks, so the watchdog must supervise the native executor and hold, **not** chunk arrival. |
| `hold_timeout_ms` | to be chosen by the runtime; proposal 200 ms, then stop |

## Clocks

Each host stamps with its own CLOCK_MONOTONIC. Proposal for D3:

1. **Offset estimation.** At session open and then every second, the runner and
   runtime exchange timestamped probes in NTP style: offset = midpoint, uncertainty = RTT/2.
   The probes double as the heartbeat. The runner forwards its completed probes, and
   the runtime re-derives and registers the estimate with `ClockOffsetEstimate.from_probes`.
   Keep the samples with the minimum RTT and reject the session if the
   uncertainty exceeds a bound (proposal 2 ms on a dedicated link).
2. **Conversion.** The runtime converts `valid_until_ns` to its clock with
   the uncertainty subtracted. Receipts carry runtime time. The runner logs both.
3. **Optional.** chrony or PTP on the dedicated link reduces drift. It does not
   replace the per-session estimate, because CLOCK_MONOTONIC is never disciplined.

## Transport

- A TCP stream on the dedicated link only, with the runtime listening on that
  interface and not on any other.
- Framing: reuse `contracts.wire` (length-prefixed JSON, 2 MiB cap, duplicate-key
  and NaN rejection). Chunks are small, and images never cross this link.
- Authentication is still to be decided in D3/D4. As a minimum, an allow-listed
  peer address plus a session token from runtime configuration; physical
  isolation is assumed but not sufficient.

## Contracts (D3, implemented in `contracts` 0.0.2)

All are stdlib-only, frozen, JSON round-trippable and validated on construction.

| Module | Types | Role |
|---|---|---|
| `contracts.sessions` | `LeaseRequest`, `SessionLease`, `LeaseRelease`, `CommandEnvelope`, `CommandReceipt`, `FaultReport`, `payload_sha256` | generic lease/command/receipt/fault records. Field order matches franka-runtime's `SessionAuthority` call sites. Its unmodified `tests/unit/test_authority.py` (11 tests) passes against this module, with only `contracts.motion`/`contracts.franka_policy` stubbed |
| `contracts.franka_state` | `FrankaStateSample` | the state stream: q, dq, finger width, gripper flags, validity, lease, executing chunk/row, fault code; runtime clock |
| `contracts.clock` | `ClockProbe`, `ClockOffsetEstimate` | NTP-style offset, minimum-RTT selection, bound on uncertainty, conservative deadline conversion |
| `contracts.droid_session` | `DroidSessionOpen`, `DroidSessionReady`, `capability_id`, `chunk_envelope`, `validate_chunk_command` | session handshake and chunk ↔ envelope binding |

Binding rules enforced by `validate_chunk_command` (runtime side, before any physical check):

- the chunk matches the contract's action spec and has at most `open_loop_horizon` rows
- the chunk's clock domain is monotonic, and the estimate maps into the lease clock
- envelope lease, clock, sequence (no replay or reordering), capability and
  payload hash (SHA-256 of the chunk's canonical JSON) all match, and runtime `now`
  lies in `[created, deadline)`
- the source step and source time equal the chunk's capture, converted with the
  registered estimate, and the envelope deadline is no later than the chunk's
  conservatively converted deadline

`DroidSessionReady.validate_for` makes the runner refuse a session whose lease,
contract fingerprint or capability differ from what it asked for, or whose homed
gripper width differs from the contract's `open_width_m`. Updating the contract
config is then a deliberate act. Capability IDs are `droid-jointpos-v1:<sha256 of the DroidIOSpec>`.

Not in D3: the transport (op names and framing on top of `contracts.wire`), the
release and heartbeat message wrappers, and authentication. These belong to D4,
together with the runtime behavior above.

## Shadow mode (D7, gated)

The runtime connects to the robot **read-only** and streams real state. The full
pipeline runs, and the runtime executes everything except sending commands:
acceptance checks, replacement, timing, would-be targets and implied
velocities are all logged. The success criteria are agreed before running: no
unexpected rejections, the deviation and velocity distributions, and latency and
clock uncertainty within bounds.
