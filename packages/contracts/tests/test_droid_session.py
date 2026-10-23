from dataclasses import replace
from pathlib import Path

import pytest

from contracts.clock import ClockOffsetEstimate, ClockProbe
from contracts.droid import DroidActionChunk, DroidIOSpec
from contracts.droid_session import (DroidSessionOpen, DroidSessionReady, capability_id, chunk_envelope,
                                     validate_chunk_command)
from contracts.episodes import SampleContext
from contracts.sessions import LeaseRequest, SessionLease, payload_sha256

ROOT = Path(__file__).resolve().parents[3]
HOME = (0., 0.27, 0., -2.03, 0., 2.35, 0.08)
OFFSET = 9_000_000_000  # runtime clock = client clock + OFFSET


@pytest.fixture
def spec():
    return DroidIOSpec.from_json((ROOT / "configs/contracts/droid_fr3_jointpos_v1.json").read_text())


@pytest.fixture
def session(spec):
    request = DroidSessionOpen(spec, payload_sha256(spec),
                               LeaseRequest("req", "fr3_01", "runner", "rt:mono", OFFSET, 60*10**9), "client:mono")
    lease = SessionLease("fr3_01", "s1", "e1", "runner", "l1", "rt:mono", OFFSET, OFFSET+60*10**9)
    ready = DroidSessionReady(lease, request.contract_sha256, capability_id(spec), 0.08, 200_000_000, 2_000_000,
                              {"franka-runtime": "0.0.4", "libfranka": "0.21.2"})
    probe = ClockProbe("client:mono", "rt:mono", 0, 100_000+OFFSET, 120_000+OFFSET, 220_000)
    return request, ready, ClockOffsetEstimate.from_probes([probe], max_uncertainty_ns=2_000_000)


def chunk(spec, step=3, captured=2_000_000_000):
    context = SampleContext("episode", step, captured, "monotonic")
    return DroidActionChunk(context, spec.action, (*HOME, 0.), ((*HOME, 0.),)*8, captured+533_333_333)


def test_open_ready_roundtrip_and_fingerprint(spec, session):
    request, ready, _ = session
    assert DroidSessionOpen.from_json(request.to_json()) == request
    assert DroidSessionReady.from_json(ready.to_json()) == ready
    ready.validate_for(request)
    with pytest.raises(ValueError, match="fingerprint"):
        replace(request, contract_sha256="0"*64)
    with pytest.raises(ValueError, match="different robots"):
        replace(request, lease_request=replace(request.lease_request, robot_id="fr3_02"))
    with pytest.raises(ValueError, match="distinctly"):
        replace(request, client_clock_id="rt:mono")
    for bad in (replace(ready, gripper_open_width_m=0.0795), replace(ready, capability_id="droid-jointpos-v1:x"),
                replace(ready, lease=replace(ready.lease, owner_id="other"))):
        with pytest.raises(ValueError):
            bad.validate_for(request)


def test_chunk_envelope_converts_to_runtime_time(spec, session):
    request, ready, estimate = session
    value = chunk(spec)
    envelope = chunk_envelope(ready.lease, 1, value, estimate, client_now_ns=2_080_000_000,
                              capability=ready.capability_id)
    assert envelope.source_observation_ns == 2_000_000_000+estimate.offset_ns
    assert envelope.deadline_ns == value.valid_until_ns+estimate.offset_ns-estimate.uncertainty_ns
    wire = DroidActionChunk.from_json(value.to_json())  # chunk hash must survive JSON transport
    validate_chunk_command(spec, envelope, wire, lease=ready.lease, estimate=estimate,
                           now_ns=envelope.created_ns+10, next_sequence=1, capability=ready.capability_id)


def test_runtime_rejects_tampered_late_or_mismatched_commands(spec, session):
    _, ready, estimate = session
    value = chunk(spec)
    envelope = chunk_envelope(ready.lease, 1, value, estimate, client_now_ns=2_080_000_000,
                              capability=ready.capability_id)
    ok = dict(lease=ready.lease, estimate=estimate, now_ns=envelope.created_ns, next_sequence=1,
              capability=ready.capability_id)
    other = DroidActionChunk(value.observation, value.action_spec, value.source_state,
                             ((*HOME[:6], 0.1, 0.),)*8, value.valid_until_ns)
    cases = [
        (envelope, other, {}),  # payload swapped
        (envelope, value, {"now_ns": envelope.deadline_ns}),  # expired in runtime time
        (envelope, value, {"next_sequence": 2}),  # replay/out of order
        (replace(envelope, deadline_ns=envelope.deadline_ns+1), value, {}),  # deadline extended
        (replace(envelope, source_step=4), value, {}),
        (envelope, value, {"estimate": replace(estimate, offset_ns=estimate.offset_ns+1)}),
        (envelope, value, {"estimate": replace(estimate, server_clock_id="other")}),
    ]
    for env, payload, change in cases:
        with pytest.raises(ValueError):
            validate_chunk_command(spec, env, payload, **{**ok, **change})
    full = DroidActionChunk(value.observation, value.action_spec, value.source_state, ((*HOME, 0.),)*15,
                            value.valid_until_ns)
    with pytest.raises(ValueError, match="horizon"):
        validate_chunk_command(spec, chunk_envelope(ready.lease, 1, full, estimate, client_now_ns=2_080_000_000,
                                                    capability=ready.capability_id), full, **ok)
    sim = DroidActionChunk(SampleContext("episode", 3, 2_000_000_000, "simulation"), value.action_spec,
                           value.source_state, value.values, value.valid_until_ns)
    with pytest.raises(ValueError, match="monotonic"):
        chunk_envelope(ready.lease, 1, sim, estimate, client_now_ns=2_080_000_000, capability=ready.capability_id)
