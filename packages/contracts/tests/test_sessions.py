from dataclasses import dataclass, replace

import pytest

from contracts.sessions import (CommandEnvelope, CommandReceipt, FaultReport, LeaseRelease, LeaseRequest,
                                SessionLease, payload_sha256)

LEASE = SessionLease("fr3_01", "s1", "e1", "runner", "l1", "rt:boot:mono", 1_000, 2_000)


def envelope(payload={"value": 1}, **changes):
    fields = dict(lease=LEASE, sequence=1, source_step=0, source_observation_ns=1_100, created_ns=1_150,
                  deadline_ns=1_500, capability_id="cap", payload_sha256=payload_sha256(payload))
    fields.update(changes)
    return CommandEnvelope(**fields)


def test_payload_hash_is_canonical_for_dicts_contracts_and_dataclasses():
    assert payload_sha256({"b": 1, "a": [1., 2]}) == payload_sha256({"a": [1., 2], "b": 1})
    assert payload_sha256(LEASE) == payload_sha256(LEASE.to_dict())

    @dataclass(frozen=True)
    class Spec:
        robot_id: str
    assert payload_sha256(Spec("r")) == payload_sha256({"robot_id": "r"})
    with pytest.raises(ValueError):
        payload_sha256({"x": float("nan")})


def test_positional_construction_matches_runtime_call_sites():
    request = LeaseRequest("req", "fr3_01", "runner", "rt:boot:mono", 1_000, 1_000)
    assert (request.request_id, request.robot_id, request.owner_id) == ("req", "fr3_01", "runner")
    e = CommandEnvelope(LEASE, 2, 1, 1_100, 1_100, 1_200, "cap", payload_sha256({}))
    assert (e.sequence, e.source_step, e.source_observation_ns, e.created_ns) == (2, 1, 1_100, 1_100)
    release = LeaseRelease(LEASE, 1_900, "complete")
    assert LeaseRelease.from_json(release.to_json()) == release


def test_lease_window_and_clock():
    LEASE.validate_at(1_000, "rt:boot:mono")
    for now, clock in ((999, "rt:boot:mono"), (2_000, "rt:boot:mono"), (1_500, "other")):
        with pytest.raises(ValueError):
            LEASE.validate_at(now, clock)
    with pytest.raises(ValueError):
        replace(LEASE, expires_ns=1_000)


def test_envelope_binding_rejects_each_mismatch():
    payload = {"value": 1}
    e = envelope(payload)
    assert CommandEnvelope.from_json(e.to_json()) == e
    good = dict(now_ns=1_200, clock_id="rt:boot:mono", next_sequence=1, capability_id="cap", payload=payload)
    e.validate_for(LEASE, **good)
    for change in ({"next_sequence": 2}, {"capability_id": "other"}, {"payload": {"value": 2}},
                   {"now_ns": 1_500}, {"now_ns": 1_149}, {"clock_id": "other"}):
        with pytest.raises(ValueError):
            e.validate_for(LEASE, **{**good, **change})
    with pytest.raises(ValueError):
        e.validate_for(replace(LEASE, owner_id="other"), **good)
    for bad in ({"created_ns": 1_000}, {"deadline_ns": 1_150}, {"sequence": 0}, {"payload_sha256": "ABC"}):
        with pytest.raises(ValueError):
            envelope(payload, **bad)


def test_receipts_and_faults_require_evidence():
    CommandReceipt("s1", "e1", "l1", 1, "accepted", "rt", 5, None, "")
    applied = CommandReceipt("s1", "e1", "l1", 1, "applied", "rt", 5, 7)
    assert CommandReceipt.from_json(applied.to_json()) == applied
    for bad in (("applied", None, ""), ("rejected", None, ""), ("queued", None, "x")):
        with pytest.raises(ValueError):
            CommandReceipt("s1", "e1", "l1", 1, bad[0], "rt", 5, bad[1], bad[2])
    fault = FaultReport("s1", "e1", "rt", 9, "lease_expired", "requested")
    assert replace(fault, stop_state="complete").stop_state == "complete"
    with pytest.raises(ValueError):
        replace(fault, stop_state="reset")
