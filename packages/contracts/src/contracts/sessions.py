"""Versioned command-session semantics: leases, sequenced envelopes, receipts and faults.

Field order matches franka-runtime's SessionAuthority call sites (positional
construction). All times are integer nanoseconds in the clock named by ``clock_id``
(the runtime's clock for leases, envelopes, receipts and faults). These records
describe authority decisions; they do not grant it. The runtime remains final.
"""
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from typing import Optional
from ._validation import JsonContract, integer, text

RECEIPT_STATUSES = ("accepted", "rejected", "applied")
STOP_STATES = ("requested", "unavailable", "complete", "timeout")


def payload_sha256(value):
    """SHA-256 of canonical JSON (sorted keys, no NaN) of a contract, dataclass or JSON value."""
    if isinstance(value, JsonContract):
        value = value.to_dict()
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _sha256(value, field):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase hex SHA-256")
    return value


@dataclass(frozen=True)
class LeaseRequest(JsonContract):
    request_id: str
    robot_id: str
    owner_id: str
    clock_id: str
    requested_ns: int
    duration_ns: int

    def __post_init__(self):
        for field in ("request_id", "robot_id", "owner_id", "clock_id"):
            text(getattr(self, field), field)
        integer(self.requested_ns, "requested_ns")
        integer(self.duration_ns, "duration_ns", 1)


@dataclass(frozen=True)
class SessionLease(JsonContract):
    robot_id: str
    session_id: str
    episode_id: str
    owner_id: str
    lease_id: str
    clock_id: str
    issued_ns: int
    expires_ns: int

    def __post_init__(self):
        for field in ("robot_id", "session_id", "episode_id", "owner_id", "lease_id", "clock_id"):
            text(getattr(self, field), field)
        integer(self.issued_ns, "issued_ns")
        if integer(self.expires_ns, "expires_ns") <= self.issued_ns:
            raise ValueError("Lease must expire after it is issued")

    def validate_at(self, now_ns, clock_id):
        if clock_id != self.clock_id:
            raise ValueError("Lease clock domain mismatch")
        if not self.issued_ns <= integer(now_ns, "now_ns") < self.expires_ns:
            raise ValueError("Lease is not yet valid or has expired")


@dataclass(frozen=True)
class LeaseRelease(JsonContract):
    lease: SessionLease
    requested_ns: int
    reason: str

    def __post_init__(self):
        if not isinstance(self.lease, SessionLease):
            raise ValueError("Typed lease required")
        integer(self.requested_ns, "requested_ns")
        text(self.reason, "reason")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["lease"] = SessionLease.from_dict(data["lease"])
        return cls(**data)


@dataclass(frozen=True)
class CommandEnvelope(JsonContract):
    """One sequenced command. ``payload_sha256`` binds the separately sent payload."""
    lease: SessionLease
    sequence: int
    source_step: int
    source_observation_ns: int
    created_ns: int
    deadline_ns: int
    capability_id: str
    payload_sha256: str

    def __post_init__(self):
        if not isinstance(self.lease, SessionLease):
            raise ValueError("Typed lease required")
        integer(self.sequence, "sequence", 1)
        integer(self.source_step, "source_step")
        integer(self.source_observation_ns, "source_observation_ns")
        integer(self.created_ns, "created_ns")
        integer(self.deadline_ns, "deadline_ns")
        if not self.source_observation_ns <= self.created_ns < self.deadline_ns:
            raise ValueError("Require source observation <= creation < deadline")
        text(self.capability_id, "capability_id")
        _sha256(self.payload_sha256, "payload_sha256")

    def validate_for(self, lease, *, now_ns, clock_id, next_sequence, capability_id, payload):
        """Acceptance-time identity, order, time and payload binding; payload semantics are separate."""
        if self.lease != lease:
            raise ValueError("Envelope lease does not match the active lease")
        lease.validate_at(now_ns, clock_id)
        if self.sequence != next_sequence:
            raise ValueError("Out-of-order or replayed command sequence")
        if self.capability_id != capability_id:
            raise ValueError("Command capability mismatch")
        if payload_sha256(payload) != self.payload_sha256:
            raise ValueError("Payload does not match envelope hash")
        if not self.created_ns <= now_ns < self.deadline_ns:
            raise ValueError("Command is premature or expired")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["lease"] = SessionLease.from_dict(data["lease"])
        return cls(**data)


@dataclass(frozen=True)
class CommandReceipt(JsonContract):
    session_id: str
    episode_id: str
    lease_id: str
    sequence: int
    status: str
    clock_id: str
    timestamp_ns: int
    native_tick: Optional[int] = None
    reason: str = ""

    def __post_init__(self):
        for field in ("session_id", "episode_id", "lease_id", "clock_id"):
            text(getattr(self, field), field)
        integer(self.sequence, "sequence", 1)
        integer(self.timestamp_ns, "timestamp_ns")
        if self.status not in RECEIPT_STATUSES:
            raise ValueError(f"status must be one of {RECEIPT_STATUSES}")
        if self.native_tick is not None:
            integer(self.native_tick, "native_tick")
        if self.status == "applied" and self.native_tick is None:
            raise ValueError("Applied receipts require native tick evidence")
        if not isinstance(self.reason, str) or (self.status == "rejected" and not self.reason.strip()):
            raise ValueError("Rejected receipts require a reason")


@dataclass(frozen=True)
class FaultReport(JsonContract):
    session_id: str
    episode_id: str
    clock_id: str
    timestamp_ns: int
    code: str
    stop_state: str

    def __post_init__(self):
        for field in ("session_id", "episode_id", "clock_id", "code"):
            text(getattr(self, field), field)
        integer(self.timestamp_ns, "timestamp_ns")
        if self.stop_state not in STOP_STATES:
            raise ValueError(f"stop_state must be one of {STOP_STATES}")
