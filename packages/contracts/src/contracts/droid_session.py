"""Remote DROID session messages and chunk-to-envelope binding. Standard library only.

The runner opens a session with the exact DroidIOSpec, the runtime replies with a
lease in its own clock domain, and every DroidActionChunk travels with a
CommandEnvelope whose times are converted to runtime time via a ClockOffsetEstimate
(deadlines conservatively). Physical guards, execution and stopping are not
described here; they remain runtime authority.
"""
from dataclasses import dataclass
from ._validation import JsonContract, integer, number, text
from .clock import ClockOffsetEstimate
from .droid import DroidActionChunk, DroidIOSpec
from .sessions import CommandEnvelope, LeaseRequest, SessionLease, payload_sha256

CAPABILITY_PREFIX = "droid-jointpos-v1"


def capability_id(spec):
    if not isinstance(spec, DroidIOSpec):
        raise ValueError("Typed DroidIOSpec required")
    return f"{CAPABILITY_PREFIX}:{payload_sha256(spec)}"


@dataclass(frozen=True)
class DroidSessionOpen(JsonContract):
    contract: DroidIOSpec
    contract_sha256: str
    lease_request: LeaseRequest
    client_clock_id: str
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.contract, DroidIOSpec) or not isinstance(self.lease_request, LeaseRequest):
            raise ValueError("Typed contract and lease request required")
        if self.contract_sha256 != payload_sha256(self.contract):
            raise ValueError("Contract fingerprint does not match the contract")
        if self.lease_request.robot_id != self.contract.robot.robot_id:
            raise ValueError("Lease request and contract name different robots")
        text(self.client_clock_id, "client_clock_id")
        if self.client_clock_id == self.lease_request.clock_id:
            raise ValueError("Client and runtime clocks must be named distinctly")
        if integer(self.schema_version, "schema_version", 1) != 1:
            raise ValueError("Unsupported session schema version")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["contract"] = DroidIOSpec.from_dict(data["contract"])
        data["lease_request"] = LeaseRequest.from_dict(data["lease_request"])
        return cls(**data)


@dataclass(frozen=True)
class DroidSessionReady(JsonContract):
    lease: SessionLease
    contract_sha256: str
    capability_id: str
    gripper_open_width_m: float
    hold_timeout_ns: int
    max_clock_uncertainty_ns: int
    runtime_versions: dict
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.lease, SessionLease):
            raise ValueError("Typed lease required")
        text(self.contract_sha256, "contract_sha256")
        text(self.capability_id, "capability_id")
        number(self.gripper_open_width_m, "gripper_open_width_m", positive=True)
        integer(self.hold_timeout_ns, "hold_timeout_ns", 1)
        integer(self.max_clock_uncertainty_ns, "max_clock_uncertainty_ns", 1)
        if (not isinstance(self.runtime_versions, dict) or not self.runtime_versions
                or not all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
                           for k, v in self.runtime_versions.items())):
            raise ValueError("runtime_versions must map component names to versions")
        if integer(self.schema_version, "schema_version", 1) != 1:
            raise ValueError("Unsupported session schema version")

    def validate_for(self, request):
        """Runner-side check that the runtime granted the session that was asked for."""
        if not isinstance(request, DroidSessionOpen):
            raise ValueError("Typed session request required")
        lease, wanted = self.lease, request.lease_request
        if (lease.robot_id, lease.owner_id, lease.clock_id) != (wanted.robot_id, wanted.owner_id, wanted.clock_id):
            raise ValueError("Lease robot, owner or clock differs from the request")
        if self.contract_sha256 != request.contract_sha256 or self.capability_id != capability_id(request.contract):
            raise ValueError("Runtime accepted a different contract or capability")
        if self.gripper_open_width_m != request.contract.gripper.open_width_m:
            raise ValueError("Homed gripper width differs from the contract's open_width_m; update the contract")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["lease"] = SessionLease.from_dict(data["lease"])
        return cls(**data)


def _clock_pair(estimate, lease, chunk):
    if not isinstance(estimate, ClockOffsetEstimate) or estimate.server_clock_id != lease.clock_id:
        raise ValueError("Clock estimate must map into the lease clock")
    if chunk.observation.clock != "monotonic":
        raise ValueError("Physical chunks use the monotonic clock domain")


def chunk_envelope(lease, sequence, chunk, estimate, *, client_now_ns, capability):
    """Runner side: bind a chunk to a runtime-time envelope (deadline converted conservatively)."""
    if not isinstance(chunk, DroidActionChunk):
        raise ValueError("Typed DroidActionChunk required")
    _clock_pair(estimate, lease, chunk)
    return CommandEnvelope(lease, sequence, chunk.observation.step_index,
                           estimate.to_server(chunk.observation.timestamp_ns), estimate.to_server(client_now_ns),
                           estimate.deadline_to_server(chunk.valid_until_ns), capability, payload_sha256(chunk))


def validate_chunk_command(spec, envelope, chunk, *, lease, estimate, now_ns, next_sequence, capability):
    """Runtime side: identity, order, payload and time binding before physical checks.

    ``estimate`` is the session's registered estimate, re-derived by the runtime
    from the client's probes. Physical limits, current-state checks and execution
    are the runtime's own and come after this.
    """
    if not isinstance(chunk, DroidActionChunk) or not isinstance(envelope, CommandEnvelope):
        raise ValueError("Typed envelope and chunk required")
    if chunk.action_spec != spec.action or len(chunk.values) > spec.open_loop_horizon:
        raise ValueError("Chunk is incompatible with the contract or exceeds the open-loop horizon")
    _clock_pair(estimate, lease, chunk)
    envelope.validate_for(lease, now_ns=now_ns, clock_id=lease.clock_id, next_sequence=next_sequence,
                          capability_id=capability, payload=chunk)
    if envelope.source_step != chunk.observation.step_index:
        raise ValueError("Envelope source step differs from the chunk's capture")
    if envelope.source_observation_ns != estimate.to_server(chunk.observation.timestamp_ns):
        raise ValueError("Envelope source time does not match the registered clock estimate")
    if envelope.deadline_ns > estimate.deadline_to_server(chunk.valid_until_ns):
        raise ValueError("Envelope deadline is later than the chunk allows")
