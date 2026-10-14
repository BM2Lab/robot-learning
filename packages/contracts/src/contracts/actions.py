"""Physical Cartesian increments; not a decoder for pretrained model actions."""
from dataclasses import dataclass
import math
from ._validation import JsonContract, integer, number, text
from .episodes import SampleContext


@dataclass(frozen=True)
class CartesianDeltaSpec(JsonContract):
    manipulator: str
    frame: str
    tcp_frame: str
    policy_hz: float
    max_translation_norm_m: float
    mode: str = "cartesian_delta"
    components: tuple[str, ...] = ("dx", "dy", "dz")
    unit: str = "m"
    reference: str = "measured_tcp_at_execution"
    orientation: str = "hold_episode_initial"

    def __post_init__(self):
        for field in ("manipulator", "frame", "tcp_frame"):
            text(getattr(self, field), field)
        object.__setattr__(self, "components", tuple(self.components))
        number(self.policy_hz, "policy_hz", positive=True)
        number(self.max_translation_norm_m, "max_translation_norm_m", positive=True)
        if (self.mode, self.components, self.unit, self.reference, self.orientation) != (
            "cartesian_delta", ("dx", "dy", "dz"), "m",
            "measured_tcp_at_execution", "hold_episode_initial"
        ):
            raise ValueError("Unsupported V0 action units, representation or reference")

    def validate_robot(self, robot):
        if (self.manipulator, self.frame, self.tcp_frame) != (
            robot.manipulator, robot.base_frame, robot.tcp_frame
        ):
            raise ValueError("Action manipulator/base/TCP mapping does not match robot")

    def validate_values(self, values):
        row = tuple(number(v, "Cartesian increment") for v in values)
        if len(row) != 3:
            raise ValueError("Cartesian increment must have three components")
        if math.sqrt(sum(v*v for v in row)) > self.max_translation_norm_m:
            raise ValueError("Cartesian increment exceeds translation norm bound")
        return row


@dataclass(frozen=True)
class ActionChunk(JsonContract):
    observation: SampleContext
    action_spec: CartesianDeltaSpec
    values: tuple[tuple[float, ...], ...]
    valid_until_ns: int

    def __post_init__(self):
        if not isinstance(self.observation, SampleContext) or not isinstance(self.action_spec, CartesianDeltaSpec):
            raise ValueError("Typed observation context and action specification are required")
        rows = tuple(self.action_spec.validate_values(row) for row in self.values)
        if not rows:
            raise ValueError("Action chunk cannot be empty")
        object.__setattr__(self, "values", rows)
        integer(self.valid_until_ns, "valid_until_ns")
        if self.valid_until_ns <= self.observation.timestamp_ns:
            raise ValueError("Action deadline must be after the source observation")

    def validate_for(self, spec, current):
        """Acceptance-time check; queued execution/reset handling is a later runner concern."""
        if self.action_spec != spec.action or len(self.values) > spec.chunk_size:
            raise ValueError("Action chunk is incompatible with policy contract")
        if (current.episode_id, current.step_index, current.clock) != (
            self.observation.episode_id, self.observation.step_index, self.observation.clock
        ):
            raise ValueError("Wrong episode, source step or clock domain")
        if current.timestamp_ns < self.observation.timestamp_ns or current.timestamp_ns >= self.valid_until_ns:
            raise ValueError("Action chunk is premature or expired")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["observation"] = SampleContext.from_dict(data["observation"])
        data["action_spec"] = CartesianDeltaSpec.from_dict(data["action_spec"])
        return cls(**data)
