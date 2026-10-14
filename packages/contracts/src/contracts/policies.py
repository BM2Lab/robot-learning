"""Versioned future FR3 policy I/O, separate from checkpoint-native features."""
from dataclasses import dataclass
from ._validation import JsonContract, integer, text
from .robot import RobotSpec
from .observations import ObservationSpec
from .actions import CartesianDeltaSpec


@dataclass(frozen=True)
class PolicyIOSpec(JsonContract):
    contract_id: str
    robot: RobotSpec
    observation: ObservationSpec
    action: CartesianDeltaSpec
    chunk_size: int
    schema_version: int = 1

    def __post_init__(self):
        text(self.contract_id, "contract_id")
        if not (isinstance(self.robot, RobotSpec) and isinstance(self.observation, ObservationSpec)
                and isinstance(self.action, CartesianDeltaSpec)):
            raise ValueError("Typed robot, observation and action specifications are required")
        integer(self.chunk_size, "chunk_size", 1)
        if integer(self.schema_version, "schema_version", 1) != 1:
            raise ValueError("Unsupported policy I/O schema version")
        self.observation.validate_robot(self.robot)
        self.action.validate_robot(self.robot)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["robot"] = RobotSpec.from_dict(data["robot"])
        data["observation"] = ObservationSpec.from_dict(data["observation"])
        data["action"] = CartesianDeltaSpec.from_dict(data["action"])
        return cls(**data)
