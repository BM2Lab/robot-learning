"""FR3 + Franka Hand measured state sample streamed by the runtime. Standard library only.

Joint order is ``panda_joint1..7``; positions in rad, velocities in rad/s, finger
width in metres. Timestamps are the runtime's clock. An invalid sample must not be
used for observations; validity is the runtime's judgement, not inferred here.
"""
from dataclasses import dataclass
from typing import Optional
from ._validation import JsonContract, integer, names, number, text

FR3_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))


@dataclass(frozen=True)
class FrankaStateSample(JsonContract):
    robot_id: str
    clock_id: str
    timestamp_ns: int
    sequence: int
    q_rad: tuple[float, ...]
    dq_rad_s: tuple[float, ...]
    gripper_width_m: float
    gripper_moving: bool
    gripper_grasped: bool
    valid: bool
    lease_id: Optional[str] = None
    executing_sequence: Optional[int] = None
    executing_row: Optional[int] = None
    fault_code: Optional[str] = None
    joint_names: tuple[str, ...] = FR3_JOINT_NAMES
    schema_version: int = 1

    def __post_init__(self):
        text(self.robot_id, "robot_id")
        text(self.clock_id, "clock_id")
        integer(self.timestamp_ns, "timestamp_ns")
        integer(self.sequence, "sequence")
        if names(self.joint_names, "joint_names") != FR3_JOINT_NAMES:
            raise ValueError("FR3 state uses panda_joint1..7 order")
        object.__setattr__(self, "joint_names", FR3_JOINT_NAMES)
        for field in ("q_rad", "dq_rad_s"):
            values = tuple(number(v, field) for v in getattr(self, field))
            if len(values) != 7:
                raise ValueError(f"{field} needs seven joints")
            object.__setattr__(self, field, values)
        if number(self.gripper_width_m, "gripper_width_m") < 0:
            raise ValueError("Finger width cannot be negative")
        for field in ("gripper_moving", "gripper_grasped", "valid"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be boolean")
        if self.lease_id is not None:
            text(self.lease_id, "lease_id")
        if (self.executing_sequence is None) != (self.executing_row is None):
            raise ValueError("Executing sequence and row are reported together")
        if self.executing_sequence is not None:
            integer(self.executing_sequence, "executing_sequence", 1)
            integer(self.executing_row, "executing_row")
            if self.lease_id is None:
                raise ValueError("Execution requires an active lease")
        if self.fault_code is not None:
            text(self.fault_code, "fault_code")
            if self.valid:
                raise ValueError("A faulted sample cannot be valid")
        if integer(self.schema_version, "schema_version", 1) != 1:
            raise ValueError("Unsupported state schema version")

    def require_valid(self):
        if not self.valid:
            raise ValueError(f"Invalid robot state sample {self.sequence}"
                             f"{'' if self.fault_code is None else ' (' + self.fault_code + ')'}")
        return self
