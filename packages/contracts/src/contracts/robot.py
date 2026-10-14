"""Named embodiment and frame contract, independent of any robot backend."""
from dataclasses import dataclass
from ._validation import JsonContract, integer, names, text


@dataclass(frozen=True)
class RobotSpec(JsonContract):
    robot_id: str
    model: str
    manipulator: str
    joint_names: tuple[str, ...]
    base_frame: str
    tcp_frame: str
    schema_version: int = 1

    def __post_init__(self):
        for field in ("robot_id", "model", "manipulator", "base_frame", "tcp_frame"):
            text(getattr(self, field), field)
        object.__setattr__(self, "joint_names", names(self.joint_names, "joint_names"))
        if self.base_frame == self.tcp_frame:
            raise ValueError("Base and TCP must be distinct named frames")
        if integer(self.schema_version, "schema_version", 1) != 1:
            raise ValueError("Unsupported RobotSpec schema version")
