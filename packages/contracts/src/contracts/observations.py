"""V0 joint-position and named RGB feature specifications (no tensor imports)."""
from dataclasses import dataclass
from ._validation import JsonContract, integer, names, text


@dataclass(frozen=True)
class CameraSpec(JsonContract):
    key: str
    optical_frame: str
    height: int
    width: int
    channels: int = 3
    color_space: str = "RGB"
    dtype: str = "uint8"
    layout: str = "HWC"

    def __post_init__(self):
        text(self.key, "camera key")
        text(self.optical_frame, "optical_frame")
        integer(self.height, "height", 1)
        integer(self.width, "width", 1)
        integer(self.channels, "channels", 1)
        if (self.channels, self.color_space, self.dtype, self.layout) != (3, "RGB", "uint8", "HWC"):
            raise ValueError("V0 cameras require uint8 RGB HWC with three channels")


@dataclass(frozen=True)
class ObservationSpec(JsonContract):
    state_key: str
    joint_names: tuple[str, ...]
    cameras: tuple[CameraSpec, ...]
    instruction_key: str
    state_unit: str = "rad"

    def __post_init__(self):
        text(self.state_key, "state_key")
        text(self.instruction_key, "instruction_key")
        object.__setattr__(self, "joint_names", names(self.joint_names, "joint_names"))
        object.__setattr__(self, "cameras", tuple(self.cameras))
        if not self.cameras or not all(isinstance(c, CameraSpec) for c in self.cameras):
            raise ValueError("Named real camera specifications are required")
        names([self.state_key, self.instruction_key, *(c.key for c in self.cameras)], "feature keys")
        if self.state_unit != "rad":
            raise ValueError("V0 observation state is measured joint positions in radians")

    def validate_robot(self, robot):
        if self.joint_names != robot.joint_names:
            raise ValueError("Observation joint order does not match the robot")

    def validate_state(self, values):
        from ._validation import number
        values = tuple(number(v, "joint position") for v in values)
        if len(values) != len(self.joint_names):
            raise ValueError("Wrong observation state dimension")
        return values

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["cameras"] = tuple(CameraSpec.from_dict(c) for c in data["cameras"])
        return cls(**data)
