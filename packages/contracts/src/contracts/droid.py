"""Explicit native openpi DROID joint-position I/O bound to an FR3 + Franka Hand.

Standard library only. Semantics are pinned to openpi 215abfb config
``pi05_droid_jointpos_polaris``: the server applies AbsoluteActions, so each row
is an ABSOLUTE seven-joint target in radians plus a gripper closure in [0, 1]
(0 open, 1 closed). This is neither the FR3 CartesianDeltaSpec nor DROID's
joint-velocity action space; never clip joint targets to [-1, 1] as the upstream
velocity example does. Images are policy inputs after openpi resize_with_pad;
raw capture geometry and drivers belong to the sensor layer, and the Franka Hand
width/closure mapping is declared here but executed by the runtime.
"""
from dataclasses import dataclass, field
from ._validation import JsonContract, integer, number, text
from .episodes import SampleContext
from .observations import CameraSpec
from .robot import RobotSpec

JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
CHUNK_SIZE = 15  # pi05 DROID action_horizon
CAMERAS = (CameraSpec("observation/exterior_image_1_left", "exterior_optical", 224, 224),
           CameraSpec("observation/wrist_image_left", "wrist_optical", 224, 224))


def _closure(value, field):
    value = number(value, field)
    if not 0 <= value <= 1:
        raise ValueError(f"{field} must be a closure in [0, 1]; clip and audit raw outputs explicitly")
    return value


def _state(values):
    values = tuple(values)
    if len(values) != 8:
        raise ValueError("DROID state is seven joint positions followed by gripper closure")
    return (*(number(v, "joint position") for v in values[:7]), _closure(values[7], "gripper closure"))


@dataclass(frozen=True)
class DroidObservationSpec(JsonContract):
    joint_key: str = "observation/joint_position"
    gripper_key: str = "observation/gripper_position"
    instruction_key: str = "prompt"
    joint_names: tuple[str, ...] = JOINT_NAMES
    joint_unit: str = "rad"
    gripper_unit: str = "closure_0_open_1_closed"
    image_preprocessing: str = "openpi_resize_with_pad"
    cameras: tuple[CameraSpec, ...] = field(default_factory=lambda: CAMERAS)

    def __post_init__(self):
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(self, "cameras", tuple(self.cameras))
        if ((self.joint_key, self.gripper_key, self.instruction_key) != (
                "observation/joint_position", "observation/gripper_position", "prompt")
                or self.joint_names != JOINT_NAMES or self.joint_unit != "rad"
                or self.gripper_unit != "closure_0_open_1_closed"
                or self.image_preprocessing != "openpi_resize_with_pad" or self.cameras != CAMERAS):
            raise ValueError("Unsupported native DROID observation contract")

    def validate_robot(self, robot):
        if self.joint_names != robot.joint_names:
            raise ValueError("Observation joint order does not match the robot")

    def validate_state(self, values):
        return _state(values)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["cameras"] = tuple(CameraSpec.from_dict(c) for c in data["cameras"])
        return cls(**data)


@dataclass(frozen=True)
class DroidActionSpec(JsonContract):
    max_deviation_from_source_rad: float
    mode: str = "droid_absolute_joint_position_gripper"
    components: tuple[str, ...] = (*JOINT_NAMES, "gripper_closure")
    joint_unit: str = "rad_absolute_target"
    gripper_unit: str = "closure_0_open_1_closed"
    policy_hz: float = 15.

    def __post_init__(self):
        object.__setattr__(self, "components", tuple(self.components))
        number(self.max_deviation_from_source_rad, "max_deviation_from_source_rad", positive=True)
        if (self.mode != "droid_absolute_joint_position_gripper"
                or self.components != (*JOINT_NAMES, "gripper_closure")
                or self.joint_unit != "rad_absolute_target" or self.gripper_unit != "closure_0_open_1_closed"
                or number(self.policy_hz, "policy_hz") != 15.):
            raise ValueError("Unsupported native DROID action contract")

    def validate_values(self, values):
        return _state(values)


@dataclass(frozen=True)
class FrankaHandClosureMap(JsonContract):
    """Maps measured finger width to DROID closure and closure to a binary command."""
    open_width_m: float
    gripper_model: str = "franka_hand"
    observation: str = "linear_width_clipped"
    command: str = "binary_threshold"
    close_above: float = 0.5

    def __post_init__(self):
        number(self.open_width_m, "open_width_m", positive=True)
        if (self.gripper_model, self.observation, self.command) != (
                "franka_hand", "linear_width_clipped", "binary_threshold"):
            raise ValueError("Unsupported gripper closure mapping")
        if number(self.close_above, "close_above") != 0.5:
            raise ValueError("DROID closure is binarized at 0.5 upstream")

    def closure_from_width(self, width_m):
        width_m = number(width_m, "finger width")
        if width_m < 0:
            raise ValueError("Finger width cannot be negative")
        return min(1., max(0., 1.-width_m/self.open_width_m))

    def command_from_closure(self, closure):
        return "close" if _closure(closure, "gripper closure") > self.close_above else "open"


@dataclass(frozen=True)
class DroidIOSpec(JsonContract):
    contract_id: str
    robot: RobotSpec
    gripper: FrankaHandClosureMap
    action: DroidActionSpec
    open_loop_horizon: int
    observation: DroidObservationSpec = field(default_factory=DroidObservationSpec)
    chunk_size: int = CHUNK_SIZE
    schema_version: int = 1

    def __post_init__(self):
        text(self.contract_id, "contract_id")
        if not (isinstance(self.robot, RobotSpec) and isinstance(self.gripper, FrankaHandClosureMap)
                and isinstance(self.action, DroidActionSpec)
                and isinstance(self.observation, DroidObservationSpec)):
            raise ValueError("Typed robot, gripper, observation and action specifications are required")
        if integer(self.schema_version, "schema_version", 1) != 1 or integer(self.chunk_size, "chunk_size", 1) != CHUNK_SIZE:
            raise ValueError("Unsupported DROID I/O specification")
        if not 1 <= integer(self.open_loop_horizon, "open_loop_horizon", 1) <= self.chunk_size:
            raise ValueError("open_loop_horizon must be within the chunk")
        self.observation.validate_robot(self.robot)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["robot"] = RobotSpec.from_dict(data["robot"])
        data["gripper"] = FrankaHandClosureMap.from_dict(data["gripper"])
        data["action"] = DroidActionSpec.from_dict(data["action"])
        data["observation"] = DroidObservationSpec.from_dict(data["observation"])
        return cls(**data)


@dataclass(frozen=True)
class DroidActionChunk(JsonContract):
    observation: SampleContext
    action_spec: DroidActionSpec
    source_state: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    valid_until_ns: int

    def __post_init__(self):
        if not isinstance(self.observation, SampleContext) or not isinstance(self.action_spec, DroidActionSpec):
            raise ValueError("Typed native DROID context/action required")
        source = _state(self.source_state)
        rows = tuple(self.action_spec.validate_values(row) for row in self.values)
        if not 1 <= len(rows) <= CHUNK_SIZE:
            raise ValueError(f"Native chunk must contain 1..{CHUNK_SIZE} rows")
        bound = self.action_spec.max_deviation_from_source_rad
        if any(abs(q-q0) > bound for row in rows for q, q0 in zip(row[:7], source[:7])):
            raise ValueError("Joint target exceeds deviation bound from source state; reject, do not clip")
        object.__setattr__(self, "source_state", source)
        object.__setattr__(self, "values", rows)
        integer(self.valid_until_ns, "deadline")
        if self.valid_until_ns <= self.observation.timestamp_ns:
            raise ValueError("Chunk deadline must follow source observation")

    def executable_rows(self, spec):
        return self.values[:spec.open_loop_horizon]

    def validate_for(self, spec, current):
        """Acceptance-time check; execution scheduling and stopping belong to the runtime."""
        if (not isinstance(spec, DroidIOSpec) or self.action_spec != spec.action
                or len(self.values) > spec.chunk_size):
            raise ValueError("Action chunk is incompatible with DROID contract")
        if (current.episode_id, current.clock) != (self.observation.episode_id, self.observation.clock) \
                or current.step_index < self.observation.step_index:
            raise ValueError("Wrong episode, source step or clock domain")
        if not self.observation.timestamp_ns <= current.timestamp_ns < self.valid_until_ns:
            raise ValueError("Action chunk is premature or expired")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["observation"] = SampleContext.from_dict(data["observation"])
        data["action_spec"] = DroidActionSpec.from_dict(data["action_spec"])
        return cls(**data)
