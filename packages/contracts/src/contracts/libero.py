"""Explicit native LIBERO I/O. Not FR3 CartesianDeltaSpec or hardware commands.

Standard library only. Pixel buffers are already in LeRobot LIBERO-processor
orientation; downstream policy preprocessing must NOT apply that flip again.
"""
from dataclasses import dataclass, field
from ._validation import JsonContract, integer, number
from .episodes import SampleContext
from .observations import CameraSpec


@dataclass(frozen=True)
class LiberoObservationSpec(JsonContract):
    state_key: str = "observation.state"
    instruction_key: str = "task"
    components: tuple[str, ...] = ("eef_x_m", "eef_y_m", "eef_z_m", "rotvec_x_rad",
        "rotvec_y_rad", "rotvec_z_rad", "left_finger_m", "right_finger_m")
    frame: str = "libero_world"
    position_site: str = "gripper0_grip_site"
    orientation_body: str = "robot0_right_hand"
    quaternion_branch: str = "native_fk_no_w_canonicalization"
    image_orientation: str = "lerobot_libero_processor_output"
    cameras: tuple[CameraSpec, ...] = field(default_factory=lambda: (
        CameraSpec("observation.images.image", "agentview_optical", 256, 256),
        CameraSpec("observation.images.image2", "eye_in_hand_optical", 256, 256)))

    def __post_init__(self):
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "cameras", tuple(self.cameras))
        expected = ("eef_x_m", "eef_y_m", "eef_z_m", "rotvec_x_rad", "rotvec_y_rad",
                    "rotvec_z_rad", "left_finger_m", "right_finger_m")
        if (self.state_key != "observation.state" or self.instruction_key != "task"
                or self.frame != "libero_world" or self.components != expected
                or self.position_site != "gripper0_grip_site" or self.orientation_body != "robot0_right_hand"
                or self.quaternion_branch != "native_fk_no_w_canonicalization"
                or self.image_orientation != "lerobot_libero_processor_output"
                or self.cameras != (CameraSpec("observation.images.image", "agentview_optical", 256, 256),
                                    CameraSpec("observation.images.image2", "eye_in_hand_optical", 256, 256))):
            raise ValueError("Unsupported native LIBERO observation contract")

    def validate_state(self, values):
        result = tuple(number(v, "LIBERO state") for v in values)
        if len(result) != 8:
            raise ValueError("LIBERO requires eight native EE/gripper state components")
        return result

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["cameras"] = tuple(CameraSpec.from_dict(c) for c in data["cameras"])
        return cls(**data)


@dataclass(frozen=True)
class LiberoActionSpec(JsonContract):
    mode: str = "libero_osc_pose_gripper"
    frame: str = "libero_world"
    unit: str = "normalized_controller_input_NOT_meters"
    policy_hz: float = 20.
    components: tuple[str, ...] = ("dx", "dy", "dz", "rx", "ry", "rz", "gripper")

    def __post_init__(self):
        object.__setattr__(self, "components", tuple(self.components))
        if (self.mode != "libero_osc_pose_gripper" or self.frame != "libero_world"
                or self.unit != "normalized_controller_input_NOT_meters"
                or number(self.policy_hz, "policy_hz") != 20.
                or self.components != ("dx", "dy", "dz", "rx", "ry", "rz", "gripper")):
            raise ValueError("Unsupported native LIBERO action contract")

    def validate_values(self, values):
        row = tuple(number(v, "native controller input") for v in values)
        if len(row) != 7 or any(abs(v) > 1 for v in row):
            raise ValueError("Expected seven bounded [-1,1] native inputs; clip and audit raw policy outputs explicitly")
        return row


@dataclass(frozen=True)
class LiberoIOSpec(JsonContract):
    contract_id: str = "libero_object0_osc_visual_v1"
    observation: LiberoObservationSpec = field(default_factory=LiberoObservationSpec)
    action: LiberoActionSpec = field(default_factory=LiberoActionSpec)
    chunk_size: int = 50
    schema_version: int = 1

    def __post_init__(self):
        if (self.contract_id != "libero_object0_osc_visual_v1"
                or integer(self.schema_version, "schema_version", 1) != 1
                or integer(self.chunk_size, "chunk_size", 1) != 50
                or not isinstance(self.observation, LiberoObservationSpec)
                or not isinstance(self.action, LiberoActionSpec)):
            raise ValueError("Unsupported LIBERO I/O specification")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["observation"] = LiberoObservationSpec.from_dict(data["observation"])
        data["action"] = LiberoActionSpec.from_dict(data["action"])
        return cls(**data)


@dataclass(frozen=True)
class LiberoActionChunk(JsonContract):
    observation: SampleContext
    action_spec: LiberoActionSpec
    values: tuple[tuple[float, ...], ...]
    valid_until_ns: int

    def __post_init__(self):
        if not isinstance(self.observation, SampleContext) or not isinstance(self.action_spec, LiberoActionSpec):
            raise ValueError("Typed native LIBERO context/action required")
        rows = tuple(self.action_spec.validate_values(row) for row in self.values)
        if not 1 <= len(rows) <= 50:
            raise ValueError("Native chunk must contain 1..50 rows")
        object.__setattr__(self, "values", rows)
        integer(self.valid_until_ns, "deadline")
        if self.valid_until_ns <= self.observation.timestamp_ns:
            raise ValueError("Chunk deadline must follow source observation")

    def validate_for(self, spec, current):
        if (not isinstance(spec, LiberoIOSpec) or self.action_spec != spec.action
                or len(self.values) > spec.chunk_size or self.observation != current
                or current.clock != "simulation" or current.timestamp_ns >= self.valid_until_ns):
            raise ValueError("Incompatible, stale or expired native LIBERO chunk")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["observation"] = SampleContext.from_dict(data["observation"])
        data["action_spec"] = LiberoActionSpec.from_dict(data["action_spec"])
        return cls(**data)
