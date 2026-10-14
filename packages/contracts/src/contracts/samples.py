"""V2 single-robot observation wire schema; no tensor or simulator dependencies."""
import base64
from dataclasses import dataclass
from ._validation import JsonContract, integer, number, text
from .episodes import SampleContext


@dataclass(frozen=True)
class RGBSample(JsonContract):
    key: str
    optical_frame: str
    height: int
    width: int
    frame_index: int
    context: SampleContext
    data_base64: str
    encoding: str = "uint8_rgb_hwc_base64"

    def __post_init__(self):
        text(self.key, "camera key")
        text(self.optical_frame, "optical frame")
        integer(self.height, "height", 1)
        integer(self.width, "width", 1)
        integer(self.frame_index, "frame index")
        if self.height > 512 or self.width > 512 or self.encoding != "uint8_rgb_hwc_base64":
            raise ValueError("Unsupported V2 image geometry/encoding")
        if not isinstance(self.context, SampleContext):
            raise ValueError("Typed image sample context required")
        raw = self.decode()
        if len(raw) != self.height*self.width*3:
            raise ValueError("RGB payload size does not match HWC geometry")

    def decode(self):
        if not isinstance(self.data_base64, str) or len(self.data_base64) > 1048576:
            raise ValueError("Invalid or oversized image payload")
        try:
            return base64.b64decode(self.data_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid base64 image") from exc

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["context"] = SampleContext.from_dict(data["context"])
        return cls(**data)


@dataclass(frozen=True)
class ObservationPacket(JsonContract):
    contract_id: str
    context: SampleContext
    state: tuple[float, ...]
    images: tuple[RGBSample, ...]
    instruction: str
    schema_version: int = 1

    def __post_init__(self):
        text(self.contract_id, "contract id")
        text(self.instruction, "instruction")
        if not isinstance(self.context, SampleContext) or integer(self.schema_version, "schema version", 1) != 1:
            raise ValueError("Unsupported observation schema/context")
        object.__setattr__(self, "state", tuple(number(v, "joint state") for v in self.state))
        object.__setattr__(self, "images", tuple(self.images))
        if not self.images or any(not isinstance(i, RGBSample) or i.context != self.context for i in self.images):
            raise ValueError("Images must share the observation capture context")
        if len({i.key for i in self.images}) != len(self.images):
            raise ValueError("Duplicate camera keys")

    def validate_for(self, spec):
        if self.contract_id != spec.contract_id:
            raise ValueError("Observation contract mismatch")
        spec.observation.validate_state(self.state)
        expected = {c.key: c for c in spec.observation.cameras}
        if set(expected) != {i.key for i in self.images}:
            raise ValueError("Observation camera set mismatch")
        for image in self.images:
            c = expected[image.key]
            if (image.optical_frame, image.height, image.width) != (c.optical_frame, c.height, c.width):
                raise ValueError("Observation camera mapping mismatch")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["context"] = SampleContext.from_dict(data["context"])
        data["images"] = tuple(RGBSample.from_dict(i) for i in data["images"])
        return cls(**data)
