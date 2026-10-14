"""Episode/observation identity and explicit clock domain."""
from dataclasses import dataclass
from ._validation import JsonContract, integer, text


@dataclass(frozen=True)
class SampleContext(JsonContract):
    episode_id: str
    step_index: int
    timestamp_ns: int
    clock: str

    def __post_init__(self):
        text(self.episode_id, "episode_id")
        integer(self.step_index, "step_index")
        integer(self.timestamp_ns, "timestamp_ns")
        if self.clock not in ("simulation", "monotonic"):
            raise ValueError("clock must be simulation or monotonic; clocks cannot be mixed")
