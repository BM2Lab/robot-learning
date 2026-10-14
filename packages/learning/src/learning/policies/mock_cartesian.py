"""Bounded deterministic transport test policy; not a reaching policy or VLA adapter."""
from contracts.actions import ActionChunk


class MockCartesianPolicy:
    def __init__(self, spec):
        self.spec = spec

    def predict(self, observation):
        observation.validate_for(self.spec)
        # Intentionally receives only policy observations, never goal coordinates.
        rows = ((0.0, 0.0, 0.001), (0.0, 0.0, -0.001))
        return ActionChunk(observation.context, self.spec.action, rows,
            observation.context.timestamp_ns + 1_000_000_000)
