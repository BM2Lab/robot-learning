"""Native LIBERO bridge helpers. No Isaac, LeRobot or model imports."""
from contracts.libero import LiberoActionChunk, LiberoIOSpec
from .local_client import LocalSimulatorClient


class LiberoSimulatorClient(LocalSimulatorClient):
    def __init__(self, socket_path, timeout_s=120):
        super().__init__(socket_path, LiberoIOSpec(), timeout_s=timeout_s)
        receipt = self.receipt["receipt"]
        if (receipt["backend"] != "isaac_physx_libero_port_not_native_mujoco"
                or receipt["image_transform_from_isaac"] != "horizontal_flip_once"):
            self._dispose()
            raise ValueError("Unsupported native visual bridge backend/orientation")


def native_features(packet, spec=None):
    """Checkpoint-ready HWC images. Do NOT run the LIBERO environment flip again."""
    import numpy as np
    spec = spec or LiberoIOSpec()
    packet.validate_for(spec)
    return {"state": np.asarray(packet.state, dtype=np.float32), "instruction": packet.instruction,
            "images": {image.key: np.frombuffer(image.decode(), dtype=np.uint8).reshape(256, 256, 3).copy()
                       for image in packet.images}}


def bounded_chunk(packet, raw_actions, *, execute_steps=1):
    """Explicit, audited native clipping; no Cartesian-unit reinterpretation."""
    import numpy as np
    spec = LiberoIOSpec()
    packet.validate_for(spec)
    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if (actions.ndim != 2 or actions.shape[1] != 7 or not 1 <= len(actions) <= 50
            or not np.isfinite(actions).all() or type(execute_steps) is not int
            or not 1 <= execute_steps <= len(actions)):
        raise ValueError("Expected a finite native [T,7] chunk and bounded execution horizon")
    selected = actions[:execute_steps]
    bounded = selected.clip(-1, 1)
    audit = {"executed_rows": execute_steps, "components_clipped": int(np.count_nonzero(selected != bounded)),
             "peak_raw_input": float(np.abs(selected).max())}
    return LiberoActionChunk(packet.context, spec.action, tuple(map(tuple, bounded.tolist())),
        packet.context.timestamp_ns+(execute_steps+1)*50_000_000), audit
