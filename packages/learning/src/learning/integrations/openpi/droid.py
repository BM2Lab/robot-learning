"""openpi DROID joint-position client. openpi-client/numpy are imported lazily; no model or robot imports.

The policy server runs upstream openpi ``serve_policy.py`` in its own environment.
Returned actions are native DROID rows (absolute joint targets + gripper closure)
bound to the source capture. Gripper closure is explicitly clipped and audited;
joint targets are never clipped, so out-of-bound chunks are rejected by the contract.
"""
import base64
import json
from pathlib import Path
import socket
import time

from contracts.droid import DroidActionChunk, DroidIOSpec
from .checkpoint import server_identity

PIN_KEYS = {"schema_version", "policy_type", "openpi", "checkpoint", "config_assets", "tokenizer", "checkpoint_features",
            "native_contract", "deployment_status", "serving_rule"}


def load_pin(path):
    pin = json.loads(Path(path).read_text())
    if set(pin) != PIN_KEYS or pin["schema_version"] != 1 or pin["policy_type"] != "openpi":
        raise ValueError("Unsupported openpi policy pin")
    if pin["checkpoint_features"]["action_shape"] != [15, 8]:
        raise ValueError("Pin does not describe a 15x8 native DROID chunk")
    if not {"dir", "files", "bytes", "tree_sha256", "norm_stats", "norm_stats_sha256"} <= set(pin["checkpoint"]) \
            or set(pin["tokenizer"]) != {"url", "sha256"}:
        raise ValueError("Pin lacks checkpoint/tokenizer identity")
    return pin


def chunk_deadline_ns(spec, context):
    """A chunk is valid until its executable rows would have finished at the policy rate."""
    return context.timestamp_ns+round(spec.open_loop_horizon*1e9/spec.action.policy_hz)


def request(spec, packet):
    """openpi DROID request from a validated packet: state split into joints and gripper closure."""
    import numpy as np
    packet.validate_for(spec)
    images = {}
    for image in packet.images:
        images[image.key] = np.frombuffer(image.decode(), dtype=np.uint8).reshape(image.height, image.width, 3)
    obs = spec.observation
    return {obs.cameras[0].key: images[obs.cameras[0].key], obs.cameras[1].key: images[obs.cameras[1].key],
            obs.joint_key: np.asarray(packet.state[:7], dtype=np.float32),
            obs.gripper_key: np.asarray(packet.state[7:], dtype=np.float32),
            obs.instruction_key: packet.instruction}


class ChunkRejected(ValueError):
    """The native chunk violated the contract; ``audit`` still describes the raw output."""

    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def bounded_chunk(spec, packet, raw_actions, audit=None):
    """Keep only the open-loop rows, clip/audit gripper closure; joint deviations are checked, never clipped."""
    import numpy as np
    actions = np.asarray(raw_actions, dtype=np.float64)
    if actions.shape != (spec.chunk_size, 8) or not np.isfinite(actions).all():
        raise ValueError(f"Expected a finite native ({spec.chunk_size}, 8) chunk, got {actions.shape}")
    executed = actions[:spec.open_loop_horizon]
    gripper = executed[:, 7]
    bounded = executed.copy()
    bounded[:, 7] = gripper.clip(0, 1)
    deviation = np.abs(actions[:, :7]-np.asarray(packet.state[:7]))
    audit = dict(audit or {}, executed_rows=len(executed),
                 gripper_clipped=int(np.count_nonzero(bounded[:, 7] != gripper)),
                 gripper_raw_range=[float(gripper.min()), float(gripper.max())],
                 max_joint_deviation_rad=float(deviation[:len(executed)].max()),
                 max_joint_deviation_full_chunk_rad=float(deviation.max()))
    try:
        chunk = DroidActionChunk(packet.context, spec.action, packet.state, tuple(map(tuple, bounded.tolist())),
                                 chunk_deadline_ns(spec, packet.context))
    except ValueError as exc:
        raise ChunkRejected(str(exc), audit) from exc
    return chunk, audit


class OpenPIDroidClient:
    """Wraps any policy with openpi's ``infer``/``get_server_metadata`` interface."""

    def __init__(self, spec, pin, policy, *, allow_unverified_server=False):
        """The server must publish the pin's identity (scripts/serve_openpi_policy.py does).

        ``allow_unverified_server`` admits upstream ``serve_policy.py``, whose metadata is
        empty; the receipt then records the server as unverified.
        """
        if not isinstance(spec, DroidIOSpec):
            raise ValueError("Typed DroidIOSpec required")
        self.spec, self.pin, self.policy = spec, pin, policy
        self.server_metadata = policy.get_server_metadata()
        self.server_verified = self.server_metadata == server_identity(pin)
        if not self.server_verified and not allow_unverified_server:
            raise ValueError(f"Server identity {self.server_metadata!r} does not match the policy pin")

    @classmethod
    def connect(cls, spec, pin, host, port, *, connect_timeout_s=5., allow_unverified_server=False):
        """Fail fast when nothing listens; upstream's client otherwise retries forever."""
        with socket.create_connection((host, port), timeout=connect_timeout_s):
            pass
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        return cls(spec, pin, WebsocketClientPolicy(host=host, port=port),
                   allow_unverified_server=allow_unverified_server)

    def infer(self, packet):
        """Blocking round trip. Returns (DroidActionChunk, audit); late chunks fail at runtime acceptance."""
        payload = request(self.spec, packet)
        start = time.monotonic_ns()
        response = self.policy.infer(payload)
        roundtrip_ns = time.monotonic_ns()-start
        if not isinstance(response, dict) or "actions" not in response:
            raise RuntimeError("openpi response has no actions")
        timing = {"roundtrip_ms": roundtrip_ns/1e6, "server_timing": response.get("server_timing"),
                  "policy_timing": response.get("policy_timing")}
        return bounded_chunk(self.spec, packet, response["actions"], timing)

    def receipt(self):
        return {"contract_id": self.spec.contract_id, "openpi": self.pin["openpi"],
                "checkpoint": self.pin["checkpoint"], "server_metadata": _jsonable(self.server_metadata),
                "server_verified": self.server_verified}


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def encode_rgb(image):
    """Base64 for contracts RGBSample; ``image`` is a uint8 HWC numpy array."""
    return base64.b64encode(image.tobytes()).decode()
