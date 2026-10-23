import base64
from pathlib import Path
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from contracts.droid import DroidIOSpec
from contracts.episodes import SampleContext
from contracts.samples import ObservationPacket, RGBSample
from learning.integrations.openpi.cameras import DroidCameraRouter, droid_packet
from learning.integrations.openpi.checkpoint import server_identity
from learning.integrations.openpi.droid import (ChunkRejected, OpenPIDroidClient, bounded_chunk,
                                                chunk_deadline_ns, load_pin, request)

ROOT = Path(__file__).resolve().parents[3]
HOME = (0., 0.27, 0., -2.03, 0., 2.35, 0.08)
EXTERIOR, WRIST = "observation/exterior_image_1_left", "observation/wrist_image_left"


@pytest.fixture
def spec():
    return DroidIOSpec.from_json((ROOT / "configs/contracts/droid_fr3_jointpos_v1.json").read_text())


@pytest.fixture
def context():
    return SampleContext("episode", 0, 2_000_000_000, "monotonic")


def packet(spec, context, closure=0.25):
    pixels = base64.b64encode(bytes(224*224*3)).decode()
    images = tuple(RGBSample(c.key, c.optical_frame, 224, 224, 0, context, pixels) for c in spec.observation.cameras)
    return ObservationPacket(spec.contract_id, context, (*HOME, closure), images, "put the cube in the bowl")


def raw_chunk(gripper=0.):
    return np.array([[*HOME, gripper]]*15)


def test_pin_matches_contract():
    pin = load_pin(ROOT / "configs/policies/pi05_droid_jointpos.json")
    assert pin["openpi"]["train_config"] == "pi05_droid_jointpos_polaris"
    assert pin["native_contract"] == "configs/contracts/droid_fr3_jointpos_v1.json"


def test_request_uses_openpi_droid_keys(spec, context):
    payload = request(spec, packet(spec, context))
    assert set(payload) == {EXTERIOR, WRIST, "observation/joint_position", "observation/gripper_position", "prompt"}
    assert payload[EXTERIOR].shape == (224, 224, 3) and payload[EXTERIOR].dtype == np.uint8
    assert payload["observation/joint_position"].shape == (7,)
    assert payload["observation/gripper_position"].tolist() == [0.25]


def test_chunk_binds_capture_and_audits_gripper_clip(spec, context):
    raw = raw_chunk(1.)
    raw[:3, 7] = [1.04, -0.02, 0.6]
    chunk, audit = bounded_chunk(spec, packet(spec, context), raw)
    assert chunk.observation == context and len(chunk.values) == audit["executed_rows"] == 8
    assert [row[7] for row in chunk.values[:3]] == [1., 0., 0.6]
    assert audit["gripper_clipped"] == 2 and audit["gripper_raw_range"] == [-0.02, 1.04]
    assert chunk.valid_until_ns == chunk_deadline_ns(spec, context) == 2_000_000_000+533_333_333
    assert chunk.executable_rows(spec) == chunk.values


@pytest.mark.parametrize("raw", [np.zeros((15, 7)), np.zeros((10, 8)), np.full((15, 8), np.nan)])
def test_malformed_chunks_rejected(spec, context, raw):
    with pytest.raises(ValueError):
        bounded_chunk(spec, packet(spec, context), raw)


def test_joint_targets_are_never_clipped(spec, context):
    raw = raw_chunk()
    raw[5, 3] += 0.7
    with pytest.raises(ChunkRejected, match="deviation") as rejected:
        bounded_chunk(spec, packet(spec, context), raw, {"roundtrip_ms": 1.})
    assert rejected.value.audit["max_joint_deviation_rad"] == pytest.approx(0.7)
    assert rejected.value.audit["roundtrip_ms"] == 1.
    raw = raw_chunk()
    raw[12, 3] += 0.7  # Beyond the open-loop horizon: never executed, so not bounded.
    chunk, audit = bounded_chunk(spec, packet(spec, context), raw)
    assert len(chunk.values) == 8 and audit["max_joint_deviation_full_chunk_rad"] == pytest.approx(0.7)
    assert audit["max_joint_deviation_rad"] == pytest.approx(0.)
    with pytest.raises(ValueError):  # Velocity-style [-1, 1] rows are not absolute joint targets.
        bounded_chunk(spec, packet(spec, context), np.clip(raw_chunk(), -1, 1))


PIN = ROOT / "configs/policies/pi05_droid_jointpos.json"


class FakePolicy:
    def __init__(self, response, metadata=None):
        self.response, self.requests = response, []
        self.metadata = server_identity(load_pin(PIN)) if metadata is None else metadata

    def get_server_metadata(self):
        return self.metadata

    def infer(self, obs):
        self.requests.append(obs)
        return self.response


def test_client_round_trip_and_receipt(spec, context):
    pin = load_pin(ROOT / "configs/policies/pi05_droid_jointpos.json")
    policy = FakePolicy({"actions": raw_chunk(), "policy_timing": {"infer_ms": 50.}})
    client = OpenPIDroidClient(spec, pin, policy)
    chunk, audit = client.infer(packet(spec, context))
    assert chunk.values[0] == (*HOME, 0.)
    assert audit["policy_timing"] == {"infer_ms": 50.} and audit["roundtrip_ms"] >= 0
    assert client.receipt()["server_verified"] is True
    with pytest.raises(RuntimeError):
        OpenPIDroidClient(spec, pin, FakePolicy({"error": "x"})).infer(packet(spec, context))


@pytest.mark.parametrize("metadata", [{}, {"openpi_commit": "other"}])
def test_client_requires_pinned_server_identity(spec, metadata):
    pin = load_pin(PIN)
    with pytest.raises(ValueError, match="identity"):
        OpenPIDroidClient(spec, pin, FakePolicy({}, metadata))
    client = OpenPIDroidClient(spec, pin, FakePolicy({}, metadata), allow_unverified_server=True)
    assert client.receipt()["server_verified"] is False


def test_connect_fails_fast_without_server(spec):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(OSError):
        OpenPIDroidClient.connect(spec, {}, "127.0.0.1", port, connect_timeout_s=1.)


class Reader:
    def __init__(self, optical_frame, capture_ns, height=360, width=640):
        self.frame = SimpleNamespace(image=np.full((height, width, 3), 200, np.uint8), optical_frame=optical_frame,
                                     sequence=7, capture_ns=capture_ns, sensor_monotonic_ns=capture_ns-32_000_000)

    def latest(self, max_age_ns):
        return self.frame


def test_router_checks_identity_skew_and_keys(spec):
    good = {EXTERIOR: Reader("exterior_optical", 1_000_000_000), WRIST: Reader("wrist_optical", 1_010_000_000)}
    router = DroidCameraRouter(spec, good, max_age_ns=100_000_000, max_skew_ns=20_000_000)
    frames, audit = router.capture()
    assert set(frames) == {EXTERIOR, WRIST}
    assert audit["capture_skew_ms"] == pytest.approx(10.) and audit["sensor_skew_ms"] == pytest.approx(10.)
    swapped = {EXTERIOR: good[WRIST], WRIST: good[EXTERIOR]}
    with pytest.raises(ValueError, match="routed"):
        DroidCameraRouter(spec, swapped, max_age_ns=10**8, max_skew_ns=10**8).capture()
    with pytest.raises(ValueError, match="skew"):
        DroidCameraRouter(spec, good, max_age_ns=10**8, max_skew_ns=5_000_000).capture()
    with pytest.raises(ValueError):
        DroidCameraRouter(spec, {EXTERIOR: good[EXTERIOR]}, max_age_ns=10**8, max_skew_ns=10**8)


def test_packet_from_16x9_frames_is_padded_not_stretched(spec, context):
    pytest.importorskip("openpi_client")
    readers = {EXTERIOR: Reader("exterior_optical", 1), WRIST: Reader("wrist_optical", 1)}
    frames, _ = DroidCameraRouter(spec, readers, max_age_ns=10**8, max_skew_ns=10**8).capture()
    value = droid_packet(spec, context, HOME, 0.04, frames, "put the cube in the bowl")
    assert value.state[7] == pytest.approx(0.5)
    image = np.frombuffer(value.images[0].decode(), np.uint8).reshape(224, 224, 3)
    assert image[0].max() == 0 and image[-1].max() == 0 and image[112, 112, 0] == 200
    assert value.images[0].frame_index == 7
    with pytest.raises(ValueError):
        droid_packet(spec, context, HOME, -0.01, frames, "put the cube in the bowl")


def test_real_openpi_client_over_local_websocket(spec, context):
    pytest.importorskip("openpi_client")
    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve
    received = []

    def handler(ws):
        ws.send(msgpack_numpy.packb(server_identity(load_pin(PIN))))
        for message in ws:
            received.append(msgpack_numpy.unpackb(message))
            ws.send(msgpack_numpy.packb({"actions": raw_chunk(0.9), "server_timing": {"infer_ms": 1.}}))

    with serve(handler, "127.0.0.1", 0) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.socket.getsockname()[1]
        pin = load_pin(ROOT / "configs/policies/pi05_droid_jointpos.json")
        client = OpenPIDroidClient.connect(spec, pin, "127.0.0.1", port)
        chunk, audit = client.infer(packet(spec, context))
        server.shutdown()
    assert client.server_verified
    assert received[0]["prompt"] == "put the cube in the bowl"
    assert received[0][WRIST].shape == (224, 224, 3)
    assert chunk.values[0][7] == pytest.approx(0.9) and audit["server_timing"] == {"infer_ms": 1.}


def test_imports_do_not_load_openpi_client_or_numpy():
    code = ("import sys, learning.integrations.openpi.droid, learning.integrations.openpi.cameras;"
            "print(sorted({'openpi_client', 'numpy', 'websockets'} & set(sys.modules)))")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"
