import base64
import io
import struct
import pytest
from contracts.episodes import SampleContext
from contracts.samples import RGBSample, ObservationPacket
from contracts.wire import encode_message, read_message, write_message, MAX_MESSAGE_BYTES


def image(context):
    return RGBSample("camera", "optical", 1, 2, 0, context, base64.b64encode(bytes(range(6))).decode())


def test_packet_roundtrip():
    context = SampleContext("episode", 0, 0, "simulation")
    packet = ObservationPacket("contract", context, (0.,)*7, (image(context),), "move")
    assert ObservationPacket.from_json(packet.to_json()) == packet
    assert packet.images[0].decode() == bytes(range(6))


def test_rgb_rejects_wrong_geometry_encoding():
    context = SampleContext("episode", 0, 0, "simulation")
    with pytest.raises(ValueError):
        RGBSample("camera", "optical", 2, 2, 0, context, "AAAA")
    with pytest.raises(ValueError):
        RGBSample("camera", "optical", 1, 2, 0, context, "not base64!")
    with pytest.raises(ValueError):
        RGBSample("camera", "optical", 1024, 1024, 0, context, "")


def test_capture_context_must_match():
    context = SampleContext("episode", 0, 0, "simulation")
    with pytest.raises(ValueError):
        ObservationPacket("contract", SampleContext("other", 0, 0, "simulation"), (0.,)*7, (image(context),), "move")


def test_stream_roundtrip_and_partial_io():
    class Partial(io.BytesIO):
        def read(self, n=-1):
            return super().read(min(n, 3))
        def write(self, data):
            return super().write(data[:3])
    stream = Partial()
    write_message(stream, {"sample": [1, "two"]})
    stream.seek(0)
    assert read_message(stream) == {"sample": [1, "two"]}


@pytest.mark.parametrize("raw", [b'{"value":NaN}', b'{"value":1,"value":2}', b'[]'])
def test_invalid_json_frames(raw):
    with pytest.raises(ValueError):
        read_message(io.BytesIO(struct.pack("!I", len(raw))+raw))


def test_truncated_and_oversized_frames():
    with pytest.raises(ConnectionError):
        read_message(io.BytesIO(encode_message({"x": 1})[:-1]))
    with pytest.raises(ValueError):
        read_message(io.BytesIO(struct.pack("!I", MAX_MESSAGE_BYTES+1)))
    with pytest.raises(ValueError):
        encode_message({"x": float("nan")})
