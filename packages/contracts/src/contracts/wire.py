"""Bounded JSON framing codec for local byte streams. Never pickle executable objects."""
import json
import struct

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 2*1024*1024


def encode_message(value):
    if not isinstance(value, dict):
        raise ValueError("Wire message must be an object")
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if not 0 < len(raw) <= MAX_MESSAGE_BYTES:
        raise ValueError("Wire message exceeds size bound")
    return struct.pack("!I", len(raw))+raw


def _read_exact(stream, size):
    parts = []
    while size:
        part = stream.read(size)
        if not part:
            raise ConnectionError("Peer disconnected before a complete frame")
        parts.append(part)
        size -= len(part)
    return b"".join(parts)


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def read_message(stream):
    size = struct.unpack("!I", _read_exact(stream, 4))[0]
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid wire frame size")
    value = json.loads(_read_exact(stream, size), parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("Wire message must be an object")
    return value


def write_message(stream, value):
    raw = encode_message(value)
    view = memoryview(raw)
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise ConnectionError("Peer disconnected during write")
        view = view[written:]
    stream.flush()
