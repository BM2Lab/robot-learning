"""Client handshake/error checks with a memory stream, no socket permissions needed."""
import hashlib
import io
import struct
from types import SimpleNamespace
import pytest
from contracts.wire import read_message, encode_message
from learning.integrations.isaaclab.local_client import LocalSimulatorClient, RemoteError


def test_client_rejects_remote_error_and_closes_cleanly(monkeypatch):
    spec = SimpleNamespace(to_dict=lambda: {"dummy": "spec"}, to_json=lambda: '{"dummy":"spec"}')
    # Isolate the handshake parser from the full robot fixture in this wire test.
    from learning.integrations.isaaclab import local_client
    monkeypatch.setattr(local_client.PolicyIOSpec, "from_dict", lambda data: spec)
    class PeerStream:
        def __init__(self):
            self.incoming = bytearray()
            self.outgoing = bytearray()
            self.closed = False
        def write(self, value):
            self.incoming.extend(value)
            if len(self.incoming) >= 4 and len(self.incoming) == 4+struct.unpack("!I", self.incoming[:4])[0]:
                request = read_message(io.BytesIO(self.incoming))
                self.incoming.clear()
                if request["op"] == "hello":
                    result = {"contract": spec.to_dict(), "status": {"backend": "simulation"},
                        "receipt": {"contract_sha256": hashlib.sha256(spec.to_json().encode()).hexdigest()}}
                    response = {"protocol": 1, "id": request["id"], "ok": True, "result": result}
                elif request["op"] == "close":
                    response = {"protocol": 1, "id": request["id"], "ok": True, "result": {"closed": True}}
                else:
                    response = {"protocol": 1, "id": request["id"], "ok": False, "error": "rejected"}
                self.outgoing.extend(encode_message(response))
            return len(value)
        def read(self, size):
            result = bytes(self.outgoing[:size])
            del self.outgoing[:size]
            return result
        def flush(self): pass
        def close(self): self.closed = True
    stream = PeerStream()
    class Connected:
        def settimeout(self, value): pass
        def connect(self, value): pass
        def makefile(self, *a, **k): return stream
        def close(self): pass
    monkeypatch.setattr(local_client.socket, "socket", lambda *a, **k: Connected())
    with LocalSimulatorClient("unused", spec) as client:
        with pytest.raises(RemoteError, match="rejected"):
            client.request("unknown")
    assert stream.closed
