"""Local simulator RPC client. Standard library + contracts only, no Isaac/ML imports."""
import socket
import hashlib
from contracts.policies import PolicyIOSpec
from contracts.samples import ObservationPacket
from contracts.wire import PROTOCOL_VERSION, read_message, write_message


class RemoteError(RuntimeError):
    pass


class LocalSimulatorClient:
    def __init__(self, socket_path, spec, timeout_s=30):
        self.spec = spec
        self.counter = 0
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(timeout_s)
        self.stream = None
        try:
            self.connection.connect(str(socket_path))
            self.stream = self.connection.makefile("rwb", buffering=0)
            receipt = self.request("hello", {"contract": spec.to_dict()})
            if PolicyIOSpec.from_dict(receipt["contract"]) != spec or receipt["status"]["backend"] != "simulation":
                raise ValueError("Remote backend or contract mismatch")
            if receipt["receipt"]["contract_sha256"] != hashlib.sha256(spec.to_json().encode()).hexdigest():
                raise ValueError("Remote contract fingerprint mismatch")
            self.receipt = receipt
        except Exception:
            self._dispose()
            raise

    def request(self, op, payload=None):
        self.counter += 1
        identity = str(self.counter)
        write_message(self.stream, {"protocol": PROTOCOL_VERSION, "id": identity,
            "op": op, "payload": {} if payload is None else payload})
        response = read_message(self.stream)
        if (response.get("protocol") != PROTOCOL_VERSION or type(response.get("protocol")) is not int
            or response.get("id") != identity or type(response.get("ok")) is not bool):
            raise ValueError("Mismatched response identity/version")
        if not response["ok"]:
            if set(response) != {"protocol", "id", "ok", "error"} or not isinstance(response["error"], str):
                raise ValueError("Malformed remote error")
            raise RemoteError(response["error"])
        if set(response) != {"protocol", "id", "ok", "result"} or not isinstance(response["result"], dict):
            raise ValueError("Malformed remote response")
        return response["result"]

    def _packet(self, result):
        packet = ObservationPacket.from_dict(result)
        packet.validate_for(self.spec)
        if packet.context.clock != "simulation":
            raise ValueError("Simulator observations must use the simulation clock domain")
        return packet

    def observe(self):
        return self._packet(self.request("observe"))

    def reset(self, seed=None):
        return self._packet(self.request("reset", {} if seed is None else {"seed": seed}))

    def submit(self, chunk):
        return self.request("submit", {"chunk": chunk.to_dict()})

    def advance(self, context):
        result = self.request("advance", {"context": context.to_dict()})
        packet = self._packet(result["observation"])
        return packet, result["terminal"], result["status"]

    def _dispose(self):
        if self.stream is not None:
            self.stream.close()
        self.connection.close()

    def close(self):
        try:
            self.request("close")
        finally:
            self._dispose()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.close()
        else:
            self._dispose()  # Never mask an earlier failure with a second RPC error.
