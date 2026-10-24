"""Client for franka-runtime's DROID chunk service (its docs/droid-runtime.md protocol).

Standard library + contracts only. The runtime keeps all authority: this client
asks for a lease, registers a clock estimate, keeps a heartbeat and submits chunks.
It cannot reset faults. A rejected command does not consume its sequence number.
"""
import socket
import threading
import time

from contracts.clock import ClockOffsetEstimate, ClockProbe
from contracts.droid import DroidIOSpec
from contracts.droid_session import DroidSessionOpen, DroidSessionReady, chunk_envelope
from contracts.franka_state import FrankaStateSample
from contracts.sessions import CommandReceipt, FaultReport, LeaseRequest, payload_sha256
from contracts.wire import PROTOCOL_VERSION, read_message, write_message


class RuntimeRequestError(RuntimeError):
    pass


class DroidRuntimeClient:
    def __init__(self, host, port, spec, *, token, client_clock_id, owner_id, lease_duration_ns,
                 max_uncertainty_ns, probes=8, heartbeat_period_s=0.25, timeout_s=2., clock=time.monotonic_ns):
        if not isinstance(spec, DroidIOSpec):
            raise ValueError("Typed DroidIOSpec required")
        if not token:
            raise ValueError("Session token required")
        self.spec, self._token, self.clock = spec, token, clock
        self.client_clock_id, self.owner_id = client_clock_id, owner_id
        self.lease_duration_ns, self.max_uncertainty_ns = lease_duration_ns, max_uncertainty_ns
        self.probe_count, self.heartbeat_period_s = probes, heartbeat_period_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._heartbeat = None
        self._counter = 0
        self.ready = self.estimate = self.hello = None
        self.next_sequence = 1
        self.heartbeat_probes = []
        self.socket = socket.create_connection((host, port), timeout=timeout_s)
        self.stream = self.socket.makefile("rwb", buffering=0)

    def request(self, op, payload=None):
        with self._lock:
            self._counter += 1
            identity = str(self._counter)
            write_message(self.stream, {"protocol": PROTOCOL_VERSION, "id": identity, "op": op, "payload": payload or {}})
            response = read_message(self.stream)
        if response.get("protocol") != PROTOCOL_VERSION or response.get("id") != identity:
            raise RuntimeRequestError("Mismatched response identity/version")
        if not response.get("ok"):
            raise RuntimeRequestError(f"{op}: {response.get('error')}")
        return response["result"]

    def _probe(self):
        sent = self.clock()
        reply = self.request("clock_probe")
        received = self.clock()
        return ClockProbe(self.client_clock_id, reply["server_clock_id"], sent, reply["server_receive_ns"],
                          reply["server_send_ns"], received)

    def open(self, request_id, *, start=True):
        """hello → probes → open → clock_register → start. Returns DroidSessionReady.

        ``start=False`` stops before ``start``: native control never starts, and a release
        then cancels the session before start.
        """
        self.hello = self.request("hello")
        if self.hello["contract_sha256"] != payload_sha256(self.spec):
            raise RuntimeRequestError("Runtime serves a different DROID contract")
        if self.hello["robot_id"] != self.spec.robot.robot_id:
            raise RuntimeRequestError("Runtime reports a different robot")
        probes = [self._probe() for _ in range(self.probe_count)]
        estimate = ClockOffsetEstimate.from_probes(probes, max_uncertainty_ns=self.max_uncertainty_ns)
        lease_request = LeaseRequest(request_id, self.spec.robot.robot_id, self.owner_id, self.hello["server_clock_id"],
                                     estimate.to_server(self.clock()), self.lease_duration_ns)
        session = DroidSessionOpen(self.spec, payload_sha256(self.spec), lease_request, self.client_clock_id)
        ready = DroidSessionReady.from_dict(self.request("open", {"session": session.to_dict(), "token": self._token}))
        ready.validate_for(session)
        registered = ClockOffsetEstimate.from_dict(self.request("clock_register", {"probes": [p.to_dict() for p in probes]}))
        if (registered.offset_ns, registered.uncertainty_ns) != (estimate.offset_ns, estimate.uncertainty_ns):
            raise RuntimeRequestError("Runtime registered a different clock estimate")
        if start:
            self.request("start")
        self.ready, self.estimate = ready, estimate
        self._heartbeat = threading.Thread(target=self._heartbeat_loop, name="droid-heartbeat", daemon=True)
        self._heartbeat.start()
        return ready

    def _heartbeat_loop(self):
        while not self._stop.wait(self.heartbeat_period_s):
            try:
                probe = self._probe()
            except (OSError, RuntimeRequestError, ValueError):
                return  # The runtime's heartbeat timeout will stop the robot.
            self.heartbeat_probes.append(probe)
            del self.heartbeat_probes[:-64]

    def to_client(self, server_ns):
        return server_ns-self.estimate.offset_ns

    def state(self):
        sample = FrankaStateSample.from_dict(self.request("state"))
        if sample.robot_id != self.spec.robot.robot_id or sample.clock_id != self.hello["server_clock_id"]:
            raise RuntimeRequestError("State sample from a different robot or clock")
        return sample

    def submit(self, chunk):
        """Bind and submit with the next sequence; only an accepted command advances it."""
        envelope = chunk_envelope(self.ready.lease, self.next_sequence, chunk, self.estimate,
                                  client_now_ns=self.clock(), capability=self.ready.capability_id)
        receipt = CommandReceipt.from_dict(self.request("submit", {"envelope": envelope.to_dict(),
                                                                    "chunk": chunk.to_dict()}))
        if receipt.status == "accepted":
            self.next_sequence += 1
        return envelope, receipt

    def receipts(self):
        result = self.request("receipts")
        fault = None if result["fault"] is None else FaultReport.from_dict(result["fault"])
        return [CommandReceipt.from_dict(r) for r in result["receipts"]], fault, result

    def release(self, reason):
        return self.request("release", {"reason": reason})

    def close(self):
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(2)
        try:
            self.request("close")
        except (OSError, RuntimeRequestError, ValueError):
            pass
        finally:
            self.stream.close()
            self.socket.close()
