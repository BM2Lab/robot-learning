"""Cross-host clock offset from timestamped probes (NTP-style), standard library only.

Each host stamps with its own CLOCK_MONOTONIC; clocks are never assumed shared.
A probe is client send → server receive → server send → client receive. The
offset maps client time to server time; its uncertainty is half the network
round trip. Deadlines are converted conservatively (earlier by the uncertainty).
"""
from dataclasses import dataclass
from ._validation import JsonContract, integer, text


@dataclass(frozen=True)
class ClockProbe(JsonContract):
    client_clock_id: str
    server_clock_id: str
    client_send_ns: int
    server_receive_ns: int
    server_send_ns: int
    client_receive_ns: int

    def __post_init__(self):
        text(self.client_clock_id, "client_clock_id")
        text(self.server_clock_id, "server_clock_id")
        for field in ("client_send_ns", "server_receive_ns", "server_send_ns", "client_receive_ns"):
            integer(getattr(self, field), field)
        if self.client_receive_ns < self.client_send_ns or self.server_send_ns < self.server_receive_ns:
            raise ValueError("Probe timestamps must be monotonic on each host")
        if self.round_trip_ns < 0:
            raise ValueError("Server processing exceeds the client round trip")

    @property
    def round_trip_ns(self):
        return (self.client_receive_ns-self.client_send_ns)-(self.server_send_ns-self.server_receive_ns)

    @property
    def offset_ns(self):
        """server_time ≈ client_time + offset."""
        return ((self.server_receive_ns-self.client_send_ns)+(self.server_send_ns-self.client_receive_ns))//2


@dataclass(frozen=True)
class ClockOffsetEstimate(JsonContract):
    client_clock_id: str
    server_clock_id: str
    offset_ns: int
    uncertainty_ns: int
    samples: int
    estimated_at_client_ns: int

    def __post_init__(self):
        text(self.client_clock_id, "client_clock_id")
        text(self.server_clock_id, "server_clock_id")
        if type(self.offset_ns) is not int:
            raise ValueError("offset_ns must be an integer")
        integer(self.uncertainty_ns, "uncertainty_ns")
        integer(self.samples, "samples", 1)
        integer(self.estimated_at_client_ns, "estimated_at_client_ns")

    @classmethod
    def from_probes(cls, probes, *, max_uncertainty_ns):
        """Minimum-round-trip probe wins; reject when even that exceeds the bound."""
        probes = tuple(probes)
        if not probes or not all(isinstance(p, ClockProbe) for p in probes):
            raise ValueError("Typed clock probes required")
        if len({(p.client_clock_id, p.server_clock_id) for p in probes}) != 1:
            raise ValueError("Probes mix clock domains")
        integer(max_uncertainty_ns, "max_uncertainty_ns", 1)
        best = min(probes, key=lambda p: p.round_trip_ns)
        uncertainty = (best.round_trip_ns+1)//2
        if uncertainty > max_uncertainty_ns:
            raise ValueError(f"Clock uncertainty {uncertainty} ns exceeds bound {max_uncertainty_ns} ns")
        return cls(best.client_clock_id, best.server_clock_id, best.offset_ns, uncertainty, len(probes),
                   max(p.client_receive_ns for p in probes))

    def to_server(self, client_ns):
        return integer(client_ns, "client_ns")+self.offset_ns

    def deadline_to_server(self, client_deadline_ns):
        """Conservative: the converted deadline is never later than the true one."""
        return self.to_server(client_deadline_ns)-self.uncertainty_ns
