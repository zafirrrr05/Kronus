"""features.txt connectivity map, edge 3: "DIGITAL TWIN [DEPLOYMENT hosts]
THE DECOY... In v4's twin-based build, the Decoy is deployed as part of
the twin's topology so the pattern can be proven and demoed." This module
is that topology: a small set of simulated hosts, one of which is where
the real SSH daemon "lives" before MTD relocates it and the Decoy takes
its place on the vacated port — see services/decoy/mtd.py.

Kept deliberately small: spec.md §7 / features.txt Part 5 calls for
10-30 simulated hosts, "enough to prove both lanes and the Decoy" — not a
realistic enterprise network.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimulatedHost:
    host_id: str
    ip: str
    role: str  # "workstation" | "server" | "decoy_host"


@dataclass
class Topology:
    hosts: list[SimulatedHost] = field(default_factory=list)
    decoy_host_id: str | None = None

    def host_by_id(self, host_id: str) -> SimulatedHost | None:
        return next((h for h in self.hosts if h.host_id == host_id), None)


def build_default_topology(host_count: int = 16) -> Topology:
    """A small internal subnet plus one host designated to run the Decoy —
    the twin-hosted deployment relationship from the connectivity map,
    made concrete rather than just asserted in docs.
    """
    hosts = [
        SimulatedHost(host_id=f"host-{i:02d}", ip=f"10.30.0.{i + 1}", role="workstation")
        for i in range(host_count - 1)
    ]
    decoy_host = SimulatedHost(host_id="host-decoy", ip="10.30.0.254", role="decoy_host")
    hosts.append(decoy_host)
    return Topology(hosts=hosts, decoy_host_id=decoy_host.host_id)
