"""features.txt component 1 (Digital Twin): "generates unlimited attacks
with zero real-world blast radius... every attack comes with a free,
perfect label." This module is that generator.

Scope discipline, stated once so it doesn't need repeating at every call
site: output from here is SYNTHETIC (DataOrigin.SYNTHETIC) and is consumed
ONLY by services/drift_watcher's improvement loop and services/*/train.py's
augmentation path. The demo and every test case that claims to run "on
real data" import twin.nsl_kdd, never this module — see docs/instructions.md.

Reuses NSLKDDRow as the output shape rather than inventing a parallel
"SyntheticFlow" type: a flow record is a flow record regardless of origin,
and telemetry_exporter's real->event conversion logic works unmodified
either way. The one attack class real NSL-KDD data structurally can't
support — lateral_movement, which needs genuine multi-hop host-to-host
sequencing NSL-KDD's row-independent schema doesn't carry — is why this
module exists at all; flood and port_scan are here mainly for volume
during retraining, not because real data can't already teach those.
"""

from __future__ import annotations

import random

from libs.constants import DataOrigin, Label, Protocol
from libs.observability import observe
from twin.nsl_kdd import NSLKDDRow
from twin.topology import Topology, build_default_topology

_BASE_FEATURES: dict[str, float] = {
    "duration": 0.0, "src_bytes": 0.0, "dst_bytes": 0.0, "land": 0.0,
    "wrong_fragment": 0.0, "urgent": 0.0, "hot": 0.0, "num_failed_logins": 0.0,
    "logged_in": 0.0, "num_compromised": 0.0, "root_shell": 0.0,
    "su_attempted": 0.0, "num_root": 0.0, "num_file_creations": 0.0,
    "num_shells": 0.0, "num_access_files": 0.0, "num_outbound_cmds": 0.0,
    "is_host_login": 0.0, "is_guest_login": 0.0, "count": 1.0, "srv_count": 1.0,
    "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0, "same_srv_rate": 1.0, "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0, "dst_host_count": 1.0, "dst_host_srv_count": 1.0,
    "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
}


def _features(**overrides: float) -> dict[str, float]:
    return {**_BASE_FEATURES, **overrides}


def generate_flood_scenario(topology: Topology, n_events: int = 500) -> list[NSLKDDRow]:
    """One source hammering one destination — the twin's version of Case 1
    (features.txt Part 3 / KRONUS_v4.pdf Case 1)."""
    attacker_ip = "203.0.113.66"
    victim = random.choice(topology.hosts)
    return [
        NSLKDDRow(
            source_ip=attacker_ip, dest_ip=victim.ip, source_port=random.randint(1024, 65535),
            dest_port=80, protocol=Protocol.TCP, total_bytes=random.randint(40, 200),
            duration_ms=random.randint(0, 5), raw_label="synthetic_flood", category="dos",
            kronus_label=Label.FLOOD, difficulty=0,
            features=_features(count=500.0, srv_count=500.0, same_srv_rate=1.0,
                                dst_host_same_src_port_rate=1.0, dst_host_count=1.0),
            origin=DataOrigin.SYNTHETIC,
        )
        for _ in range(n_events)
    ]


def generate_port_scan_scenario(topology: Topology) -> list[NSLKDDRow]:
    """One source, many destination ports across many hosts — Case 2."""
    attacker_ip = "203.0.113.77"
    rows = []
    for host in topology.hosts:
        for port in (22, 23, 80, 443, 3389, 8080):
            rows.append(NSLKDDRow(
                source_ip=attacker_ip, dest_ip=host.ip, source_port=random.randint(1024, 65535),
                dest_port=port, protocol=Protocol.TCP, total_bytes=random.randint(40, 80),
                duration_ms=random.randint(0, 3), raw_label="synthetic_scan", category="probe",
                kronus_label=Label.PORT_SCAN, difficulty=0,
                features=_features(dst_host_count=float(len(topology.hosts)), diff_srv_rate=0.8,
                                    same_srv_rate=0.1, srv_diff_host_rate=0.8),
                origin=DataOrigin.SYNTHETIC,
            ))
    return rows


def generate_lateral_movement_scenario(topology: Topology, hop_count: int = 5) -> list[NSLKDDRow]:
    """A compromised host walks to several internal peers in sequence,
    logging in each time — the multi-hop, host-to-host pattern real
    flow-level datasets don't carry (see module docstring). This is the
    scenario the Detective is trained on for lateral_movement specifically.
    """
    if len(topology.hosts) < hop_count + 1:
        raise ValueError("topology needs more hosts than hop_count")
    hop_hosts = random.sample(topology.hosts, hop_count + 1)
    rows = []
    for src, dst in zip(hop_hosts, hop_hosts[1:], strict=False):
        rows.append(NSLKDDRow(
            source_ip=src.ip, dest_ip=dst.ip, source_port=random.randint(1024, 65535),
            dest_port=445, protocol=Protocol.TCP, total_bytes=random.randint(2000, 8000),
            duration_ms=random.randint(50, 400), raw_label="synthetic_lateral_movement",
            category="lateral_movement", kronus_label=Label.LATERAL_MOVEMENT, difficulty=0,
            features=_features(logged_in=1.0, num_compromised=1.0, hot=1.0,
                                dst_host_count=3.0, same_srv_rate=1.0),
            origin=DataOrigin.SYNTHETIC,
        ))
    return rows


def generate_benign_scenario(topology: Topology, n_events: int = 500) -> list[NSLKDDRow]:
    """Ordinary internal chatter — Locust-style background load, standing
    in for normal staff traffic (features.txt component 1)."""
    rows = []
    for _ in range(n_events):
        src, dst = random.sample(topology.hosts, 2)
        rows.append(NSLKDDRow(
            source_ip=src.ip, dest_ip=dst.ip, source_port=random.randint(1024, 65535),
            dest_port=random.choice((80, 443, 22, 25)), protocol=Protocol.TCP,
            total_bytes=random.randint(100, 4000), duration_ms=random.randint(5, 500),
            raw_label="synthetic_normal", category="normal", kronus_label=Label.BENIGN,
            difficulty=0, features=_features(logged_in=1.0), origin=DataOrigin.SYNTHETIC,
        ))
    return rows


def generate_mixed_probe_set(seed: int | None = None) -> list[NSLKDDRow]:
    """One batch spanning all four scenarios — what services/drift_watcher
    injects as a continuous, perfectly-labeled probe set (features.txt
    component 9's "twin-injected probes")."""
    if seed is not None:
        random.seed(seed)
    topology = build_default_topology()
    with observe("digital_twin", "generate_mixed_probe_set"):
        return (
            generate_benign_scenario(topology, n_events=200)
            + generate_flood_scenario(topology, n_events=100)
            + generate_port_scan_scenario(topology)
            + generate_lateral_movement_scenario(topology)
        )
