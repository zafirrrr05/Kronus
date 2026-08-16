"""The real-data source. Every demo run and every test case in this repo
that claims to run "on real data" means: rows from this loader.

NSL-KDD ships 41 KDD-Cup-99-style features + label + difficulty, with IP
addresses stripped for anonymization — connections are already reduced to
statistical/categorical features. Two things this module has to do that a
live Zeek feed wouldn't need:

1. Map each row's raw `label` (one of ~35 specific attack names) onto
   KRONUS's fixed Label enum (flood / port_scan / benign / ...) — see
   ATTACK_CATEGORY below. R2L/U2R rows have no clean KRONUS-label home
   (there is no schema slot for them — see libs/constants.py's Label enum,
   which is fixed by spec.md §3.3) so they're folded into Bouncer/Detective
   training as generic non-benign examples rather than force-fit onto a
   label the system doesn't define. Reported honestly in docs, not papered
   over.
2. Reconstruct plausible source/dest IPs, since NSL-KDD has none. See
   `_synthetic_ip` — hosts are derived deterministically from each row's
   own protocol/service/flag/dst_host_count, so a run of scan-shaped rows
   collapses onto a small source cluster fanning out to many destinations
   (the actual graph shape the Detective needs), grounded in the dataset's
   own precomputed connection-rate features rather than invented ones.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from libs.constants import DataOrigin, Label, Protocol
from libs.observability import observe

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

# Standard NSL-KDD / KDD-Cup-99 attack-name -> category mapping. "normal" and
# every DoS/Probe subtype are covered explicitly (those two categories map
# cleanly onto KRONUS's flood/port_scan labels); R2L and U2R subtypes are
# listed too, tagged for the "generic attack signal, not a specific KRONUS
# label" treatment described above.
ATTACK_CATEGORY: dict[str, str] = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos", "mailbomb": "dos",
    # Probe
    "satan": "probe", "ipsweep": "probe", "nmap": "probe",
    "portsweep": "probe", "mscan": "probe", "saint": "probe",
    # R2L
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l",
    "spy": "r2l", "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l",
    "snmpgetattack": "r2l", "httptunnel": "r2l", "sendmail": "r2l",
    "named": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "xterm": "u2r", "ps": "u2r", "sqlattack": "u2r",
}

CATEGORY_TO_LABEL: dict[str, Label] = {
    "normal": Label.BENIGN,
    "dos": Label.FLOOD,
    "probe": Label.PORT_SCAN,
    # r2l/u2r deliberately absent: no direct KRONUS label; see module docstring.
}

# A small, well-known service -> port map, enough to give common NSL-KDD
# `service` values a plausible dest_port; anything else falls back to None
# rather than a fabricated number.
SERVICE_PORT: dict[str, int] = {
    "http": 80, "http_443": 443, "ftp": 21, "ftp_data": 20, "ssh": 22,
    "telnet": 23, "smtp": 25, "domain": 53, "domain_u": 53, "pop_3": 110,
    "imap4": 143, "sql_net": 1433, "ldap": 389, "netbios_ns": 137,
    "netbios_dgm": 138, "netbios_ssn": 139, "printer": 515, "nntp": 119,
    "exec": 512, "login": 513, "shell": 514, "whois": 43, "finger": 79,
    "gopher": 70, "irc": 194, "X11": 6000, "Z39_50": 210, "auth": 113,
}

PROTOCOL_MAP: dict[str, Protocol] = {
    "tcp": Protocol.TCP,
    "udp": Protocol.UDP,
    "icmp": Protocol.ICMP,
}


def _synthetic_ip(prefix: str, *parts: str) -> str:
    """prefix is a 2-octet string (e.g. "10.10"); returns a valid 4-octet
    IPv4 address by appending exactly 2 more octets derived from the hash.
    """
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    h = int(digest[:8], 16)
    return f"{prefix}.{(h >> 8) & 0xFF}.{h & 0xFF}"


@dataclass(frozen=True)
class NSLKDDRow:
    """One reconstructed flow, ready to become a TelemetryEvent
    (services/telemetry_exporter) or a training example (bouncer/detective).
    """

    source_ip: str
    dest_ip: str
    source_port: int | None
    dest_port: int | None
    protocol: Protocol
    total_bytes: int
    duration_ms: int
    raw_label: str
    category: str  # normal | dos | probe | r2l | u2r
    kronus_label: Label | None  # None for r2l/u2r — see module docstring
    difficulty: int
    features: dict[str, float]  # the 41 raw features, for model training
    origin: DataOrigin = DataOrigin.REAL


def _row_to_nslkdd_row(row: pd.Series, index: int) -> NSLKDDRow:
    category = ATTACK_CATEGORY.get(row["label"], "r2l")  # unseen rare label -> treat as r2l-tier signal, never silently dropped
    bucket = min(int(row["dst_host_count"]) // 25, 10)
    # Source-identity clustering needs to be category-aware, not just
    # temporally bounded. Caught for real (see services/detective/train.py's
    # module history): even with a fixed row-position session bound, common
    # normal-traffic signatures (e.g. plain "tcp, http, SF" is a large
    # fraction of any real capture) still collapsed many unrelated normal
    # connections onto a handful of synthetic sources, producing artificial
    # fan-out that looked exactly like scanning — real IDS ground truth
    # doesn't have this problem because attacks genuinely DO concentrate
    # from few real hosts while normal traffic genuinely DOES disperse
    # across many; reconstructing that asymmetry (using the label available
    # for training data, not for live detection) is what the fix needs to
    # simulate, not paper over with a bigger bucket:
    #   - dos/probe: cluster broadly (unbounded session) — a real attack's
    #     traffic sharing a signature really should collapse onto few hosts.
    #   - normal/r2l/u2r: session = this row's own index, i.e. effectively
    #     unique — real normal traffic comes from many different real
    #     clients and must not be merged just for sharing a common service.
    session = 0 if category in ("dos", "probe") else index
    source_ip = _synthetic_ip(
        "10.10", str(row["protocol_type"]), str(row["service"]), str(row["flag"]),
        str(bucket), str(session),
    )
    dest_ip = _synthetic_ip("10.20", str(row["service"]), str(index % 4999))
    protocol = PROTOCOL_MAP.get(str(row["protocol_type"]), Protocol.OTHER)
    port = SERVICE_PORT.get(str(row["service"]))
    features = {c: float(row[c]) for c in NSL_KDD_COLUMNS[:41] if _is_num(row[c])}
    return NSLKDDRow(
        source_ip=source_ip,
        dest_ip=dest_ip,
        source_port=None,
        dest_port=port,
        protocol=protocol,
        total_bytes=int(row["src_bytes"]) + int(row["dst_bytes"]),
        duration_ms=int(row["duration"]) * 1000,
        raw_label=str(row["label"]),
        category=category,
        kronus_label=CATEGORY_TO_LABEL.get(category),
        difficulty=int(row["difficulty"]),
        features=features,
    )


def _is_num(v) -> bool:
    return isinstance(v, (int, float))


def load_nsl_kdd(path: str | Path) -> list[NSLKDDRow]:
    """Loads one NSL-KDD split (KDDTrain+.txt or KDDTest+.txt — see
    scripts/download_data.py for how to fetch them). Categorical columns
    (protocol_type, service, flag) stay as strings here; numeric feature
    encoding for model training happens in services/bouncer/features.py and
    services/graph_builder, which is where it belongs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first "
            "(see docs/setup.md)."
        )
    with observe("digital_twin", "load_nsl_kdd", path=str(path)):
        df = pd.read_csv(path, names=NSL_KDD_COLUMNS, header=None)
        return [_row_to_nslkdd_row(row, i) for i, row in df.iterrows()]


def load_nsl_kdd_dataframe(path: str | Path) -> pd.DataFrame:
    """Raw dataframe access for model training code that wants vectorized
    pandas operations rather than a list of dataclasses."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first "
            "(see docs/setup.md)."
        )
    df = pd.read_csv(path, names=NSL_KDD_COLUMNS, header=None)
    df["category"] = df["label"].map(lambda label: ATTACK_CATEGORY.get(label, "r2l"))
    return df
