"""UNSW-NB15 external-validation data source.

UNSW-NB15 is the third independent dataset KRONUS is evaluated on, alongside
NSL-KDD (twin/nsl_kdd.py, the training set) and CIC-IDS2017
(twin/cicids2017.py, the first external set). It was captured at the Australian
Centre for Cyber Security (ACCS), UNSW Canberra, in 2015 by the IXIA PerfectStorm
tool, which generated a hybrid of real modern normal activity and synthetic
contemporary attack traffic.

Official dataset:
    https://research.unsw.edu.au/projects/unsw-nb15-dataset
Citation:
    Nour Moustafa and Jill Slay, "UNSW-NB15: a comprehensive data set for
    network intrusion detection systems (UNSW-NB15 network data set)",
    2015 Military Communications and Information Systems Conference (MilCIS).

WHY THIS LOADER IS NOT LIKE twin/cicids2017.py
----------------------------------------------
CIC-IDS2017's labelled-flow CSVs carry real Source IP / Destination IP / ports,
so its graph shape is "the genuine article" (see that module's docstring). The
UNSW-NB15 *pre-split* release — UNSW_NB15_training-set.csv / testing-set.csv,
the author-supplied split and the part small enough to be practical here — ships
NO IP addresses and no port numbers. It is therefore in the same situation as
NSL-KDD, and this loader solves it the same way, deliberately: sources and
destinations are reconstructed deterministically from each row's own recorded
connection-rate counters (`ct_srv_src`, `ct_dst_ltm`, `ct_src_dport_ltm`, ...),
which is exactly the "grounded in the dataset's own precomputed connection-rate
features rather than invented ones" principle twin/nsl_kdd.py's `_synthetic_ip`
documents. See `_reconstruct_hosts` below for how that is applied here.

The full 100 GB PCAP corpus and the headerless UNSW-NB15_1..4.csv files DO carry
real addresses but are out of scope for a laptop-scale experiment (see
docs/experiments_guide.md's disk budget). Reported honestly in the metrics:
`"synthetic_hosts": true`.

LABEL MAPPING
-------------
UNSW-NB15's `attack_cat` is a 10-value multi-class column with a companion
binary `label` (0=normal, 1=attack). Mapping onto KRONUS's fixed Label enum
(libs/constants.py, spec.md §3.3) follows the same rule as every other loader:

  Normal          -> normal   (Label.BENIGN)
  DoS             -> dos      (Label.FLOOD)      <- the only flood family here
  Reconnaissance  -> probe    (Label.PORT_SCAN)  <- drives the Detective lane
  Generic, Exploits, Fuzzers, Analysis, Backdoor, Shellcode, Worms
                  -> r2l      (no clean KRONUS label — generic non-benign
                               signal, identical treatment to NSL-KDD's r2l/u2r)

Unlike CIC-IDS2017's two DrDoS-only days, UNSW-NB15 carries a genuine
Reconnaissance class, so BOTH KRONUS detection lanes can be trained on it —
the Bouncer on DoS-vs-normal, the Detective on Reconnaissance-vs-normal.

The output type is the SAME NSLKDDRow shape every other loader produces, so
the whole downstream pipeline (services/telemetry_exporter/converters,
FlowFeaturizer, WindowedGraphBuilder) works with no UNSW-specific code path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from libs.constants import DataOrigin, Label, Protocol
from libs.observability import observe
from twin.nsl_kdd import NSLKDDRow

# The author-supplied pre-split files. Both are optional: a caller can train on
# the training file alone (the runner does its own stratified 67/33 split), and
# the testing file is used when present for a genuine held-out check.
UNSW_NB15_FILES = [
    "UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv",
]

# attack_cat (casefolded after stripping) -> KRONUS category. Every one of the
# ten documented values is listed explicitly so an unrecognized label is a
# visible gap rather than a silent default.
UNSW_ATTACK_CATEGORY: dict[str, str] = {
    "normal": "normal",
    # Flood family
    "dos": "dos",
    # Probe family
    "reconnaissance": "probe",
    # No clean KRONUS label -> generic non-benign signal (as NSL-KDD r2l/u2r).
    "generic": "r2l",
    "exploits": "r2l",
    "fuzzers": "r2l",
    "analysis": "r2l",
    "backdoor": "r2l",
    "shellcode": "r2l",
    "worms": "r2l",
}

CATEGORY_TO_LABEL: dict[str, Label] = {
    "normal": Label.BENIGN,
    "dos": Label.FLOOD,
    "probe": Label.PORT_SCAN,
    # r2l deliberately absent — no direct KRONUS label; see module docstring.
}

# UNSW-NB15's `proto` column holds IANA protocol names, not numbers (133 distinct
# values, most of them exotic). Only the three KRONUS's Protocol enum can express
# are mapped; everything else falls back to OTHER rather than being force-fit.
PROTOCOL_MAP: dict[str, Protocol] = {
    "tcp": Protocol.TCP,
    "udp": Protocol.UDP,
    "icmp": Protocol.ICMP,
}

# The 13 `service` values present in the release -> a plausible destination
# port. '-' means "no well-known service" and maps to None, never a fabricated
# number. Parallels twin/nsl_kdd.py's SERVICE_PORT.
SERVICE_PORT: dict[str, int] = {
    "dhcp": 67,
    "dns": 53,
    "ftp": 21,
    "ftp-data": 20,
    "http": 80,
    "irc": 194,
    "pop3": 110,
    "radius": 1812,
    "smtp": 25,
    "snmp": 161,
    "ssh": 22,
    "ssl": 443,
}

_ATTACK_CAT = "attack_cat"
_LABEL_BINARY = "label"

# Columns consumed as flow identity; everything else numeric becomes `features`.
_IDENTITY_COLS = {"id", _ATTACK_CAT, _LABEL_BINARY}


def _synthetic_ip(prefix: str, *parts: str) -> str:
    """Deterministic 4-octet IPv4 from a 2-octet prefix + hashed parts.

    Identical construction to twin/nsl_kdd.py's helper of the same name, kept
    local (rather than imported) in the same spirit as the other loaders'
    duplicated _safe_float/_safe_int helpers: each loader stays independently
    readable, and the two are free to diverge if their needs do.
    """
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    h = int(digest[:8], 16)
    return f"{prefix}.{(h >> 8) & 0xFF}.{h & 0xFF}"


def _reconstruct_hosts(row: pd.Series, index: int, category: str) -> tuple[str, str]:
    """Reconstruct (source_ip, dest_ip) for a row that has no real addresses.

    The graph shape the Detective must learn is a *topological* asymmetry:
    scanning/flooding traffic genuinely concentrates from few sources onto many
    destinations (or many sources onto one victim), while normal traffic
    genuinely disperses across many independent clients. twin/nsl_kdd.py
    derives that asymmetry from `dst_host_count` with a category-aware session
    rule; UNSW-NB15 has its own equivalents of exactly those counters, so the
    same rule is applied to them:

    - Source: `session` is pinned to 0 for the attack categories whose real
      traffic *should* collapse onto few hosts (dos, probe), and set to the
      row's own index otherwise, so normal/r2l rows never get merged just for
      sharing a service signature. The bucket is UNSW's own `ct_srv_src`
      (connections to the same service from the same source in the last 100),
      which is the closest analogue of NSL-KDD's dst_host_count.
    - Destination: a probe must fan OUT, so its destination varies with the row
      index (a scan hitting many victims); a flood must concentrate ON a
      victim, so its destination is pinned to the bucket; everything else is
      keyed by (service, index) so normal traffic spreads across destinations.
    """
    bucket = min(int(row.get("ct_srv_src", 1)) // 25, 10)
    proto = str(row.get("proto", "other"))
    service = str(row.get("service", "-"))
    state = str(row.get("state", "no"))

    session = 0 if category in ("dos", "probe") else index
    source_ip = _synthetic_ip("10.10", proto, service, state, str(bucket), str(session))

    if category == "probe":
        dest_key = str(index % 4999)          # scan: many distinct victims
    elif category == "dos":
        dest_key = f"victim:{bucket}"          # flood: few victim hosts
    else:
        dest_key = f"{service}:{index % 4999}"  # normal: dispersed
    dest_ip = _synthetic_ip("10.20", service, dest_key)

    return source_ip, dest_ip


def _safe_float(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f != f or f in (float("inf"), float("-inf")):
        return 0.0
    return f


def _is_numlike(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def _normalize_label(raw) -> str:
    return " ".join(str(raw).split()).strip().casefold()


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The published CSVs carry a UTF-8 BOM before the first header ('\\ufeffid')
    and inconsistent padding. Strip both so callers can use clean names."""
    return df.rename(columns={c: str(c).replace("﻿", "").strip() for c in df.columns})


def _row_to_nslkdd_row(
    row: dict | pd.Series, index: int, feature_cols: list[str]
) -> NSLKDDRow | None:
    category = UNSW_ATTACK_CATEGORY.get(_normalize_label(row.get(_ATTACK_CAT, "")))
    if category is None:
        # Unrecognized attack_cat (a stray repeated header, or a revision this
        # map doesn't know) is dropped and counted by the caller — never
        # force-fit onto a label the schema doesn't define.
        return None

    source_ip, dest_ip = _reconstruct_hosts(row, index, category)
    service = str(row.get("service", "-"))
    protocol = PROTOCOL_MAP.get(str(row.get("proto", "")).strip().casefold(), Protocol.OTHER)

    duration_s = _safe_float(row.get("dur", 0.0))
    duration_ms = max(int(duration_s * 1000.0), 0)
    total_bytes = max(
        int(_safe_float(row.get("sbytes", 0.0)) + _safe_float(row.get("dbytes", 0.0))), 0
    )

    features = {c: _safe_float(row[c]) for c in feature_cols if _is_numlike(row.get(c))}

    return NSLKDDRow(
        source_ip=source_ip,
        dest_ip=dest_ip,
        # The pre-split release has no port columns; the source port is
        # therefore genuinely absent (None, not a stand-in), while the
        # destination port comes from the service name where one is known.
        source_port=None,
        dest_port=SERVICE_PORT.get(service),
        protocol=protocol,
        total_bytes=total_bytes,
        duration_ms=duration_ms,
        raw_label=str(row.get(_ATTACK_CAT, "")).strip(),
        category=category,
        kronus_label=CATEGORY_TO_LABEL.get(category),
        difficulty=0,  # UNSW-NB15 has no NSL-KDD-style difficulty column
        features=features,
        origin=DataOrigin.REAL,
    )


def _read_one_csv(path: Path) -> pd.DataFrame:
    # utf-8-sig strips the BOM the published files begin with; low_memory=False
    # because several columns mix ints with the odd empty cell.
    return _clean_columns(pd.read_csv(path, low_memory=False, encoding="utf-8-sig"))


def _resolve_csv_paths(path: Path) -> list[Path]:
    if path.is_dir():
        csv_paths = [path / name for name in UNSW_NB15_FILES if (path / name).exists()]
        if not csv_paths:
            csv_paths = sorted(path.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(
                f"No UNSW-NB15 CSVs found in {path}/. Run "
                "`python scripts/download_unsw_nb15.py` (see docs/setup.md §external)."
            )
        # Training file first so a limit-based load gets the larger corpus.
        csv_paths.sort(key=lambda p: 0 if "training" in p.name else 1)
        return csv_paths
    if path.is_file():
        return [path]
    raise FileNotFoundError(
        f"{path} not found. Run `python scripts/download_unsw_nb15.py` first "
        "(see docs/setup.md §external)."
    )


def load_unsw_nb15(
    path: str | Path, limit: int | None = None, sample_per_file: int | None = None
) -> list[NSLKDDRow]:
    """Load UNSW-NB15 labelled flows into the shared NSLKDDRow shape.

    `path` may be a single CSV or a directory holding the pre-split files.
    Rows with an unmappable `attack_cat` are dropped (the caller can compare
    the returned length against the file's row count to see how many).
    """
    path = Path(path)
    csv_paths = _resolve_csv_paths(path)

    with observe("digital_twin", "load_unsw_nb15", path=str(path)):
        rows: list[NSLKDDRow] = []
        for csv_path in csv_paths:
            df = _read_one_csv(csv_path)
            if _ATTACK_CAT not in df.columns:
                raise ValueError(
                    f"{csv_path} has no 'attack_cat' column after cleaning; "
                    f"columns seen: {list(df.columns)[:8]}... — is this a "
                    "UNSW-NB15 pre-split CSV?"
                )
            if sample_per_file is not None and sample_per_file > 0 and len(df) > sample_per_file:
                df = df.iloc[:sample_per_file]
            feature_cols = [c for c in df.columns if c not in _IDENTITY_COLS]
            records = df.to_dict("records")
            for i, row in enumerate(records):
                converted = _row_to_nslkdd_row(row, i, feature_cols)
                if converted is not None:
                    rows.append(converted)
                    if limit is not None and len(rows) >= limit:
                        return rows
        return rows


def load_unsw_nb15_dataframe(path: str | Path) -> pd.DataFrame:
    """Raw combined dataframe (cleaned column names + a normalized `category`
    column) for harness code that wants vectorized pandas access rather than a
    list of dataclasses. Parallels twin/cicids2017.load_cicids2017_dataframe."""
    path = Path(path)
    csv_paths = _resolve_csv_paths(path)
    df = pd.concat([_read_one_csv(p) for p in csv_paths], ignore_index=True)
    df["category"] = df[_ATTACK_CAT].map(
        lambda label: UNSW_ATTACK_CATEGORY.get(_normalize_label(label))
    )
    return df
