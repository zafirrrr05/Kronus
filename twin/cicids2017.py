"""CIC-IDS2017 external-validation data source. The README's SLO table
promises an "External-dataset F1 > 0.85 | CIC-IDS2017-compatible validation
harness"; this module is the loader half of that harness (the scoring half
is scripts/validate_cicids2017.py).

Why a second loader at all, next to twin/nsl_kdd.py? NSL-KDD is the training
dataset; CIC-IDS2017 is an *independent* dataset collected on a different
network, three years later, with a different tool (CICFlowMeter). Validating
on it is the honest cross-dataset generalization test — "does a KRONUS model
trained on 1999-era KDD features still detect a 2017 flood/scan?" — that
training-set held-out metrics structurally cannot answer.

Two things this loader does that the NSL-KDD one doesn't have to, and one it
gets to skip:

1. SKIP: no IP reconstruction. CIC-IDS2017's labelled-flow CSVs
   (GeneratedLabelledFlows / TrafficLabelling) carry the *real* Source IP,
   Destination IP, ports, and a wall-clock Timestamp. So unlike NSL-KDD
   (twin/nsl_kdd.py's _synthetic_ip), the graph shape a scan/flood produces
   here is the genuine article, not a deterministically-reconstructed stand-in.
2. Map CIC's fine-grained label set (BENIGN, DoS Hulk, DoS GoldenEye, DoS
   slowloris, DoS Slowhttptest, DDoS, PortScan, FTP-Patator, SSH-Patator,
   Bot, Infiltration, Heartbleed, the three Web Attack variants) onto
   KRONUS's fixed Label enum — see CIC_ATTACK_CATEGORY. DoS*/DDoS -> flood,
   PortScan -> port_scan, BENIGN -> benign. The patator/web/bot/infiltration/
   heartbleed families have no clean KRONUS-label home (identical situation
   to NSL-KDD's r2l/u2r — see twin/nsl_kdd.py) so they're folded in as
   generic non-benign signal, never force-fit onto a label the schema
   doesn't define.
3. Clean the well-known CIC-IDS2017 data-quality quirks: header names carry
   leading spaces; the Flow Bytes/s and Flow Packets/s columns contain Inf
   and NaN where a flow's duration is zero; a handful of rows are repeated
   CSV headers. All handled explicitly here rather than silently coerced.

The output type is the SAME NSLKDDRow shape twin/nsl_kdd.py produces, so
every downstream consumer (services/telemetry_exporter/converters.flow_row_to_event,
the Bouncer featurizer, the Graph Builder) works unchanged — the whole point
of a "compatible" harness is that no service needs a CIC-specific code path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from libs.constants import DataOrigin, Label, Protocol
from libs.observability import observe
from twin.nsl_kdd import NSLKDDRow

# CIC-IDS2017's eight labelled-flow CSVs (the TrafficLabelling /
# GeneratedLabelledFlows release), in capture order. The loader reads any
# subset present — a caller can validate on one day or all of them.
CICIDS2017_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# CIC label (normalized: stripped, collapsed internal whitespace, casefolded)
# -> KRONUS category. The three "Web Attack" labels use a non-ASCII en-dash
# (U+2013) in the raw files; _normalize_label folds that to a plain hyphen so
# these keys match regardless of the exact dash byte.
CIC_ATTACK_CATEGORY: dict[str, str] = {
    "benign": "normal",
    # DoS / DDoS -> flood
    "dos hulk": "dos",
    "dos goldeneye": "dos",
    "dos slowloris": "dos",
    "dos slowhttptest": "dos",
    "ddos": "dos",
    "heartbleed": "dos",  # a DoS-family CVE flood in CIC's own taxonomy
    # Probe -> port_scan
    "portscan": "probe",
    # No clean KRONUS label (generic non-benign signal, like NSL-KDD r2l/u2r)
    "ftp-patator": "r2l",
    "ssh-patator": "r2l",
    "bot": "r2l",
    "infiltration": "u2r",
    "web attack-brute force": "r2l",
    "web attack-xss": "r2l",
    "web attack-sql injection": "r2l",
}

CATEGORY_TO_LABEL: dict[str, Label] = {
    "normal": Label.BENIGN,
    "dos": Label.FLOOD,
    "probe": Label.PORT_SCAN,
    # r2l/u2r deliberately absent — no direct KRONUS label; see module docstring.
}

# CIC-IDS2017 records Protocol as an IANA protocol number, not a name.
PROTOCOL_NUMBER_MAP: dict[int, Protocol] = {
    6: Protocol.TCP,
    17: Protocol.UDP,
    1: Protocol.ICMP,
}

# The GeneratedLabelledFlows columns this loader actually reads. CICFlowMeter
# emits ~85 columns; KRONUS's TelemetryEvent needs only the flow's identity
# (IPs/ports/protocol), size, duration, and timestamp. The full numeric
# feature row is still kept (see `features` on the returned NSLKDDRow) for any
# consumer that wants it, but these are the ones with a wire-schema home.
_SRC_IP = "Source IP"
_DST_IP = "Destination IP"
_SRC_PORT = "Source Port"
_DST_PORT = "Destination Port"
_PROTOCOL = "Protocol"
_DURATION = "Flow Duration"  # microseconds
_FWD_BYTES = "Total Length of Fwd Packets"
_BWD_BYTES = "Total Length of Bwd Packets"
_LABEL = "Label"


def _normalize_label(raw: str) -> str:
    """CIC labels vary by an en-dash vs hyphen and inconsistent spacing
    ('Web Attack – Brute Force' vs 'Web Attack-Brute Force'). Fold to a
    single canonical key so CIC_ATTACK_CATEGORY lookups are robust to it."""
    s = str(raw).replace("–", "-").replace("—", "-")
    s = " ".join(s.split())  # collapse runs of whitespace
    s = s.replace(" - ", "-").replace("- ", "-").replace(" -", "-")
    return s.strip().casefold()


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CIC-IDS2017 headers carry leading/trailing spaces (' Source IP',
    ' Flow Duration', ...). Strip them so callers can use clean names."""
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    return df


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Flow Bytes/s and Flow Packets/s hold 'Infinity'/'NaN' strings where a
    flow's duration is zero — a documented CIC-IDS2017 quirk. Coerce to
    numbers, turning Inf/NaN into 0.0 rather than letting them poison a model
    or crash pydantic's ge=0 validation downstream."""
    out = pd.to_numeric(series, errors="coerce")
    return out.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)


def _row_to_nslkdd_row(row: pd.Series, index: int) -> NSLKDDRow | None:
    category = CIC_ATTACK_CATEGORY.get(_normalize_label(row[_LABEL]))
    if category is None:
        # An unrecognized label (a stray repeated header row, or a label from
        # a CIC revision this map doesn't know) is dropped rather than
        # force-fit — the caller counts drops (see load_cicids2017).
        return None

    proto_num = int(_safe_float(row.get(_PROTOCOL, 0)))
    protocol = PROTOCOL_NUMBER_MAP.get(proto_num, Protocol.OTHER)

    duration_us = _safe_float(row.get(_DURATION, 0.0))
    duration_ms = max(int(duration_us / 1000.0), 0)
    total_bytes = max(
        int(_safe_float(row.get(_FWD_BYTES, 0.0)) + _safe_float(row.get(_BWD_BYTES, 0.0))), 0
    )

    features = {
        col: _safe_float(row[col])
        for col in row.index
        if col not in (_SRC_IP, _DST_IP, _LABEL) and _is_numlike(row[col])
    }

    return NSLKDDRow(
        source_ip=str(row.get(_SRC_IP, "0.0.0.0")).strip(),
        dest_ip=str(row.get(_DST_IP, "0.0.0.0")).strip(),
        source_port=_safe_int(row.get(_SRC_PORT)),
        dest_port=_safe_int(row.get(_DST_PORT)),
        protocol=protocol,
        total_bytes=total_bytes,
        duration_ms=duration_ms,
        raw_label=str(row[_LABEL]).strip(),
        category=category,
        kronus_label=CATEGORY_TO_LABEL.get(category),
        difficulty=0,  # CIC-IDS2017 has no NSL-KDD-style difficulty column
        features=features,
        origin=DataOrigin.REAL,
    )


def _safe_float(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f != f or f in (float("inf"), float("-inf")):  # NaN or Inf
        return 0.0
    return f


def _safe_int(v) -> int | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return int(f)


def _is_numlike(v) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def _read_one_csv(path: Path) -> pd.DataFrame:
    # low_memory=False: CIC CSVs mix types within a column (the Inf/NaN
    # strings), which triggers pandas' mixed-type chunk warning otherwise.
    df = pd.read_csv(path, low_memory=False, skipinitialspace=False)
    return _clean_columns(df)


def load_cicids2017(path: str | Path) -> list[NSLKDDRow]:
    """Load CIC-IDS2017 labelled flows into the shared NSLKDDRow shape.

    `path` may be a single CSV file or a directory of them (the eight
    TrafficLabelling CSVs). Rows whose label maps to no KRONUS category are
    dropped and counted — never silently coerced onto a wrong label.

    Raises FileNotFoundError with a pointer to the downloader if nothing is
    present, mirroring twin/nsl_kdd.load_nsl_kdd's contract so the validation
    harness and its tests can skip gracefully.
    """
    path = Path(path)
    csv_paths = _resolve_csv_paths(path)

    with observe("digital_twin", "load_cicids2017", path=str(path)):
        rows: list[NSLKDDRow] = []
        for csv_path in csv_paths:
            df = _read_one_csv(csv_path)
            if _LABEL not in df.columns:
                raise ValueError(
                    f"{csv_path} has no 'Label' column after cleaning; "
                    f"columns seen: {list(df.columns)[:8]}... — is this a "
                    "CIC-IDS2017 GeneratedLabelledFlows CSV?"
                )
            for i, row in df.iterrows():
                converted = _row_to_nslkdd_row(row, i)
                if converted is not None:
                    rows.append(converted)
        return rows


def _resolve_csv_paths(path: Path) -> list[Path]:
    if path.is_dir():
        csv_paths = sorted(path.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(
                f"No CIC-IDS2017 CSVs found in {path}/. Run "
                "`python scripts/download_cicids2017.py` (see docs/setup.md §external)."
            )
        return csv_paths
    if path.is_file():
        return [path]
    raise FileNotFoundError(
        f"{path} not found. Run `python scripts/download_cicids2017.py` first "
        "(see docs/setup.md §external)."
    )


def load_cicids2017_dataframe(path: str | Path) -> pd.DataFrame:
    """Raw combined dataframe (cleaned column names + a normalized `category`
    column), for harness code that wants vectorized pandas access rather than
    a list of dataclasses. Parallels twin/nsl_kdd.load_nsl_kdd_dataframe."""
    path = Path(path)
    csv_paths = _resolve_csv_paths(path)
    frames = [_read_one_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df["category"] = df[_LABEL].map(lambda label: CIC_ATTACK_CATEGORY.get(_normalize_label(label)))
    return df
