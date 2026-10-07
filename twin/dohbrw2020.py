"""The CIRA-CIC-DoHBrw-2020 external-validation loader.

Dataset: CIRA-CIC-DoHBrw-2020 (Canadian Institute for Cybersecurity, UNB) —
the DoH / DoH-tunnel capture behind:

    Mohammadreza MontazeriShatoori, Logan Davidson, Gaya Dharmawansa,
    Arash Habibi Lashkari, "Detection of DoH Tunnels using Time-series
    Classification of Encrypted Traffic", 5th IEEE Cyber Science and
    Technology Congress (CyberSciTech), 2020.

Unlike every other loader in this package, this dataset ships with REAL IP
addresses, REAL ports and a REAL capture clock, so nothing is reconstructed:
`source_ip`/`dest_ip` are the addresses the flows actually carried. That makes
this the one external experiment here with genuine graph topology rather than
a topology inferred from per-row counters (contrast twin/unsw_nb15.py and
twin/nsl_kdd.py, both of which disclose reconstruction).

WHAT THE LABELS ACTUALLY ARE
----------------------------
The per-class CSVs this loader reads are named for the traffic they hold, and
the label *column* does not name the class: the benign file's trailing column
is `Label` with the single value "Benign", while each tunnel file's trailing
column is `DoH` with the single value "True" (it asserts "this flow is DoH",
which is true of every row in the file and therefore useless as a class).
The class therefore comes from the FILENAME, with the label column only as a
secondary hint. `resolve_class` is explicit about that order, and an
unrecognized name resolves to None so the file is dropped and visible rather
than coerced onto a guessed class. "Malicious-DoH.csv" — the mirror's union
of all three tunnels — deliberately resolves to None: it does not say which
tool produced a given flow, and this loader will not invent that.

KRONUS MAPPING
--------------
    Benign-DoH.csv   (benign DoH)          -> normal            -> Label.BENIGN
    DNSCat2-DoH.csv  (dnscat2 tunnel)      -> lateral_movement  -> Label.LATERAL_MOVEMENT
    dns2tcp-DoH.csv  (dns2tcp tunnel)      -> lateral_movement  -> Label.LATERAL_MOVEMENT
    iodine-DoH.csv   (iodine tunnel)       -> lateral_movement  -> Label.LATERAL_MOVEMENT

`lateral_movement` is the KRONUS label for tunnelled/exfiltrating traffic (see
libs/constants.py's fixed Label enum): a DNS tunnel is a host moving data out
through an established channel, not a volumetric flood. No row here maps to
Label.FLOOD, which is why scripts/run_dohbrw2020_experiment.py does not train
the Bouncer on this dataset — see that script's module docstring.

ORDERING
--------
Rows are returned sorted by the dataset's own `TimeStamp`, so a replay is in
true capture order. The absolute clock is NOT carried into NSLKDDRow (it has
no timestamp slot, and adding the capture epoch to `features` would leak the
label outright — the benign capture is December 2019 and the tunnel captures
are March 2020, so a timestamp column would separate the classes by itself).
Runners replay these rows on the same synthetic clock the other external
experiments use; only the *sequence* is the dataset's.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from libs.constants import DataOrigin, Label, Protocol
from twin.nsl_kdd import NSLKDDRow

# Filename (or label-value) token -> (KRONUS category, KRONUS label).
# Matched against the tokens of a filename first, then the label column:
# "Benign-DoH.csv" -> "benign"; "dns2tcp-DoH.csv" -> "dns2tcp".
DOHBRW_CLASSES: dict[str, tuple[str, Label]] = {
    "benign": ("normal", Label.BENIGN),
    # The original release also ships the non-DoH control traffic; it is
    # benign DNS/HTTPS dialled directly, so it maps the same way.
    "nondoh": ("normal", Label.BENIGN),
    "dnscat2": ("lateral_movement", Label.LATERAL_MOVEMENT),
    "dnscat": ("lateral_movement", Label.LATERAL_MOVEMENT),
    "dns2tcp": ("lateral_movement", Label.LATERAL_MOVEMENT),
    "iodine": ("lateral_movement", Label.LATERAL_MOVEMENT),
}

# The four per-class files this experiment reads, preferred in this order.
DOHBRW_FILES: list[str] = [
    "Benign-DoH.csv",
    "DNSCat2-DoH.csv",
    "dns2tcp-DoH.csv",
    "iodine-DoH.csv",
]

# Columns that name the class rather than measure the flow.
LABEL_COLUMNS: tuple[str, ...] = ("Label", "DoH")


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(text).casefold()) if t]


def resolve_class(name: str, label_value: str = "") -> tuple[str, Label] | None:
    """Map a file's name (then its label column) onto (category, KRONUS label).

    The filename is consulted first because it is the only thing in these
    files that actually names the class — see the module docstring. Returns
    None when nothing is recognized, which callers treat as "skip this file",
    never as "guess".
    """
    for source in (name, label_value):
        for token in _tokens(source):
            if token in DOHBRW_CLASSES:
                return DOHBRW_CLASSES[token]
    return None


# Well-known ports -> the transport the flow ran over. The dataset carries no
# protocol column, so this is derived, not read. DoH is HTTPS (TCP/443) and
# plain DNS is UDP/53; the iodine capture records some flows server->client,
# so BOTH endpoints are checked, destination first.
PORT_PROTOCOL: dict[int, Protocol] = {
    443: Protocol.TCP,
    8443: Protocol.TCP,
    80: Protocol.TCP,
    8080: Protocol.TCP,
    53: Protocol.UDP,
    5353: Protocol.UDP,
}


def _protocol_for(source_port: int | None, dest_port: int | None) -> Protocol:
    for port in (dest_port, source_port):
        if port in PORT_PROTOCOL:
            return PORT_PROTOCOL[port]
    return Protocol.OTHER


def _int_or_none(value) -> int | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and any surviving BOM from the header."""
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df


def _label_value(df: pd.DataFrame) -> str:
    for col in LABEL_COLUMNS:
        if col in df.columns and len(df):
            return str(df[col].iloc[0])
    return ""


def _read_one_csv(path: Path, sample_per_file: int | None) -> pd.DataFrame | None:
    """Read one per-class CSV, capped at `sample_per_file` rows.

    `nrows` streams rather than materializing the whole file, which matters
    here: dns2tcp-DoH.csv alone is ~102 MB and this is scoped for a laptop
    with modest RAM. The cap takes each file's leading rows, which is a
    deterministic prefix, not a random sample.
    """
    try:
        df = pd.read_csv(
            path, encoding="utf-8-sig", low_memory=False, nrows=sample_per_file
        )
    except Exception:
        return None
    return _clean_columns(df)


def _resolve_csv_paths(path: Path) -> list[Path]:
    """Accept a directory (or a single CSV) and return the CSVs to read.

    The four known per-class files come first, in a fixed order, so a `limit`
    always samples them before anything else. Other CSVs in the directory are
    still read, last: an unrecognized file must be *seen* and then dropped by
    `resolve_class`, not silently ignored by never being opened — that is what
    makes a mis-named file show up as a drop instead of as absent data.
    """
    if path.is_dir():
        known = [path / n for n in DOHBRW_FILES if (path / n).exists()]
        others = sorted(p for p in path.glob("*.csv") if p.name not in DOHBRW_FILES)
        found = known + others
    elif path.suffix == ".csv" and path.exists():
        found = [path]
    else:
        found = []
    if not found:
        raise FileNotFoundError(
            f"no DoHBrw2020 CSVs found at {path} — run scripts/download_dohbrw2020.py"
        )
    return found


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric column that is not a label column.

    SourceIP/DestinationIP/TimeStamp fall out on their own (they do not coerce
    to numbers), which keeps the rule simple and means no host identity or
    capture time can reach a model through `features`.
    """
    cols = []
    for col in df.columns:
        if col in LABEL_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _rows_from_frame(
    df: pd.DataFrame, category: str, kronus_label: Label, raw_label: str
) -> list[NSLKDDRow]:
    rows: list[NSLKDDRow] = []
    feature_cols = _numeric_feature_columns(df)

    for _, series in df.iterrows():
        source_ip = str(series.get("SourceIP", "")).strip()
        dest_ip = str(series.get("DestinationIP", "")).strip()
        if not source_ip or not dest_ip:
            continue

        source_port = _int_or_none(series.get("SourcePort"))
        dest_port = _int_or_none(series.get("DestinationPort"))

        sent = series.get("FlowBytesSent", 0)
        received = series.get("FlowBytesReceived", 0)
        sent = 0.0 if pd.isna(sent) else float(sent)
        received = 0.0 if pd.isna(received) else float(received)
        total_bytes = int(max(round(sent + received), 0))

        duration_s = series.get("Duration", 0)
        duration_s = 0.0 if pd.isna(duration_s) else float(duration_s)
        duration_ms = int(max(round(duration_s * 1000.0), 0))

        features: dict[str, float] = {}
        for col in feature_cols:
            value = series.get(col)
            if pd.isna(value):
                continue
            features[col] = float(value)

        rows.append(
            NSLKDDRow(
                source_ip=source_ip,
                dest_ip=dest_ip,
                source_port=source_port,
                dest_port=dest_port,
                protocol=_protocol_for(source_port, dest_port),
                total_bytes=total_bytes,
                duration_ms=duration_ms,
                raw_label=raw_label,
                category=category,
                kronus_label=kronus_label,
                difficulty=0,
                features=features,
                origin=DataOrigin.REAL,
            )
        )
    return rows


def _sort_by_capture_time(df: pd.DataFrame) -> pd.DataFrame:
    """Order rows by the dataset's own TimeStamp (unparseable rows last)."""
    if "TimeStamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["TimeStamp"], errors="coerce")
    return df.assign(_ts=ts).sort_values("_ts", kind="stable", na_position="last").drop(
        columns="_ts"
    )


def load_dohbrw2020(
    path: str | Path,
    limit: int | None = None,
    sample_per_file: int | None = None,
) -> list[NSLKDDRow]:
    """Load DoHBrw2020 CSVs into the shared NSLKDDRow shape.

    `sample_per_file` caps rows read per CSV (streamed, so memory stays
    bounded); `limit` caps the total returned. Files whose class cannot be
    resolved are skipped — see `resolve_class`.
    """
    paths = _resolve_csv_paths(Path(path))
    rows: list[NSLKDDRow] = []
    for csv_path in paths:
        df = _read_one_csv(csv_path, sample_per_file)
        if df is None or df.empty:
            continue
        resolved = resolve_class(csv_path.name, _label_value(df))
        if resolved is None:
            continue
        category, kronus_label = resolved
        df = _sort_by_capture_time(df)
        rows.extend(_rows_from_frame(df, category, kronus_label, csv_path.stem))
        if limit is not None and limit > 0 and len(rows) >= limit:
            return rows[:limit]
    return rows


def load_dohbrw2020_dataframe(
    path: str | Path, sample_per_file: int | None = None
) -> pd.DataFrame:
    """Every row as a DataFrame, with a `category` column (NaN when unmapped).

    Unlike `load_dohbrw2020`, rows from an unresolvable file are KEPT and
    tagged NaN — so "this file contributed nothing" is visible in the data
    rather than only inferable from a missing file.
    """
    paths = _resolve_csv_paths(Path(path))
    frames = []
    for csv_path in paths:
        df = _read_one_csv(csv_path, sample_per_file)
        if df is None or df.empty:
            continue
        resolved = resolve_class(csv_path.name, _label_value(df))
        df = df.copy()
        df["category"] = resolved[0] if resolved else None
        frames.append(_sort_by_capture_time(df))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(rows: list[NSLKDDRow]) -> dict[str, int]:
    """Category -> count, for reporting what actually loaded."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.category] = counts.get(row.category, 0) + 1
    return counts
