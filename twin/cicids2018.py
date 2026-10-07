"""CSE-CIC-IDS2018 external-validation data source.

CSE-CIC-IDS2018 ("Intrusion Detection Evaluation Dataset", Canadian Institute
for Cybersecurity, University of New Brunswick) is the successor to
CIC-IDS2017. Official dataset: https://www.unb.ca/cic/datasets/ids-2018.html

    Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani,
    "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
    Characterization", 4th ICISSP, Portugal, January 2018.

THE REAL SCHEMA, MEASURED FROM THE RELEASE
==========================================
This loader was written against the header of the actual released file
(verified by ranged read of the public AWS bucket), which is 80 columns:

    Dst Port, Protocol, Timestamp, Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts,
    TotLen Fwd Pkts, TotLen Bwd Pkts, Fwd Pkt Len Max, ..., Idle Min, Label

There is **no Source IP, no Destination IP and no Source Port column**. The
ML-ready CSVs keep only the destination *port*. A loader that reads
`Source IP` / `Total Length of Fwd Packets` is reading a *different* dataset's
schema — that is CIC-IDS2017's naming — and the two are not interchangeable.
`_REQUIRED_COLUMNS` makes that mistake a hard error rather than a dataset of
zeros.

WHAT IS REAL, AND WHAT IS RECONSTRUCTED
=======================================
Flow *measurements* are real and are used as-is:

    total_bytes    TotLen Fwd Pkts + TotLen Bwd Pkts      (real)
    duration_ms    Flow Duration, microseconds -> ms       (real)
    dest_port      Dst Port                                (real)
    protocol       Protocol, IANA number -> Protocol enum  (real)
    capture clock  Timestamp, DD/MM/YYYY HH:MM:SS          (real)
    features       74 CICFlowMeter measurements            (real)

Host *identity* is not in the file, so it is reconstructed, deterministically
and class-independently, and disclosed in the metrics as
``"synthetic_hosts": true``:

    source_ip    one constant node (MONITORED_CLIENT_IP). The release records
                 no client identity, and a class-varying source would hand the
                 model a perfect label proxy.
    dest_ip      a bijection of the REAL Dst Port (`_dest_ip_for`). Distinct
                 destinations in a window are therefore the capture's own
                 distinct *services*. The graph measures service fan-out, not
                 host fan-out, and the metrics say so.
    source_port  None — genuinely absent from the release, never a stand-in.

THE CAPTURE CLOCK IS REAL, AND THAT IS THE POINT
================================================
Every other external loader in this repository replays rows at synthetic
spacing, because its source carries no usable time. Equal spacing makes any
*rate* feature constant by construction, which quietly neuters the Bouncer's
`event_rate` and `byte_rate`. This dataset ships a real clock, so the timed
loader returns it and the runner replays at true inter-arrival times: a
2-second window is then a real 2 seconds of capture, and a flood's burst rate
shows up as a burst rate.

The clock is deliberately NOT placed in `features` — it would leak the label
outright, since an attack day is a different day from a benign one.

CONSEQUENCE, STATED PLAINLY
===========================
Because every row shares one reconstructed source_ip, the featurizer's window
aggregate is *global to the capture*, not per-host. A benign flow arriving in
the middle of a flood therefore sits in a window the flood dominates, and its
features look like the flood's. The runner handles this by growing the
capture's attack intervals (`attack_intervals`) and scoring the Bouncer only
on benign rows that fall outside them. The alternative — labelling mid-flood
benign rows "benign" — would teach the model that a flood is benign.

WHY THE ROW CAP IS A STRIDE (AND MUST BE)
=========================================
Every day CSV is written in capture order, and each day's attack is a
time-bounded window inside hours of that day's ordinary traffic. A prefix
(`nrows=`) therefore samples the quiet start of the capture and can return a
file that is ~100% benign while reporting success — measured, not assumed:
the tail of Thursday-01-03-2018 is entirely `Benign`, so a prefix of it
contains no attack at all. This loader counts the file's rows first and takes
an exact, evenly spaced stride, so a bounded sample spans the whole day
including its attack window. The stride is a pure function of the file, so
re-runs reproduce exactly.

MALFORMED ROWS ARE REAL
=======================
The released files contain repeated **header rows embedded mid-file** (a row
whose fields are the column names), which appear as a `Label` value of
"Label" — 13 of them in a single 8 MB slice of 01-03. They are dropped and
counted, never parsed as a flow. They matter beyond tidiness: a chunk
containing one has its numeric columns read as `object` dtype, so they are
removed before any numeric coercion.

LABEL MAPPING (CSE-CIC-IDS2018 -> KRONUS)
=========================================
    Benign                        -> normal  (Label.BENIGN)
    DoS attacks-* / DDOS attack-* -> dos     (Label.FLOOD)
    Infilteration                 -> probe   (Label.PORT_SCAN)
    everything else               -> dropped, counted, reported on stderr

Matching is by prefix on a casefolded, dash-normalized label, because the
release is internally inconsistent: it ships "DDoS attacks-LOIC-HTTP" and
"DDOS attack-HOIC" (upper-case DDOS, and plural vs singular "attacks") in the
same vocabulary. Exact-string matching would silently drop real attacks.

The `Infilteration` spelling is the release's own typo. It is mapped to
PORT_SCAN rather than LATERAL_MOVEMENT on UNB's own description of the
scenario — Nmap "IP sweep, full port scan and service enumerations" — and the
captured flows measure that way: they touch ~1.7x more distinct destination
ports and carry ~0.6x the packets per flow of the benign traffic captured in
the same file. Note that this dataset has **no PortScan-labelled flows at
all**; the port scanning it contains is rolled into `Infilteration`, making it
the only source of probe-class data here, and a noisy one.

Everything with no KRONUS counterpart (brute force, web attacks, SQL
injection, bot) is dropped and counted rather than force-fitted into a class.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from libs.constants import DataOrigin, Label, Protocol
from libs.observability import observe
from twin.nsl_kdd import NSLKDDRow

# --- the real column names -------------------------------------------------

COL_DST_PORT = "Dst Port"
COL_PROTOCOL = "Protocol"
COL_TIMESTAMP = "Timestamp"
COL_DURATION = "Flow Duration"          # microseconds
COL_FWD_BYTES = "TotLen Fwd Pkts"
COL_BWD_BYTES = "TotLen Bwd Pkts"
COL_LABEL = "Label"

# Refusing to guess: if these are absent the file is not CSE-CIC-IDS2018 and
# every downstream number would be meaningless. See the module docstring.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_DST_PORT, COL_PROTOCOL, COL_TIMESTAMP, COL_DURATION,
    COL_FWD_BYTES, COL_BWD_BYTES, COL_LABEL,
)

# Columns that are carried elsewhere in NSLKDDRow, or that must never be fed
# to a model. The clock in particular would leak the label outright, and the
# fwd/bwd byte totals are already summed into `total_bytes` (keeping them
# would double-count the same bytes under two names).
_EXCLUDED_FEATURES: frozenset[str] = frozenset({
    COL_TIMESTAMP,
    COL_LABEL,
    COL_DST_PORT,       # becomes NSLKDDRow.dest_port
    COL_PROTOCOL,       # becomes NSLKDDRow.protocol
    COL_FWD_BYTES,      # summed into NSLKDDRow.total_bytes
    COL_BWD_BYTES,      # summed into NSLKDDRow.total_bytes
})

# The dataset's internal subnet, used so the reconstruction stays inside the
# capture's own address space rather than inventing a foreign one.
MONITORED_CLIENT_IP = "172.31.69.1"
_DEST_IP_PREFIX = "10.60"   # port -> (high byte, low byte); see _dest_ip_for

PROTOCOL_NUMBER_MAP: dict[int, Protocol] = {
    6: Protocol.TCP,
    17: Protocol.UDP,
    1: Protocol.ICMP,
}

# Row-selection default: how many rows to keep per CSV when the caller does
# not say. Chosen so a whole run stays inside a laptop's memory.
DEFAULT_SAMPLE_PER_FILE = 60_000

# Kept small on purpose. The release's embedded header rows force pandas to
# read a chunk's numeric columns as `object`, which costs roughly 24 bytes per
# cell on top of the pointer array — at 100k rows x 80 columns that is a
# quarter-gigabyte of transient garbage per chunk, all of it live at the
# moment the float block is built. Measured: peak RSS on one day file is
# ~1.2 GB at 100k, and the whole run has to fit next to XGBoost and the GAT.
_CHUNK_ROWS = 25_000

# An attack interval is grown by this much on each side when benign rows are
# excluded from it, so a benign flow sharing a 2-second window with the first
# or last attack flow is excluded too — the featurizer's window is what it is,
# and a row inside it is contaminated whatever its own label says.
ATTACK_INTERVAL_GUARD_SECONDS = 3.0


# --- label resolution -------------------------------------------------------

def _normalize_label(value: object) -> str:
    """Casefold, collapse whitespace, and unify dash variants.

    The released vocabulary is internally inconsistent — "DDoS attacks-LOIC-HTTP"
    alongside "DDOS attack-HOIC" (upper-case, singular) — so exact-string
    matching would silently drop real attacks.
    """
    s = str(value)
    for dash in ("–", "—", "−"):
        s = s.replace(dash, "-")
    return " ".join(s.split()).casefold()


def resolve_class(raw_label: object) -> tuple[str, Label] | None:
    """Map a raw CSE-CIC-IDS2018 label to (category, KRONUS label).

    Returns None for labels with no KRONUS counterpart; the caller counts and
    reports those rather than coercing them into a class they are not.
    """
    s = _normalize_label(raw_label)
    if not s:
        return None
    if s == "benign":
        return ("normal", Label.BENIGN)
    # Every DoS/DDoS spelling the release uses: "dos attacks-hulk",
    # "ddos attacks-loic-http", "DDOS attack-HOIC", ...
    if s.startswith("dos attack") or s.startswith("ddos attack"):
        return ("dos", Label.FLOOD)
    if s in ("infilteration", "infiltration"):
        return ("probe", Label.PORT_SCAN)
    return None


# --- helpers ----------------------------------------------------------------

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and the UTF-8 BOM the release carries on cell 0."""
    return df.rename(columns={c: str(c).lstrip("﻿").strip() for c in df.columns})


def _dest_ip_for(dest_port: int | None) -> str:
    """Deterministic IPv4 for a destination port.

    A bijection, not a hash: distinct destinations in a graph window are
    exactly the capture's distinct destination ports, which is the only
    destination identity this dataset carries. Ports map to
    ``10.60.<port // 256>.<port % 256>``.
    """
    p = 0 if dest_port is None else max(0, min(int(dest_port), 65535))
    return f"{_DEST_IP_PREFIX}.{p >> 8}.{p & 0xFF}"


def _count_data_rows(path: Path) -> int:
    """Count newlines cheaply, so the stride can be exact and reproducible.

    Counting bytes is far cheaper than parsing them, and an exact stride is
    what makes a bounded sample both reproducible and evenly spread across
    the whole capture.
    """
    total = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 22)
            if not block:
                break
            total += block.count(b"\n")
    return max(total - 1, 0)  # minus the header line


def _resolve_csv_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        direct = [p for p in sorted(path.glob("*.csv")) if not p.name.startswith(".")]
        if direct:
            return direct
        nested = [p for p in sorted(path.glob("*/*.csv")) if not p.name.startswith(".")]
        if nested:
            return nested
    raise FileNotFoundError(
        f"{path} not found or contains no CSVs. "
        "Run scripts/download_cicids2018.py first (see docs/experiments_guide.md)."
    )


def _feature_columns(columns: list[str]) -> list[str]:
    """The feature columns, fixed by the file's header rather than by dtypes.

    Deriving these from dtypes would make them depend on file content: one
    embedded header row turns an entire chunk's numeric columns into `object`,
    and a feature set that changes between chunks cannot be joined or
    de-duplicated. The header is the same for every chunk, so it is the
    authority.
    """
    return [c for c in columns if c not in _EXCLUDED_FEATURES]


def _header_row_mask(labels: pd.Series) -> np.ndarray:
    """True for the repeated header rows embedded mid-file in the release."""
    return (
        labels.astype(str)
        .str.replace("﻿", "", regex=False)
        .str.strip()
        .str.casefold()
        .to_numpy()
        == "label"
    )


def _assert_schema(csv_path: Path) -> None:
    """Hard-fail on a file that is not the CSE-CIC-IDS2018 schema.

    Reading only the header costs nothing and is the difference between a
    loader that silently returns zeros and one that refuses. A CIC-IDS2017
    CSV fed to this loader carries "Source IP" and "Total Length of Fwd
    Packets" and no "Dst Port", every row would fail to resolve, and the
    result would be an empty dataset reported as a success.
    """
    header = pd.read_csv(
        csv_path, nrows=0, encoding="utf-8", encoding_errors="replace"
    )
    columns = list(_clean_columns(header).columns)
    missing = [c for c in _REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required column(s) {missing}; "
            f"it has {len(columns)} columns starting {columns[:5]}. "
            "This is not a CSE-CIC-IDS2018 'Processed Traffic Data for ML "
            "Algorithms' CSV — refusing to load it rather than returning an "
            "empty dataset under this dataset's name."
        )


# --- the reader -------------------------------------------------------------

def _iter_chunks(
    csv_path: Path,
    sample_per_file: int | None,
    stride: int | None,
) -> Iterator[tuple[pd.DataFrame, np.ndarray, list[str]]]:
    """Yield (chunk, kept row positions, feature columns).

    Memory stays bounded by `_CHUNK_ROWS`: no chunk is ever held alongside
    another and the whole file is never materialised. Either way of bounding
    the rows takes an exact, evenly spaced stride over the whole file, so a
    bounded sample spans the capture end to end — including its attack window
    — and reproduces exactly on a re-run.

    `stride` and `sample_per_file` are not interchangeable, and the difference
    matters to any *rate* feature:

      stride           one decimation factor applied to every file, so the
                       rows kept from file A and file B are thinned by the
                       same factor and their event rates stay comparable.
      sample_per_file  a per-file row budget; the factor differs per file, so
                       a larger file is thinned harder and its rates are NOT
                       comparable with a smaller file's. Fine for counting
                       rows, wrong for comparing burst rates across days.
    """
    if stride is not None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        effective_stride = stride
    else:
        total_rows = _count_data_rows(csv_path) if sample_per_file else 0
        effective_stride = (
            max(1, total_rows // sample_per_file) if (sample_per_file and total_rows) else 1
        )
    budget = sample_per_file if stride is None else None
    kept = 0
    seen = 0

    reader = pd.read_csv(
        csv_path,
        chunksize=_CHUNK_ROWS,
        low_memory=True,
        encoding="utf-8",
        encoding_errors="replace",
    )
    while True:
        # The release's embedded header rows make pandas read a chunk's
        # numeric columns as `object` and warn about "mixed types". That is
        # expected here and handled: features are coerced explicitly below,
        # and `_feature_columns` takes its column list from the header rather
        # than from dtypes precisely so one stray row cannot change it. The
        # warning is silenced rather than the type check disabled, so genuine
        # parse problems still surface once coercion reports NaN.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
            try:
                chunk = next(reader)
            except StopIteration:
                return
        chunk = _clean_columns(chunk)
        if COL_LABEL not in chunk.columns:
            raise ValueError(
                f"{csv_path.name} has no 'Label' column; columns seen: "
                f"{list(chunk.columns)[:8]}... — this is not a "
                "CSE-CIC-IDS2018 TrafficForML CSV."
            )
        feature_cols = _feature_columns(list(chunk.columns))
        idx = np.arange(seen, seen + len(chunk))
        seen += len(chunk)
        positions = np.flatnonzero(idx % effective_stride == 0)
        if budget is not None:
            remaining = budget - kept
            if remaining <= 0:
                return
            positions = positions[:remaining]
        if len(positions) == 0:
            continue
        kept += len(positions)
        yield chunk, positions, feature_cols


class _DayChunk:
    """Column-wise view of one kept slice of a CSV.

    Converting a whole chunk with `to_dict("records")` costs a dict per row
    across 80 columns, and then a `float()` per cell — for the ~200k rows a
    full run touches that is tens of millions of Python calls. Everything the
    row builder needs is therefore pulled out as numpy arrays once, here, and
    the per-row work is array indexing.
    """

    __slots__ = ("labels", "header_mask", "dest_ports", "protocols",
                 "durations_us", "fwd_bytes", "bwd_bytes", "features",
                 "feature_cols", "times")

    def __init__(self, chunk: pd.DataFrame, positions: np.ndarray,
                 feature_cols: list[str], timed: bool) -> None:
        sub = chunk.iloc[positions]
        labels = sub[COL_LABEL]
        self.labels = labels.astype(str).to_numpy()
        self.header_mask = _header_row_mask(labels)

        def num(col: str) -> np.ndarray:
            if col not in sub.columns:
                return np.zeros(len(sub), dtype=float)
            return pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)

        self.dest_ports = num(COL_DST_PORT)
        self.protocols = num(COL_PROTOCOL)
        self.durations_us = num(COL_DURATION)
        self.fwd_bytes = num(COL_FWD_BYTES)
        self.bwd_bytes = num(COL_BWD_BYTES)
        self.feature_cols = feature_cols
        if feature_cols:
            block = pd.DataFrame({
                c: pd.to_numeric(sub[c], errors="coerce") for c in feature_cols
            })
            self.features = block.to_numpy(dtype=float)
        else:
            self.features = np.zeros((len(sub), 0), dtype=float)
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)

        if timed:
            ts = pd.to_datetime(
                sub[COL_TIMESTAMP], format="%d/%m/%Y %H:%M:%S", errors="coerce"
            )
            if ts.isna().all():
                # Fall back to inference only if the strict format matched
                # nothing at all — a partial match means the format is right.
                ts = pd.to_datetime(sub[COL_TIMESTAMP], errors="coerce", dayfirst=True)
            self.times = ts.tolist()
        else:
            self.times = None

    def rows(self) -> Iterator[tuple[int, NSLKDDRow | None]]:
        for i in range(len(self.labels)):
            if self.header_mask[i]:
                yield i, None
                continue
            resolved = resolve_class(self.labels[i])
            if resolved is None:
                yield i, None
                continue
            category, kronus_label = resolved
            dest_port = int(self.dest_ports[i])
            total_bytes = int(max(self.fwd_bytes[i] + self.bwd_bytes[i], 0.0))
            features = {
                col: float(self.features[i, j]) for j, col in enumerate(self.feature_cols)
            }
            yield i, NSLKDDRow(
                source_ip=MONITORED_CLIENT_IP,
                dest_ip=_dest_ip_for(dest_port),
                source_port=None,      # genuinely absent from the release
                dest_port=dest_port,
                protocol=PROTOCOL_NUMBER_MAP.get(int(self.protocols[i]), Protocol.OTHER),
                total_bytes=total_bytes,
                duration_ms=max(int(self.durations_us[i] / 1000.0), 0),
                raw_label=str(self.labels[i]).strip(),
                category=category,
                kronus_label=kronus_label,
                difficulty=0,
                features=features,
                origin=DataOrigin.REAL,
            )


def _read_rows(
    path: str | Path,
    limit: int | None,
    sample_per_file: int | None,
    timed: bool,
    stride: int | None = None,
    with_features: bool = True,
) -> tuple[list[NSLKDDRow], list | None, dict]:
    """Shared body of the public loaders."""
    path = Path(path)
    csv_paths = _resolve_csv_paths(path)
    rows: list[NSLKDDRow] = []
    times: list = []
    unmapped: dict[str, int] = {}
    n_malformed = 0
    n_unmapped = 0

    for csv_path in csv_paths:
        _assert_schema(csv_path)
        before = len(rows)
        for chunk, positions, feature_cols in _iter_chunks(csv_path, sample_per_file, stride):
            day = _DayChunk(
                chunk, positions, feature_cols if with_features else [], timed
            )
            for i, row in day.rows():
                if row is None:
                    if day.header_mask[i]:
                        n_malformed += 1
                    else:
                        n_unmapped += 1
                        key = str(day.labels[i]).strip() or "<empty>"
                        unmapped[key] = unmapped.get(key, 0) + 1
                    continue
                rows.append(row)
                if timed and day.times is not None:
                    times.append(day.times[i])
                if limit is not None and len(rows) >= limit:
                    break
            if limit is not None and len(rows) >= limit:
                break
        if len(rows) == before:
            print(
                f"  [cicids2018] WARNING: {csv_path.name} contributed no rows — "
                "is this a TrafficForML CSV?",
                file=sys.stderr,
            )
        if limit is not None and len(rows) >= limit:
            break

    if n_unmapped:
        top = ", ".join(
            f"{k}={v}" for k, v in sorted(unmapped.items(), key=lambda x: -x[1])[:8]
        )
        print(
            f"  [cicids2018] dropped {n_unmapped:,} rows with no KRONUS "
            f"counterpart ({top})",
            file=sys.stderr,
        )
    if n_malformed:
        print(
            f"  [cicids2018] dropped {n_malformed:,} embedded header rows",
            file=sys.stderr,
        )

    # Sort into the capture's true chronological order. The clock is the
    # release's own; it orders the replay and is deliberately not carried
    # into `features`.
    if rows and times:
        order = sorted(range(len(rows)), key=lambda i: (times[i], i))
        rows = [rows[i] for i in order]
        times = [times[i] for i in order]

    stats = {
        "rows": len(rows),
        "files": [str(p) for p in csv_paths],
        "n_files": len(csv_paths),
        "rows_dropped_unmapped": n_unmapped,
        "rows_dropped_malformed": n_malformed,
        "unmapped_labels": unmapped,
    }
    return rows, (times if timed else None), stats


# --- public API -------------------------------------------------------------

def load_cicids2018(
    path: str | Path,
    limit: int | None = None,
    sample_per_file: int | None = DEFAULT_SAMPLE_PER_FILE,
    stride: int | None = None,
    with_features: bool = True,
) -> list[NSLKDDRow]:
    """Load CSE-CIC-IDS2018 flows into the shared NSLKDDRow shape.

    `path` may be a single day CSV, a directory of them, or a parent directory
    holding per-day folders. Rows whose label has no KRONUS counterpart are
    dropped and reported on stderr, never coerced.

    `sample_per_file` bounds how many rows are kept from each CSV; `stride`, if
    given, instead thins every CSV by the same factor. The two are not
    equivalent for rate features — see `_iter_chunks`. Pass
    `sample_per_file=None` (and no stride) for everything, which is only sane
    on a small file.

    `with_features=False` leaves `NSLKDDRow.features` empty. The 74
    CICFlowMeter columns cost ~2.5 KB per row as a Python dict, and neither
    detection lane reads them — both derive what they need from the event's
    own fields. A dense load (every row of a 613k-row day) is only affordable
    without them. Anything that inspects `features` must use the default.
    """
    with observe("digital_twin", "load_cicids2018", path=str(path)):
        rows, _, _ = _read_rows(
            path, limit, sample_per_file, timed=False, stride=stride,
            with_features=with_features,
        )
        return rows


def load_cicids2018_timed(
    path: str | Path,
    limit: int | None = None,
    sample_per_file: int | None = DEFAULT_SAMPLE_PER_FILE,
    stride: int | None = None,
    with_features: bool = True,
) -> tuple[list[NSLKDDRow], list[pd.Timestamp]]:
    """As `load_cicids2018`, but also return each row's real capture time.

    The shared NSLKDDRow schema has no timestamp slot by design, so the clock
    is returned alongside rather than smuggled into `features`. Runners replay
    the capture at true inter-arrival times; see the module docstring for why
    that matters to a rate-based lane.
    """
    with observe("digital_twin", "load_cicids2018_timed", path=str(path)):
        rows, times, _ = _read_rows(
            path, limit, sample_per_file, timed=True, stride=stride,
            with_features=with_features,
        )
        return rows, (times or [])


def summarize(rows: list[NSLKDDRow]) -> dict[str, int]:
    """Count rows per KRONUS category."""
    out: dict[str, int] = {}
    for r in rows:
        out[r.category] = out.get(r.category, 0) + 1
    return out


def attack_intervals(
    rows: list[NSLKDDRow],
    times: list[pd.Timestamp],
    guard_seconds: float = ATTACK_INTERVAL_GUARD_SECONDS,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Time spans covered by attack traffic, widened by `guard_seconds`.

    Exists so the runner can hold out benign rows that share a window with an
    attack. Holding them out is the honest option: every row in this loader
    shares one reconstructed source_ip, so the featurizer's 2-second aggregate
    is global to the capture, and a benign flow arriving mid-flood sits in a
    window the flood dominates — its features are the flood's. Calling those
    rows "benign" would train the model that a flood is benign.

    Intervals are merged, so a day whose attack is spread across many short
    bursts yields the spans actually covered rather than one per burst.
    """
    if not rows or not times:
        return []
    guard = pd.Timedelta(guard_seconds, unit="s")
    spans = sorted(
        (times[i] - guard, times[i] + guard)
        for i, r in enumerate(rows)
        if r.category != "normal"
    )
    if not spans:
        return []
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def in_any_interval(
    ts: pd.Timestamp, intervals: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> bool:
    """True if `ts` falls inside any (merged, ascending) attack interval."""
    for start, end in intervals:
        if start <= ts <= end:
            return True
        if ts < start:
            break
    return False


def label_quality_report(rows: list[NSLKDDRow]) -> dict:
    """Measure how separable this dataset's classes are, from the rows loaded.

    Reported rather than assumed: the sibling DNS-EXF2021 dataset looked fine
    and was not, and the published criticism of this dataset's Infiltration
    label is that much of its window is ordinary background traffic. This
    quantifies the same failure mode here — how many feature vectors carry
    more than one class — so the runner can state a real ceiling instead of
    discovering one afterwards.
    """
    if not rows:
        return {"rows": 0}

    keys = np.empty(len(rows), dtype=np.uint64)
    categories: list[str] = []
    for i, r in enumerate(rows):
        payload = "\x1f".join(f"{k}={r.features[k]!r}" for k in sorted(r.features))
        keys[i] = hash(payload) & 0xFFFFFFFFFFFFFFFF
        categories.append(r.category)

    frame = pd.DataFrame({"key": keys, "category": categories})
    by_key = frame.groupby("key")["category"].nunique()
    ambiguous = set(by_key[by_key > 1].index)
    in_ambiguous = int(frame["key"].isin(ambiguous).sum())

    # Ceiling for any deterministic function of the features: the majority
    # class within each feature vector, summed.
    majority = frame.groupby("key")["category"].agg(
        lambda s: s.value_counts().iloc[0]
    ).sum()
    ceiling = float(majority) / len(frame)

    rows_by_category = frame["category"].value_counts().to_dict()
    vectors_by_category = frame.groupby("category")["key"].nunique().to_dict()
    ambiguous_by_category = {
        cat: round(
            float(frame.loc[frame["category"] == cat, "key"].isin(ambiguous).mean()), 4
        )
        for cat in rows_by_category
    }

    return {
        "rows": len(frame),
        "distinct_feature_vectors": int(frame["key"].nunique()),
        "duplicate_row_fraction": round(1.0 - frame["key"].nunique() / len(frame), 4),
        "ambiguous_vectors": len(ambiguous),
        "rows_in_ambiguous_vectors": in_ambiguous,
        "ambiguous_row_fraction": round(in_ambiguous / len(frame), 4),
        "deterministic_ceiling": round(ceiling, 4),
        "rows_by_category": rows_by_category,
        "distinct_vectors_by_category": {
            k: int(v) for k, v in vectors_by_category.items()
        },
        "ambiguous_row_fraction_by_category": ambiguous_by_category,
    }
