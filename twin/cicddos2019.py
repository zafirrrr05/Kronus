"""CIC-DDoS2019 external-validation data source.

CIC-DDoS2019 ("DDoS Evaluation Dataset", Canadian Institute for Cybersecurity,
University of New Brunswick) is the reflection/amplification DDoS benchmark —
the dataset papers cite when they want volumetric attack traffic that is not
CSE-CIC-IDS2018's application-layer floods. Official page:
https://www.unb.ca/cic/datasets/ddos-2019.html

    Sharafaldin, Lashkari, Hakak, Ghorbani, "Developing Realistic Distributed
    Denial of Service (DDoS) Attack Dataset and Taxonomy", ICCST 2019.
    doi:10.1109/ccst.2019.8888419

WHY THE SCHEMA IS A TRAP, AND HOW THIS MODULE AVOIDS IT
This release's CSVs mix two of CIC's conventions. The socket columns look like
CIC-IDS2017's ("Source IP", "Total Length of Fwd Packets"), while the label
vocabulary is this dataset's own (DrDoS_LDAP, UDPLag, Syn, ...). That
combination is precisely what a retired loader in this tree was shaped for: it
read CIC-IDS2017's column names while being named for CSE-CIC-IDS2018 and
mapped this vocabulary, and against real files it returned an all-zero dataset
while reporting success. Two guards exist here so that cannot recur:

  1. `_assert_schema` hard-fails on a file whose header is not this dataset's.
     Only the header is read, so the check is free, and the failure is loud.
  2. `resolve_class` drops an unrecognized label and counts it. An attack
     family that nobody wrote a mapping for becomes a *reported drop*, never a
     silent flood.

WHAT THIS LOADER DOES NOT HAVE TO DO, UNLIKE THE 2018 ONE NEXT DOOR
The ML-ready CSVs carry the **real** Source IP, Source Port, Destination IP,
Destination Port and a wall-clock Timestamp. So:

  * No host reconstruction. `synthetic_hosts` is false for this experiment,
    and the graph shape a flood produces is the capture's own.
  * No attack-interval benign holdout, but the reason is the real source IP
    rather than an absence of overlap. The Bouncer's featurizer keys its
    2-second window by `source_ip`, and here the source IP is real, so a benign
    flow's window contains that host's own traffic. In CSE-CIC-IDS2018 every
    row shared one reconstructed source, which made the window aggregate global
    to the capture and forced concurrent benign rows to be held out.
    Here the overlap is real but tiny, and measured rather than assumed: the
    flood comes from the attacker (`172.16.0.5`) plus the victim
    (`192.168.50.1` on 01-12, `192.168.50.4` on 03-11), and only the victim
    also carries benign rows — 330 of them on 01-12 and 375 on 03-11, 0.66% of
    benign in each case. The runner drops exactly those rows so that no benign
    window contains flood traffic, and reports the drop in the metrics.
    `host_overlap_report` measures the overlap rather than asserting it.

THE CLOCK IS REAL, AS IN EXPERIMENT F
Rows are returned with their capture time so the runner can replay at true
inter-arival spacing; a 2-second window is a real 2 seconds. The clock is
excluded from `features` for the same reason as everywhere else: an attack day
is a different day from a benign one, so a timestamp would hand over the label.

WHY THIS IS A BOUNCER-ONLY EXPERIMENT
Every labelled attack family in this dataset is volumetric — reflection and
amplification floods (LDAP, MSSQL, NetBIOS, Portmap, SNMP, SSDP, DNS, NTP,
TFTP) and direct floods (UDP, UDPLag, Syn, WebDDoS). There is no port-scan or
lateral-movement class anywhere in it. The Bouncer's contract is flood-vs-
benign, which fits exactly; the Detective has nothing to train on, so it is
skipped and the metrics say so — the mirror image of Experiment D, which was
Detective-only for the opposite reason.

STORAGE
The release ships as ZIP archives of per-attack CSVs. Reading a CSV member
straight out of its archive (rather than extracting first) keeps peak disk use
at the archive size instead of the several GB the expansion would cost, which
matters on a laptop. `_iter_members` therefore opens members in place; a plain
directory of CSVs is accepted too, for a caller who has already unpacked them.

Output type is the same `NSLKDDRow` shape every other loader here produces, so
the converters, the Bouncer featurizer and the graph builder are unchanged.
"""

from __future__ import annotations

import io
import warnings
import zipfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from libs.constants import DataOrigin, Label, Protocol
from twin.nsl_kdd import NSLKDDRow

# --- the real 88-column header ---------------------------------------------
# Column names are the release's, verbatim, leading spaces included: the
# header is internally inconsistent about them (" Source IP" has one,
# "Total Length of Fwd Packets" does not), so `_clean_columns` strips them and
# everything below is written against the stripped form.
COL_INDEX = "Unnamed: 0"
COL_FLOW_ID = "Flow ID"
COL_SRC_IP = "Source IP"
COL_SRC_PORT = "Source Port"
COL_DST_IP = "Destination IP"
COL_DST_PORT = "Destination Port"
COL_PROTOCOL = "Protocol"
COL_TIMESTAMP = "Timestamp"
COL_DURATION = "Flow Duration"                 # microseconds
COL_FWD_BYTES = "Total Length of Fwd Packets"  # 2017-style naming, not 2018's
COL_BWD_BYTES = "Total Length of Bwd Packets"
COL_SIMILAR_HTTP = "SimillarHTTP"
COL_INBOUND = "Inbound"
COL_LABEL = "Label"

# The columns a file must have to be this dataset. Deliberately the identity
# columns rather than the feature columns: the 2017 schema shares "Flow
# Duration" and "Timestamp" with this one, so those two cannot distinguish it,
# while " Dst Port" (2018), "Destination Port" (2017) and " Source IP" name
# three different releases.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_SRC_IP,
    COL_SRC_PORT,
    COL_DST_IP,
    COL_DST_PORT,
    COL_PROTOCOL,
    COL_TIMESTAMP,
    COL_FWD_BYTES,
    COL_BWD_BYTES,
    COL_LABEL,
)

# Identity, label and non-measurement columns, removed from `features` so the
# model sees only flow measurements:
#   * the index and Flow ID are row identity, and Flow ID *contains* the IPs
#     and ports concatenated — leaving it in would hand the label over twice;
#   * the five socket columns and the timestamp are the leak surface described
#     in the module docstring;
#   * the label itself.
# `SimillarHTTP` and `Inbound` are handled separately, below.
_EXCLUDED_FEATURES: frozenset[str] = frozenset({
    COL_INDEX,
    COL_FLOW_ID,
    COL_SRC_IP,
    COL_SRC_PORT,
    COL_DST_IP,
    COL_DST_PORT,
    COL_PROTOCOL,
    COL_TIMESTAMP,
    COL_LABEL,
})

# `Inbound` is excluded because it is measured to be a label proxy, not a flow
# measurement. An earlier version of this comment said it was a per-file
# constant; that is FALSE — both values appear within every member — but the
# exclusion is right for a stronger reason. `Inbound` marks a flow arriving at
# the monitored host, and a volumetric flood is inbound by construction, so the
# column tracks the label. Read as a classifier ("attack iff Inbound == 1") it
# scores 99.35% on 03-11/Portmap, 99.61% on 01-12/UDPLag, 99.73% on 03-11/UDPLag
# and 99.98% on 01-12/Syn. Keeping it would let the Bouncer separate attack from
# benign without looking at any traffic. `SimillarHTTP` is excluded for a weaker
# but sufficient reason — it is a constant 0 and carries no signal. Neither fact
# can be asserted by a unit test, which sees only synthetic fixtures; the runner
# re-measures the label census against the real archive and reports it.
_ARTIFACT_COLUMNS: frozenset[str] = frozenset({COL_SIMILAR_HTTP, COL_INBOUND})

PROTOCOL_NUMBER_MAP: dict[int, Protocol] = {
    6: Protocol.TCP,
    17: Protocol.UDP,
    1: Protocol.ICMP,
}

# The label vocabulary of this release. Every entry is an attack family the
# dataset's own taxonomy documents, and every one is volumetric, so all of them
# map to KRONUS's single flood class. The set is explicit rather than a
# catch-all so an unknown family is dropped and counted (see resolve_class)
# instead of being assumed to be a flood.
_FLOOD_FAMILIES: frozenset[str] = frozenset({
    "ldap", "mssql", "netbios", "portmap", "snmp", "ssdp",
    "udp", "udplag", "syn", "tftp", "webddos", "dns", "ntp",
})

# Applied when a family name is prefixed (the release writes both
# "DrDoS_LDAP" and a bare "LDAP" depending on the file).
_DRDOS_PREFIXES: tuple[str, ...] = ("drdos_", "drdos", "drdos ")

DEFAULT_SAMPLE_PER_FILE = 60_000
_CHUNK_ROWS = 25_000

# The release's own timestamp format: "2018-12-01 13:34:27.403713". Kept as an
# explicit format (rather than inferred) so a malformed row becomes a NaN the
# caller can see, instead of a silently re-interpreted day/month order.
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


# --- label resolution -------------------------------------------------------

def _normalize_label(value: object) -> str:
    """Casefold, strip, and collapse the separators the release mixes.

    The release writes family names with underscores ("DrDoS_LDAP"), with a
    space ("DrDoS LDAP"), and bare ("LDAP") — all three in the same file set.
    """
    text = str(value).strip().casefold()
    for separator in ("-", " ", "\t"):
        text = text.replace(separator, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def resolve_class(raw_label: object) -> tuple[str, Label] | None:
    """Map a raw label to (category, KRONUS label), or None if unrecognized.

    Returns None rather than a default so an unmapped family is counted as a
    drop by the caller. Silently calling an unknown label a flood would be the
    same class of error as the loader this module's docstring describes.
    """
    text = _normalize_label(raw_label)
    if not text:
        return None
    if text == "label":
        # The release's own embedded header row, seen mid-file.
        return None
    if text in {"benign", "normal"}:
        return ("normal", Label.BENIGN)

    family = text
    for prefix in _DRDOS_PREFIXES:
        if family.startswith(prefix):
            family = family[len(prefix):]
            break
    family = family.strip("_")
    if family in _FLOOD_FAMILIES:
        return ("dos", Label.FLOOD)
    return None


# --- reading ----------------------------------------------------------------

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the release's inconsistent leading spaces from column names."""
    df.columns = [str(c).strip() for c in df.columns]
    return df


@dataclass(frozen=True)
class _Member:
    """One CSV to read, and how to open it.

    A zip member cannot be re-opened after its ZipFile is closed, so the
    archive is held open for the life of the load and each member records its
    own name within it. This is what lets the loader stream a 1.5 GB archive
    without unpacking it.
    """

    name: str
    size: int
    archive: zipfile.ZipFile | None
    path: Path | None

    def open(self) -> io.BufferedReader | io.BytesIO:
        if self.archive is not None:
            return self.archive.open(self.name, "r")
        assert self.path is not None
        return open(self.path, "rb")  # noqa: SIM115 - closed by the caller


def _iter_members(path: Path) -> tuple[list[_Member], zipfile.ZipFile | None]:
    """Resolve `path` to CSV members, in a stable order.

    Accepts the downloaded archives (a .zip, or a directory holding them) as
    well as a directory of already-extracted CSVs, so a caller who has
    unpacked the release by hand is not forced back through the archive.
    """
    if path.is_file() and path.suffix.lower() == ".zip":
        return _zip_members(path), None

    if path.is_dir():
        zips = [p for p in sorted(path.glob("*.zip")) if not p.name.startswith(".")]
        if zips:
            members: list[_Member] = []
            for archive_path in zips:
                members.extend(_zip_members(archive_path))
            return members, None
        csvs = [p for p in sorted(path.rglob("*.csv")) if not p.name.startswith(".")]
        if csvs:
            return [
                _Member(name=p.name, size=p.stat().st_size, archive=None, path=p)
                for p in csvs
            ], None

    raise FileNotFoundError(
        f"{path} is not a CIC-DDoS2019 archive or a directory of CSVs. "
        "Run scripts/download_cicddos2019.py first (see docs/experiments_guide.md)."
    )


def _zip_members(archive_path: Path) -> list[_Member]:
    archive = zipfile.ZipFile(archive_path)
    infos = [
        info
        for info in archive.infolist()
        if info.filename.lower().endswith(".csv") and not info.is_dir()
    ]
    # Sorted by name so a bounded stride samples the same rows on every run,
    # whatever order the archive happens to store its members in.
    infos.sort(key=lambda i: i.filename)
    return [
        _Member(
            name=info.filename,
            size=info.file_size,
            archive=archive,
            path=None,
        )
        for info in infos
    ]


def _count_data_rows(member: _Member) -> int:
    """Count newlines cheaply, so the stride can be exact and reproducible."""
    total = 0
    with member.open() as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            total += block.count(b"\n")
    return max(total - 1, 0)  # minus the header line


def _feature_columns(columns: list[str]) -> list[str]:
    """The feature columns, fixed by the header rather than by dtypes.

    Deriving these from dtypes would make them depend on file content, and a
    feature set that changes between chunks cannot be joined or de-duplicated.
    The header is the same for every chunk, so it is the authority.
    """
    return [
        c for c in columns
        if c not in _EXCLUDED_FEATURES and c not in _ARTIFACT_COLUMNS
    ]


def _assert_schema(member: _Member) -> None:
    """Hard-fail on a file that is not the CIC-DDoS2019 schema.

    Reading only the header costs nothing and is the difference between a
    loader that silently returns zeros and one that refuses. The 2017 and 2018
    schemas both share "Flow Duration" and "Timestamp" with this one, so the
    check is written against the columns that actually differ.
    """
    with member.open() as handle:
        header = pd.read_csv(handle, nrows=0, encoding="utf-8", encoding_errors="replace")
    columns = list(_clean_columns(header).columns)
    missing = [c for c in _REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(
            f"{member.name} is missing required column(s) {missing}; it has "
            f"{len(columns)} columns starting {columns[:6]}. This is not a "
            "CIC-DDoS2019 CSV — refusing to load it rather than returning an "
            "empty dataset under this dataset's name."
        )


def _iter_chunks(
    member: _Member,
    sample_per_file: int | None,
    stride: int | None,
) -> Iterator[tuple[pd.DataFrame, np.ndarray, list[str]]]:
    """Yield (chunk, kept row positions, feature columns).

    Memory stays bounded by `_CHUNK_ROWS` and every entry is read at most
    once, streaming out of the archive. Either way of bounding the rows takes
    an exact, evenly spaced stride over the whole file, so a bounded sample
    spans the capture end to end — including its attack window — and
    reproduces exactly on a re-run.

    `stride` and `sample_per_file` are not interchangeable, and the difference
    matters to any *rate* feature:

      stride           one decimation factor applied to every file, so rows
                       kept from file A and file B are thinned by the same
                       factor and their event rates stay comparable.
      sample_per_file  a per-file row budget; the factor differs per file, so
                       a larger file is thinned harder and its rates are NOT
                       comparable with a smaller file's. Fine for counting
                       rows, wrong for comparing burst rates across files.
    """
    if stride is not None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        effective_stride = stride
    else:
        total_rows = _count_data_rows(member) if sample_per_file else 0
        effective_stride = (
            max(1, total_rows // sample_per_file)
            if (sample_per_file and total_rows)
            else 1
        )
    budget = sample_per_file if stride is None else None
    kept = 0
    seen = 0

    with member.open() as handle:
        reader = pd.read_csv(
            handle,
            chunksize=_CHUNK_ROWS,
            low_memory=True,
            encoding="utf-8",
            encoding_errors="replace",
        )
        while True:
            # The release contains repeated header rows mid-file, which make
            # pandas read a chunk's numeric columns as `object`. That is
            # expected and handled: features are coerced explicitly below, and
            # `_feature_columns` takes its column list from the header rather
            # than from dtypes precisely so one stray row cannot change it.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
                try:
                    chunk = next(reader)
                except StopIteration:
                    return
            chunk = _clean_columns(chunk)
            if COL_LABEL not in chunk.columns:
                raise ValueError(
                    f"{member.name} has no 'Label' column; columns seen: "
                    f"{list(chunk.columns)[:8]}... — this is not a "
                    "CIC-DDoS2019 CSV."
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


class _MemberChunk:
    """Column-wise view of one kept slice of a CSV.

    Converting a whole chunk with `to_dict("records")` costs a dict per row
    across 88 columns and then a `float()` per cell — for the rows a full run
    touches that is tens of millions of Python calls. Everything the row
    builder needs is pulled out as numpy arrays once, here, and the per-row
    work is array indexing.
    """

    __slots__ = ("labels", "header_mask", "src_ips", "dst_ips", "src_ports",
                 "dst_ports", "protocols", "durations_us", "fwd_bytes",
                 "bwd_bytes", "features", "feature_cols", "times")

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

        def text(col: str) -> np.ndarray:
            if col not in sub.columns:
                return np.array([""] * len(sub), dtype=object)
            return sub[col].astype(str).str.strip().to_numpy()

        # The socket columns are real here, so they are carried through as
        # text rather than reconstructed from anything.
        self.src_ips = text(COL_SRC_IP)
        self.dst_ips = text(COL_DST_IP)
        self.src_ports = num(COL_SRC_PORT)
        self.dst_ports = num(COL_DST_PORT)
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
            ts = pd.to_datetime(sub[COL_TIMESTAMP], format=_TIMESTAMP_FORMAT, errors="coerce")
            if ts.isna().all():
                # Fall back to inference only if the strict format matched
                # nothing at all — a partial match means the format is right.
                ts = pd.to_datetime(sub[COL_TIMESTAMP], errors="coerce")
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
            source_ip = str(self.src_ips[i]).strip()
            dest_ip = str(self.dst_ips[i]).strip()
            if not source_ip or not dest_ip:
                # A row with no socket identity cannot be placed in the graph
                # or in a per-host window; dropping it is the honest choice.
                yield i, None
                continue
            features = {
                col: float(self.features[i, j]) for j, col in enumerate(self.feature_cols)
            }
            yield i, NSLKDDRow(
                source_ip=source_ip,
                dest_ip=dest_ip,
                source_port=_port_or_none(self.src_ports[i]),
                dest_port=int(self.dst_ports[i]),
                protocol=PROTOCOL_NUMBER_MAP.get(int(self.protocols[i]), Protocol.OTHER),
                total_bytes=int(max(self.fwd_bytes[i] + self.bwd_bytes[i], 0.0)),
                duration_ms=max(int(self.durations_us[i] / 1000.0), 0),
                raw_label=str(self.labels[i]).strip(),
                category=category,
                kronus_label=kronus_label,
                difficulty=0,
                features=features,
                origin=DataOrigin.REAL,
            )


def _port_or_none(value: float) -> int | None:
    """A real source port, or None when the release recorded no usable one."""
    port = int(value)
    return port if 0 <= port <= 65535 else None


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
    members, _ = _iter_members(path)
    rows: list[NSLKDDRow] = []
    times: list = []
    unmapped: dict[str, int] = {}
    per_file: dict[str, int] = {}

    try:
        for member in members:
            if limit is not None and len(rows) >= limit:
                break
            _assert_schema(member)
            kept_here = 0
            for chunk, positions, feature_cols in _iter_chunks(
                member, sample_per_file, stride
            ):
                if not with_features:
                    feature_cols = []
                view = _MemberChunk(chunk, positions, feature_cols, timed)
                for i, row in view.rows():
                    if row is None:
                        # Count *why* it was dropped, so an unmapped label is
                        # visible in the report instead of just missing.
                        raw = str(view.labels[i]).strip()
                        if not view.header_mask[i]:
                            unmapped[raw] = unmapped.get(raw, 0) + 1
                        continue
                    rows.append(row)
                    if timed:
                        times.append(view.times[i])
                    kept_here += 1
                    if limit is not None and len(rows) >= limit:
                        break
                if limit is not None and len(rows) >= limit:
                    break
            per_file[member.name] = kept_here
    finally:
        for member in members:
            if member.archive is not None:
                member.archive.close()

    report = {
        "unmapped_labels": dict(sorted(unmapped.items(), key=lambda kv: -kv[1])),
        "rows_per_file": per_file,
    }

    # Sort into the capture's true chronological order. The clock is the
    # release's own; it orders the replay and is deliberately not carried into
    # `features`. Members are read in name order, which is not time order —
    # the archives group flows by attack family, not by when they arrived — so
    # without this an interleaved capture would replay in the wrong order and
    # every rate feature built from it would be wrong.
    if rows and times:
        order = sorted(range(len(rows)), key=lambda i: (times[i], i))
        rows = [rows[i] for i in order]
        times = [times[i] for i in order]

    return rows, (times if timed else None), report


# --- public API -------------------------------------------------------------

def load_cicddos2019(
    path: str | Path,
    limit: int | None = None,
    sample_per_file: int | None = DEFAULT_SAMPLE_PER_FILE,
    stride: int | None = None,
    with_features: bool = True,
) -> list[NSLKDDRow]:
    """Load rows from an archive (or a directory of them, or of extracted CSVs)."""
    rows, _, _ = _read_rows(path, limit, sample_per_file, False, stride, with_features)
    return rows


def load_cicddos2019_timed(
    path: str | Path,
    limit: int | None = None,
    sample_per_file: int | None = DEFAULT_SAMPLE_PER_FILE,
    stride: int | None = None,
    with_features: bool = True,
) -> tuple[list[NSLKDDRow], list]:
    """Load rows with their real capture times, aligned index-for-index.

    This dataset has a real clock and the runner replays at true inter-arrival
    times, so the times must come back parsed rather than inferred.
    """
    rows, times, _ = _read_rows(path, limit, sample_per_file, True, stride, with_features)
    assert times is not None
    return rows, times


def load_cicddos2019_report(
    path: str | Path,
    limit: int | None = None,
    sample_per_file: int | None = DEFAULT_SAMPLE_PER_FILE,
    stride: int | None = None,
) -> tuple[list[NSLKDDRow], dict]:
    """Load rows plus the drop report (unmapped labels, rows kept per file)."""
    rows, _, report = _read_rows(path, limit, sample_per_file, False, stride, True)
    return rows, report


def summarize(rows: list[NSLKDDRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.category] = counts.get(row.category, 0) + 1
    return dict(sorted(counts.items()))


def label_quality_report(rows: list[NSLKDDRow]) -> dict:
    """Label consistency: rows whose feature vector carries two classes.

    This is the measurement that decides whether a weak score is the model's
    fault or the label's. Two rows with byte-identical features and different
    labels are irreducible — no function of the features can get both right,
    so they bound what any model can score.
    """
    if not rows:
        return {"rows": 0}
    by_key: dict[tuple, Counter] = {}
    for row in rows:
        key = tuple(sorted(row.features.items()))
        by_key.setdefault(key, Counter())[row.category] += 1

    total = len(rows)
    ambiguous_keys = [k for k, counts in by_key.items() if len(counts) > 1]
    rows_in_ambiguous = sum(sum(by_key[k].values()) for k in ambiguous_keys)
    # A deterministic function of the features can do no better than the
    # majority class on each duplicated vector.
    achievable = total - rows_in_ambiguous + sum(
        max(by_key[k].values()) for k in ambiguous_keys
    )
    return {
        "rows": total,
        "distinct_vectors": len(by_key),
        "ambiguous_vectors": len(ambiguous_keys),
        "rows_in_ambiguous_vectors": rows_in_ambiguous,
        "ambiguous_row_fraction": round(rows_in_ambiguous / total, 4),
        "deterministic_ceiling": round(achievable / total, 4),
        "by_category": summarize(rows),
    }


def host_overlap_report(rows: list[NSLKDDRow]) -> dict:
    """Do attack and benign flows share a source host?

    The 2018 loader next door needed an attack-interval benign holdout because
    every row shared one reconstructed source IP, making the Bouncer's window
    aggregate global to the capture. This dataset carries real source IPs, so
    the question becomes empirical: if the attack hosts and the benign hosts
    are disjoint, a benign flow's 2-second window cannot contain a flood and
    the holdout is unnecessary. Returned as a measurement, not an assumption.
    """
    attacks = {r.source_ip for r in rows if r.kronus_label is Label.FLOOD}
    benign = {r.source_ip for r in rows if r.kronus_label is Label.BENIGN}
    return {
        "attack_hosts": len(attacks),
        "benign_hosts": len(benign),
        "shared_hosts": len(attacks & benign),
        "attack_hosts_that_also_send_benign": sorted(attacks & benign)[:10],
    }
