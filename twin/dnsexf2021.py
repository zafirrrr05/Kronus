"""The CIC-Bell-DNS-EXF-2021 external-validation loader.

Dataset: CIC-Bell-DNS-EXF-2021 (Canadian Institute for Cybersecurity, UNB) —
the DNS exfiltration capture behind:

    Samaneh Mahdavifar, Ali A. Ghorbani, "DeNAs: Deep Network Attack
    Signature and its Application to DNS Exfiltration Detection", 2021.

The dataset itself is described by UNB as "a collection of benign and
malicious DNS traffic" covering DNS tunnelling / exfiltration via several
tools and payload types, split into a *benign* corpus and two attack corpora
(heavy and light).

READ THIS BEFORE TRUSTING ANY METRIC THIS LOADER FEEDS
=====================================================
This is the one external dataset in KRONUS whose **labels are not recoverable
from its own features**, and that is measured rather than suspected. Every
number below was obtained from the released per-class files themselves by
hashing each row's full observable feature vector (all 14 columns except
`timestamp` and `Label`) and grouping:

    536,138 rows  ->  46,462 distinct feature vectors   (91.3% exact dupes)
    64 vectors carry more than one KRONUS class, involving 390,215 rows
    deterministic ceiling (benign vs lateral_movement) = 0.8210

On the dataset's own three-way split (benign / heavy_attack / light_attack)
the ceiling is lower still — 0.7414 — because vectors shared between the
heavy and light corpora collide there too; merging them under KRONUS's single
`lateral_movement` label is what lifts the bound to 0.8210.

A ceiling is a hard bound, not a model limitation: where one feature vector
occurs as `lateral_movement` 2,496 times and `normal` 1,092 times, no function
of those features can be right on more than 2,496/3,588 of those rows. The
collisions are not marginal — the *same* base32 exfiltration payload
(`FHEPFCELEHFCEPFFFACACACACACACABN`) appears with an identical feature vector
under both labels. The cause is that the label is a **capture-level**
annotation — "this capture contained an exfiltration run" — stamped onto every
row of the capture, including the ordinary DNS lookups the monitored machine
made while the attack was running. The dataset's own accompanying README
records the same collapse.

The consequence is more severe than a low ceiling. Of 294,353 exfiltration
rows, **294,204 — 99.95% — sit on a feature vector that also carries a benign
label.** Dropping the ambiguous vectors, which is what this loader does by
default, therefore leaves the attack class with **149 rows carrying only 32
distinct feature signatures**, against 46,366 signatures of benign traffic.
Because the loader also de-duplicates, those 149 rows collapse further: what
the experiment actually trains on is **32 attack rows against 46,366 benign
rows** — a 1:1,449 class ratio, and every attack row is a signature seen
exactly once.
The entire exfiltration class in this dataset is a set of roughly 96 distinct
query signatures repeated thousands of times each, and almost every one of
them is indistinguishable from ordinary DNS by these features.

There is a second, independent problem: the dataset carries **no IP address,
no port, no byte count and no flow duration**. Its columns are entirely DNS
query statistics plus the queried name. So every quantity the KRONUS lanes
actually consume has to be reconstructed, and the one genuinely discriminating
real signal (`entropy`, the domain's character entropy) is not something
either lane reads.

The honest consequence, recorded in the experiment metrics: **no valid
detection metric can be produced from this dataset.** A metric computed after
the guard below describes a 32-signature lookup table; a metric computed
without it is bounded by 0.8210 and is dominated by label noise. The
experiment reports both facts and labels the pipeline output as
non-reportable rather than quoting either as a result.

WHAT THIS LOADER DOES ABOUT IT
------------------------------
Two preprocessing steps, both disclosed in the metrics, keep the experiment
from being actively false rather than merely weak:

1. **Ambiguous vectors are dropped.** Any feature vector carrying more than
   one per-file class is removed entirely (`drop_ambiguous=True`, the
   default). What remains has a well-defined label by construction, so the
   reported score is not measuring the label noise itself.
2. **Vectors are de-duplicated.** The 536,138 rows collapse to one row per
   distinct vector, because 57,469 copies of one observation is one
   observation — keeping them would weight it 57,469x in training and would
   also put byte-identical rows in both the train and the test split, which
   would inflate the score with pure memorisation.

KRONUS MAPPING
--------------
    benign_labeled/       benign DNS           -> normal           -> Label.BENIGN
    heavy_attack_labeled/ exfiltration (heavy) -> lateral_movement -> Label.LATERAL_MOVEMENT
    light_attack_labeled/ exfiltration (light) -> lateral_movement -> Label.LATERAL_MOVEMENT

`lateral_movement` is KRONUS's label for tunnelled/exfiltrating traffic (see
libs/constants.py's fixed Label enum) — a host moving data out through an
established channel. Nothing here is a volumetric flood, so no row maps to
Label.FLOOD and scripts/run_dnsexf2021_experiment.py does not train the
Bouncer. That is the same reasoning as twin/dohbrw2020.py.

RECONSTRUCTION (disclosed as `"synthetic_hosts": true`)
------------------------------------------------------
`source_ip`: a single constant monitored client (MONITORED_CLIENT_IP). The
dataset records no client identity at all, and inventing a per-class-varying
source would hand the model a perfect label proxy — the reconstruction is
therefore deliberately class-independent.

`dest_ip`: a deterministic synthetic IPv4 derived from the row's real `sld`
(the queried second-level domain). This is the one piece of genuine graph
content available here: because the address is a pure function of the domain,
the number of distinct destinations in a window IS the number of distinct
domains actually queried, with no invention in between.

`dest_port` = 53 and `protocol` = UDP: DNS is what this dataset is, so these
are the dataset's subject matter rather than invented measurements.
`source_port` is None — genuinely absent, never a stand-in number.

`total_bytes`: derived from the real `len` column (the query name's length)
via a documented DNS packet-size formula. The dataset records no byte count,
so this is a derivation, not a measurement.

`duration_ms` = 0: the dataset records no flow duration. It is left at zero
rather than fabricated; nothing downstream depends on it here, since the
Bouncer (the lane that reads durations) does not train on this dataset.
"""

from __future__ import annotations

import hashlib
import re
import sys
import zlib
from pathlib import Path

import pandas as pd

from libs.constants import DataOrigin, Label, Protocol
from twin.nsl_kdd import NSLKDDRow

# Per-class directory (or filename) token -> (KRONUS category, KRONUS label).
# The class comes from the DIRECTORY the file sits in, which is how this
# dataset is actually organised. The filename alone is not safe to use: the
# benign corpus ships a file called `stateless_features-light_benign.csv`,
# whose tokens contain BOTH "light" and "benign", so a first-token-wins scan
# over filenames would file benign traffic under an attack class.
DNS_EXF_CLASSES: dict[str, tuple[str, Label]] = {
    "benign": ("normal", Label.BENIGN),
    "heavy": ("lateral_movement", Label.LATERAL_MOVEMENT),
    "light": ("lateral_movement", Label.LATERAL_MOVEMENT),
}

# The three published per-class directories, in a fixed order.
DNS_EXF_DIRS: list[str] = ["benign_labeled", "heavy_attack_labeled", "light_attack_labeled"]

# Columns that describe the container rather than the query.
IDENTITY_COLUMNS: tuple[str, ...] = ("timestamp", "Label")

# The full observable feature vector — the 14 columns a model could read. This
# is what the ambiguity measurement groups on, so it must stay complete.
FEATURE_COLUMNS: list[str] = [
    "FQDN_count", "subdomain_length", "upper", "lower", "numeric", "entropy",
    "special", "labels", "labels_max", "labels_average", "longest_word",
    "sld", "len", "subdomain",
]

# The subset of the above that can live in NSLKDDRow.features, which is
# float-valued. `sld` and `longest_word` are strings (the latter is usually a
# number but carries the literal "N" in some rows) and so are carried through
# the graph instead — `sld` becomes the destination address.
_STRING_FEATURE_COLUMNS = frozenset({"sld", "longest_word"})

# The dataset has no client identity; attributing every row to one monitored
# host is the only class-independent choice. See the module docstring.
MONITORED_CLIENT_IP = "10.30.0.1"

# DNS is the dataset's subject matter, so these are its semantics, not guesses.
DNS_PORT = 53
DNS_PROTOCOL = Protocol.UDP

# A DNS query on the wire: 20-byte IPv4 header + 8-byte UDP header + 12-byte
# DNS header + 4-byte qtype/qclass, plus the query name itself (`len`). Used
# to derive `total_bytes` from a real column, since no byte count is recorded.
_DNS_OVERHEAD_BYTES = 20 + 8 + 12 + 4


def _tokens(text: str) -> list[str]:
    # Split on ANY non-alphanumeric run, not just "_": these filenames mix
    # separators ("stateless_features-light_benign"), and splitting on "_"
    # alone would leave "features-light" glued together and hide the fact that
    # the name carries two contradictory class words.
    return [t for t in re.split(r"[^a-z0-9]+", str(text).casefold()) if t]


def resolve_class(name: str, label_value: str = "") -> tuple[str, Label] | None:
    """Map a directory (then file) name onto (category, KRONUS label).

    A source that carries *conflicting* class tokens resolves to nothing
    rather than to whichever token happens to come first: the released benign
    corpus contains `stateless_features-light_benign.csv`, whose name holds
    both "light" and "benign". First-token-wins would file benign traffic
    under an attack class, so this returns None and the caller tries the next
    source. The directory is the authoritative source — see DNS_EXF_CLASSES.
    """
    for source in (name, label_value):
        matches = [DNS_EXF_CLASSES[t] for t in _tokens(source) if t in DNS_EXF_CLASSES]
        if not matches:
            continue
        if len({category for category, _ in matches}) > 1:
            continue  # name contradicts itself — never guess
        return matches[0]
    return None


def _resolve_csv_paths(path: Path) -> list[Path]:
    """Accept a directory tree (or a single CSV) and return the CSVs to read.

    Only the three known per-class directories are walked when present, so an
    unrelated CSV sitting in the same tree cannot be silently folded into a
    class. A flattened directory still works via `resolve_class`'s filename
    fallback — including the mirror's combined
    `CIC-Bell-DNS-EXF-2021_stateless.csv`, which carries a `Label` column.
    """
    if path.is_dir():
        found: list[Path] = []
        for name in DNS_EXF_DIRS:
            sub = path / name
            if sub.is_dir():
                found.extend(sorted(sub.glob("*.csv")))
        if found:
            return found
        # No per-class tree. A flattened copy can still load (the label column
        # and the filename are consulted per row), but files whose names carry
        # contradictory class tokens — `stateless_features-light_benign.csv`
        # and `...-benign_heavy_*.csv` — resolve to nothing rather than being
        # guessed at, so say so loudly instead of silently dropping them.
        found = sorted({p for p in list(path.glob("*.csv")) + list(path.glob("**/*.csv"))})
        if found:
            print(
                f"warning: {path} has no per-class directories "
                f"({', '.join(DNS_EXF_DIRS)}); falling back to "
                f"{len(found)} flattened CSV(s). Files whose names carry "
                "conflicting class tokens will be skipped, not guessed.",
                file=sys.stderr,
            )
        if not found:
            raise FileNotFoundError(
                f"no CIC-Bell-DNS-EXF-2021 CSVs found at {path} — run "
                "scripts/download_dnsexf2021.py"
            )
        return found
    if path.suffix == ".csv" and path.exists():
        return [path]
    raise FileNotFoundError(
        f"{path} not found — run scripts/download_dnsexf2021.py"
    )


def _file_class(path: Path) -> tuple[str, Label] | None:
    """A file's class from its DIRECTORY (authoritative), then its name.

    The label column is deliberately NOT consulted here: a file's class is a
    property of where it lives, and an aggregate file can legitimately hold
    more than one class across its rows (`_row_class` handles that case).
    """
    return resolve_class(path.parent.name) or resolve_class(path.stem)


def _row_class(series: pd.Series, file_class: tuple[str, Label] | None):
    """A row's class: the file's, or failing that the row's own label value.

    The fallback is what makes the mirror's single combined
    `CIC-Bell-DNS-EXF-2021_stateless.csv` usable — its `Label` column varies
    per row, so resolving it once from the first row would mislabel the file.
    """
    if file_class is not None:
        return file_class
    return resolve_class(str(series.get("Label", "")))


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df


def _read_one_csv(path: Path, sample_per_file: int | None) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False, nrows=sample_per_file)
    except Exception:
        return None
    return _clean_columns(df)


def _vector_key(frame: pd.DataFrame) -> pd.Series:
    """A stable hash of each row's full observable feature vector.

    crc32 rather than `hash()` because the value must be identical across
    processes and runs — the ambiguity set is a property of the data, not of
    an interpreter session.
    """
    cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
    # A row-wise list comprehension rather than DataFrame.agg(axis=1): the
    # latter is orders of magnitude slower and this runs over every row of
    # every file, several times per experiment.
    values = frame[cols].astype(str).to_numpy()
    keys = [zlib.crc32("\x1f".join(row).encode()) for row in values]
    return pd.Series(keys, index=frame.index, dtype="int64")


def find_ambiguous_vectors(paths: list[Path], sample_per_file: int | None = None) -> dict[int, list[int]]:
    """Feature vectors carrying more than one per-file class.

    Returns {vector_key: [n_normal, n_lateral_movement]}. This is the
    measurement behind the module docstring's ceiling: it reads every row of
    every file once and keeps only the collisions, so its memory footprint is
    the size of that (small) map rather than of the dataset.
    """
    masks: dict[int, list[int]] = {}
    for csv_path in paths:
        df = _read_one_csv(csv_path, sample_per_file)
        if df is None or df.empty:
            continue
        file_class = _file_class(csv_path)
        for key, (_, series) in zip(_vector_key(df), df.iterrows(), strict=True):
            resolved = _row_class(series, file_class)
            if resolved is None:
                continue
            idx = 0 if resolved[0] == "normal" else 1
            row = masks.get(key)
            if row is None:
                masks[key] = row = [0, 0]
            row[idx] += 1
    return {k: v for k, v in masks.items() if sum(1 for c in v if c) > 1}


def _dest_ip_for(sld: str) -> str:
    """Deterministic synthetic IPv4 for a queried domain.

    A pure function of the real `sld`, so distinct-domain counts in a graph
    window are the dataset's own distinct-domain counts. Same construction as
    twin/nsl_kdd.py's `_synthetic_ip` (sha256 -> two octets), on a distinct
    prefix so these addresses cannot be confused with another loader's.
    """
    digest = hashlib.sha256(str(sld).encode()).hexdigest()
    h = int(digest[:8], 16)
    return f"10.40.{(h >> 8) & 0xFF}.{h & 0xFF}"


def _safe_float(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f != f or f in (float("inf"), float("-inf")):
        return 0.0
    return f


def _row_to_nslkdd_row(series: pd.Series, category: str, kronus_label: Label, raw_label: str) -> NSLKDDRow:
    sld = str(series.get("sld", "")).strip()

    # `len` is the query name's length, a real column; everything else in this
    # sum is protocol overhead, not a measurement. See the module docstring.
    name_len = max(int(_safe_float(series.get("len", 0))), 0)
    total_bytes = max(_DNS_OVERHEAD_BYTES + name_len, 0)

    features: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        if col in _STRING_FEATURE_COLUMNS or col not in series.index:
            continue
        value = series.get(col)
        if value is None or pd.isna(value):
            continue
        features[col] = _safe_float(value)

    return NSLKDDRow(
        source_ip=MONITORED_CLIENT_IP,
        dest_ip=_dest_ip_for(sld),
        source_port=None,          # genuinely absent, never a stand-in
        dest_port=DNS_PORT,
        protocol=DNS_PROTOCOL,
        total_bytes=total_bytes,
        duration_ms=0,             # not recorded by this dataset; see docstring
        raw_label=raw_label,
        category=category,
        kronus_label=kronus_label,
        difficulty=0,
        features=features,
        origin=DataOrigin.REAL,
    )


def _sort_by_capture_time(df: pd.DataFrame) -> pd.DataFrame:
    """Order rows by the dataset's own `timestamp` (unparseable rows last)."""
    if "timestamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.assign(_ts=ts).sort_values("_ts", kind="stable", na_position="last").drop(columns="_ts")


def load_dnsexf2021(
    path: str | Path,
    limit: int | None = None,
    sample_per_file: int | None = None,
    drop_ambiguous: bool = True,
    dedupe: bool = True,
) -> list[NSLKDDRow]:
    """Load CIC-Bell-DNS-EXF-2021 into the shared NSLKDDRow shape.

    `drop_ambiguous` removes every feature vector that carries more than one
    per-file class, and `dedupe` keeps one row per distinct vector. Both
    default on and both are documented at length in the module docstring —
    turning them off reproduces the raw corpus, label noise included.
    """
    path = Path(path)
    paths = _resolve_csv_paths(path)

    ambiguous = find_ambiguous_vectors(paths, sample_per_file) if drop_ambiguous else {}

    rows: list[NSLKDDRow] = []
    seen_keys: set[int] = set()
    unresolved: list[str] = []

    for csv_path in paths:
        df = _read_one_csv(csv_path, sample_per_file)
        if df is None or df.empty:
            continue
        file_class = _file_class(csv_path)
        if file_class is None and "Label" not in df.columns:
            unresolved.append(csv_path.name)
            continue

        df = _sort_by_capture_time(df)
        keys = _vector_key(df)
        raw_label = f"{csv_path.parent.name}/{csv_path.stem}"
        n_before = len(rows)

        for key, (_, series) in zip(keys, df.iterrows(), strict=True):
            resolved = _row_class(series, file_class)
            if resolved is None:
                continue
            if key in ambiguous:
                continue
            if dedupe:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            category, kronus_label = resolved
            rows.append(_row_to_nslkdd_row(series, category, kronus_label, raw_label))
            if limit is not None and limit > 0 and len(rows) >= limit:
                return rows
        if len(rows) == n_before:
            unresolved.append(csv_path.name)

    if unresolved:
        # Loud, never silent: a file contributing zero rows is either
        # unresolvable or entirely ambiguous, and both are worth knowing about.
        print(
            f"warning: {len(unresolved)} file(s) contributed no rows: "
            + ", ".join(unresolved),
            file=sys.stderr,
        )
    return rows


def load_dnsexf2021_dataframe(
    path: str | Path, sample_per_file: int | None = None
) -> pd.DataFrame:
    """Every raw row as a DataFrame, tagged with `category` and `vector_key`.

    Unmapped rows are KEPT with a NaN category so "this file contributed
    nothing" stays visible, and `vector_key` lets a caller recompute the
    ambiguity and de-duplication statistics for itself.
    """
    paths = _resolve_csv_paths(Path(path))
    frames = []
    for csv_path in paths:
        df = _read_one_csv(csv_path, sample_per_file)
        if df is None or df.empty:
            continue
        file_class = _file_class(csv_path)
        df = _sort_by_capture_time(df).copy()
        classes = [_row_class(series, file_class) for _, series in df.iterrows()]
        df["category"] = [c[0] if c else None for c in classes]
        df["vector_key"] = _vector_key(df)
        df["source_file"] = csv_path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ambiguity_report(path: str | Path, sample_per_file: int | None = None) -> dict:
    """The measured label-quality numbers the experiment metrics disclose.

    Everything here is computed from the data on hand rather than hard-coded,
    so the disclosure cannot drift away from the corpus it describes.
    """
    paths = _resolve_csv_paths(Path(path))
    df = load_dnsexf2021_dataframe(Path(path), sample_per_file)
    if df.empty:
        return {"rows": 0}

    ambiguous = find_ambiguous_vectors(paths, sample_per_file)
    n_rows = len(df)
    n_vectors = int(df["vector_key"].nunique())
    ambiguous_rows = int(df["vector_key"].isin(ambiguous.keys()).sum())
    by_category = df["category"].value_counts(dropna=False).to_dict()

    ceiling = 0.0
    mapped = df[df["category"].notna()]
    n_mapped = len(mapped)
    if n_mapped:
        per_key = (
            mapped.groupby("vector_key")["category"]
            .value_counts()
            .groupby(level=0)
            .max()
            .sum()
        )
        ceiling = float(per_key) / n_mapped

    amb_mask = df["vector_key"].isin(ambiguous.keys())
    survived = mapped[~amb_mask]
    rows_by_cat = {str(k): int(v) for k, v in by_category.items()}
    amb_by_cat = df[amb_mask]["category"].value_counts().to_dict()

    return {
        "rows": n_rows,
        "rows_with_known_category": n_mapped,
        "distinct_feature_vectors": n_vectors,
        "duplicate_row_fraction": round(1.0 - n_vectors / n_rows, 4) if n_rows else 0.0,
        "ambiguous_vectors": len(ambiguous),
        "rows_in_ambiguous_vectors": ambiguous_rows,
        "ambiguous_row_fraction": round(ambiguous_rows / n_rows, 4) if n_rows else 0.0,
        "deterministic_ceiling": round(ceiling, 4),
        "n_files": len(paths),
        "rows_per_category": rows_by_cat,
        "distinct_feature_vectors_by_category": {
            str(k): int(v) for k, v in mapped.groupby("category")["vector_key"].nunique().items()
        },
        # What is LEFT after the ambiguous-vector guard — the numbers that show
        # how little of the attack class the guard can keep.
        "distinct_vectors_after_guard_by_category": {
            str(k): int(v) for k, v in survived.groupby("category")["vector_key"].nunique().items()
        },
        "rows_after_guard_by_category": {
            str(k): int(v) for k, v in survived["category"].value_counts().items()
        },
        "ambiguous_row_fraction_by_category": {
            str(cat): round(int(amb_by_cat.get(cat, 0)) / int(total), 4)
            for cat, total in by_category.items()
            if total
        },
    }


def summarize(rows: list[NSLKDDRow]) -> dict[str, int]:
    """Category -> count, for reporting what actually loaded."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.category] = counts.get(row.category, 0) + 1
    return counts
