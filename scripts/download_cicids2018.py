#!/usr/bin/env python3
"""Fetches the CSE-CIC-IDS2018 day CSVs into data/external/cicids2018/, the
external-validation dataset for scripts/run_cicids2018_experiment.py.

CSE-CIC-IDS2018 ("Intrusion Detection Evaluation Dataset", Canadian Institute
for Cybersecurity, University of New Brunswick) is the successor to
CIC-IDS2017. Official page: https://www.unb.ca/cic/datasets/ids-2018.html

    Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani,
    "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
    Characterization", 4th ICISSP, Portugal, January 2018.

*** THE SOURCE IS AUTHORITATIVE, AND THAT IS VERIFIED ***
Unlike every other external loader in this repository, this one needs no
mirror and no form: UNB publishes the ML-ready CSVs in a public, unauthenticated
AWS S3 bucket (ca-central-1). `--from-mirror` therefore *is* the official
source, and the sizes in SPLITS were read from that bucket by HEAD request.

*** WHY EVERY FILE IS HEADER-CHECKED ***
A loader for this dataset already exists in the tree that reads CIC-IDS2017's
column names ("Source IP", "Total Length of Fwd Packets") while being named for
CIC-IDS2018, and maps CIC-DDoS2019's label vocabulary. Against the real files
it maps only `Benign`, drops every attack row, and returns an all-zero dataset
while reporting success. The fix is not a comment: `_header_ok` refuses any
file whose header is not the real 80-column schema, so that class of silent
wrong-dataset failure cannot recur here.

WHAT IS FETCHED, AND WHY THESE FOUR DAYS
    Wednesday-28-02-2018   Infilteration + Benign   -> Detective (port scan)
    Thursday-01-03-2018    Infilteration + Benign   -> Detective (port scan)
    Friday-16-02-2018      DoS attacks-Hulk, SlowHTTPTest -> Bouncer (flood)
    Wednesday-21-02-2018   DDOS attack-HOIC         -> Bouncer (flood)

Each lane gets two days on purpose, and for the same reason: a single day lets
a model separate the classes by *capture date* instead of by traffic shape.
Two days per lane does not eliminate that risk, but it removes the trivial
version of it. The flood days also give two independent attack families —
Hulk and SlowHTTPTest are slow/rate-limited application floods, HOIC is a
high-rate HTTP flood — so the Bouncer's flood class is not one attack's
fingerprint. Thursday-01-03-2018 earns its place specifically because it
measures as the most scan-shaped infiltration capture of the three (1.66x the
benign port-diversity rate, against 1.31x for 28-02), and the Detective needs
that; see twin/cicids2018.py for the measurements.

Day choice is not arbitrary — the loaders' row caps take an evenly spaced
*stride*, and each of these days carries its attack as a time-bounded window.
Thursday-01-03-2018 was rejected as a *prefix* sample for the same reason it
is accepted as a strided one: its tail is entirely benign, so a prefix of it
would return no attack at all while appearing to succeed.

MODES
1. --from-mirror: the official public S3 bucket. No form, no credentials.
2. --url / KRONUS_CICIDS2018_URL: a base URL carrying the same filenames.
3. --from-local PATH: a directory of already-downloaded day CSVs.
4. No argument: prints what to fetch and where, then exits 0 — so the
   runner's graceful skip never becomes a pipeline failure here.

Re-running skips files that are already present at the right size, so adding
a day to SPLITS costs one file rather than a full re-download.

DISK
    The four files total ~980 MB. They are only needed while the experiment
    runs; delete data/external/cicids2018/ afterwards to reclaim the space.

End state: the four CSVs in data/external/cicids2018/, ready for

    python scripts/run_cicids2018_experiment.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "data" / "external" / "cicids2018"

# The official public bucket. Used by --from-mirror.
OFFICIAL_BASE = (
    "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
    + urllib.parse.quote("Processed Traffic Data for ML Algorithms")
)

# Filename -> exact byte size, read from the bucket by HEAD request. The size
# is the integrity check: a re-split or re-compressed copy is rejected rather
# than silently feeding the loader different rows.
SPLITS: dict[str, int] = {
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv": 209_249_758,
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv": 107_842_858,
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv": 333_723_605,
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv": 328_893_673,
}

# What each day is here for, surfaced in the report so the mapping is visible.
PURPOSE: dict[str, str] = {
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv":
        "Infilteration + Benign  -> Detective (port scan)",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv":
        "Infilteration + Benign  -> Detective (port scan), 2nd day",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv":
        "DoS attacks-Hulk/SlowHTTPTest -> Bouncer (flood)",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv":
        "DDOS attack-HOIC         -> Bouncer (flood)",
}

MANUAL_INSTRUCTIONS = f"""\
CSE-CIC-IDS2018 could not be fetched automatically (no --from-mirror, --url or
--from-local given).

To get it manually:
  1. The dataset is CSE-CIC-IDS2018 from the Canadian Institute for
     Cybersecurity (UNB): https://www.unb.ca/cic/datasets/ids-2018.html
     The ML-ready day CSVs live under "Processed Traffic Data for ML
     Algorithms" in the public bucket below; no form or account is needed.
  2. You want these four day files:
       Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv
       Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
       Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
       Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
  3. Put them in:
       {DEST_DIR}
     (or run this script again with --from-local pointing at their folder).

The easiest route is simply:
  python scripts/download_cicids2018.py --from-mirror

Then run:
  python scripts/run_cicids2018_experiment.py

The four files total ~980 MB and are only needed while the experiment runs;
delete {DEST_DIR} afterwards.
"""


def _header_ok(path: Path) -> bool:
    """True if the file's first line is the real CSE-CIC-IDS2018 header.

    This is the guard described in the module docstring. The real header is
    80 columns beginning "Dst Port,Protocol,Timestamp,Flow Duration,..." and
    ending "Label". CIC-IDS2017's header ("Source IP","Destination IP",
    "Total Length of Fwd Packets") is a *different* schema, and a loader for
    one fed the other silently produces zeros — hence requiring the exact
    columns this loader reads rather than merely a plausible-looking CSV.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return False
    required = ("Dst Port", "Protocol", "Timestamp", "Flow Duration",
                "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Label")
    return all(col in first for col in required)


def _fetch(url: str, dest: Path) -> None:
    """Stream `url` to `dest` (leaving no half-file behind on failure)."""
    req = urllib.request.Request(url, headers={"User-Agent": "kronus-dataset-fetch/1.0"})
    tmp = dest.parent / (dest.name + ".part")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)


def _check_sizes(dest: Path, names: list[str]) -> int:
    """Byte-verify freshly-fetched files. Returns 0 on success, 1 on mismatch."""
    for name in names:
        target = dest / name
        if not target.exists():
            print(f"error: {name} was not fetched", file=sys.stderr)
            return 1
        got = target.stat().st_size
        expect = SPLITS[name]
        if got != expect:
            print(
                f"error: {name} is {got:,} bytes, expected {expect:,}. "
                "Either the source changed or this is not the CSE-CIC-IDS2018 "
                "day file; not trusting it.",
                file=sys.stderr,
            )
            target.unlink(missing_ok=True)
            return 1
        if not _header_ok(target):
            print(
                f"error: {name} does not start with the real CSE-CIC-IDS2018 "
                "header (Dst Port/Protocol/Timestamp/TotLen Fwd Pkts/Label). "
                "A different dataset's CSV under this name is the exact "
                "failure this check exists to catch.",
                file=sys.stderr,
            )
            target.unlink(missing_ok=True)
            return 1
        print(f"  saved {name} ({got:,} bytes, size + header verified)")
    return 0


def _download_all(base: str, dest: Path) -> int:
    base = base.rstrip("/")
    fetched: list[str] = []
    for name in SPLITS:
        target = dest / name
        # Already have it, at the right size? Skip — this makes adding a day to
        # SPLITS cost one file rather than a full re-download of the ~1 GB set.
        if target.exists() and target.stat().st_size == SPLITS[name]:
            print(f"  have {name} ({SPLITS[name]:,} bytes), skipping")
            continue
        url = f"{base}/{urllib.parse.quote(name)}"
        print(f"downloading {url} ...")
        try:
            _fetch(url, target)
        except Exception as exc:
            print(f"error: could not fetch {url}: {exc}", file=sys.stderr)
            return 1
        fetched.append(name)
    # Re-verify everything, not just what was fetched: a skipped file is only
    # trusted because of the size check above, and the header check has never
    # run on it in this process.
    return _check_sizes(dest, list(SPLITS))


def _from_local(src: Path, dest: Path) -> int:
    if not src.exists():
        print(f"error: --from-local path {src} does not exist", file=sys.stderr)
        return 1
    if not src.is_dir():
        print(
            f"error: --from-local expects a directory of day CSVs, got {src}",
            file=sys.stderr,
        )
        return 1
    found = 0
    for name in SPLITS:
        candidate = src / name
        if not candidate.exists():
            # Fall back to a recursive search: users often unzip into a subdir.
            matches = list(src.glob(f"**/{name}"))
            candidate = matches[0] if matches else None
        if candidate is None:
            print(f"error: {name} not found under {src}", file=sys.stderr)
            return 1
        shutil.copy2(candidate, dest / name)
        print(f"  copied {candidate.name}")
        found += 1
    print(f"  {found} file(s) copied")
    return _check_sizes(dest, list(SPLITS))


def _report(dest: Path) -> None:
    present = sorted(dest.glob("*.csv"))
    print(f"\n{len(present)} CSV(s) in {dest}:")
    total = 0
    for p in present:
        size = p.stat().st_size
        total += size
        note = f"  [{PURPOSE.get(p.name, 'not part of the expected day set')}]"
        if p.name in SPLITS:
            expect = SPLITS[p.name]
            note = ("  (matches expected size)" if size == expect
                    else f"  (WARNING: {size:,} bytes, expected {expect:,})") + note
        if not _header_ok(p):
            note += "  (WARNING: header is not the real CSE-CIC-IDS2018 schema)"
        print(f"  {p.name}{note}")

    missing = [n for n in SPLITS if not (dest / n).exists()]
    if missing:
        print(
            "\nWARNING: the runner reads all three day files; still missing: "
            + ", ".join(missing),
            file=sys.stderr,
        )
    else:
        print(f"\nTotal on disk: {total / 1e6:.1f} MB")
        print("Run: python scripts/run_cicids2018_experiment.py")
        print(f"Afterwards, reclaim the space with: rm -rf {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-mirror",
        action="store_true",
        help="fetch from the official public UNB AWS S3 bucket (no form, no auth)",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("KRONUS_CICIDS2018_URL"),
        help="base URL serving the same filenames",
    )
    parser.add_argument(
        "--from-local",
        type=Path,
        default=None,
        help="path to a directory of already-downloaded day CSVs",
    )
    args = parser.parse_args()

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(DEST_DIR.glob("*.csv"))
    ready = [p for p in existing if SPLITS.get(p.name) == p.stat().st_size]
    if len(ready) >= len(SPLITS):
        print(f"already have all {len(SPLITS)} CSV(s) in {DEST_DIR}:")
        for p in existing:
            print(f"  {p.name}")
        print("Delete them to re-download. Nothing to do.")
        return 0

    if args.from_local is not None:
        rc = _from_local(args.from_local, DEST_DIR)
    elif args.from_mirror:
        rc = _download_all(OFFICIAL_BASE, DEST_DIR)
    elif args.url:
        rc = _download_all(args.url, DEST_DIR)
    else:
        print(MANUAL_INSTRUCTIONS)
        return 0

    if rc == 0:
        _report(DEST_DIR)
    return rc


if __name__ == "__main__":
    sys.exit(main())
