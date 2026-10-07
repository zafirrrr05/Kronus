#!/usr/bin/env python3
"""Fetches the CIC-Bell-DNS-EXF-2021 per-class CSVs into
data/external/dnsexf2021/, the external-validation dataset for
scripts/run_dnsexf2021_experiment.py.

CIC-Bell-DNS-EXF-2021 (Canadian Institute for Cybersecurity, UNB; Mahdavifar &
Ghorbani) ships as sixteen per-class CSVs under three directories:

    benign_labeled/        4 files   benign DNS
    heavy_attack_labeled/  6 files   heavy exfiltration (audio, compressed,
                                     exe, image, text, video)
    light_attack_labeled/  6 files   light exfiltration

THE DIRECTORY IS THE LABEL. This script therefore preserves that structure
rather than flattening it: twin/dnsexf2021.py resolves each file's class from
its parent directory, and a flattened copy loses that. (A flattened copy still
loads, via the filename/label-column fallback — but the tree is the real
provenance, so it is what gets written.)

The canonical UNB page publishes these behind a request form. This script
supports several modes:

1. --from-mirror: the public GitHub repository that redistributes the released
   per-class CSVs (Pinkal-Kumar/Benign-Attacked-Classification). Every file is
   byte-size-verified against the origin listing before it is trusted.
2. --url / KRONUS_DNSEXF2021_URL: a base URL serving the same
   <dir>/<file>.csv paths, or a .zip of the tree.
3. --from-local PATH: a directory tree, or a .zip, already downloaded.
4. No argument: prints exactly what to download and where to put it, then
   exits 0 — so the experiment runner's graceful skip is never turned into a
   pipeline failure by a network this script can't reach.

Whichever mode runs, the end state is the same: sixteen CSVs under three
per-class directories in data/external/dnsexf2021/, ready for:

    python scripts/run_dnsexf2021_experiment.py

A NOTE ON THIS DATASET'S LABELS: before running the experiment, read the
module docstring of twin/dnsexf2021.py. The label is a capture-level
annotation and is not recoverable from the features (measured ceiling 0.8210),
and the dataset carries no IPs, ports, bytes or durations. 99.95% of its
exfiltration rows share a feature vector with a benign row, leaving the attack
class with 32 distinct signatures once those are removed — so the experiment
reports that no valid detection metric can be produced rather than quoting one.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "data" / "external" / "dnsexf2021"

# <dir>/<file>.csv -> the byte size published for it. The size is the identity
# check: a mirror that reshuffles, truncates or re-labels these files changes
# their length, and that must be a hard error rather than a silent pass.
SPLITS: dict[str, int] = {
    "benign_labeled/stateless_features-benign_heavy_1.csv": 6_006_488,
    "benign_labeled/stateless_features-benign_heavy_2.csv": 4_794_767,
    "benign_labeled/stateless_features-benign_heavy_3.csv": 6_951_358,
    "benign_labeled/stateless_features-light_benign.csv": 5_863_210,
    "heavy_attack_labeled/stateless_features-heavy_audio.csv": 3_778_187,
    "heavy_attack_labeled/stateless_features-heavy_compressed.csv": 3_774_365,
    "heavy_attack_labeled/stateless_features-heavy_exe.csv": 3_657_520,
    "heavy_attack_labeled/stateless_features-heavy_image.csv": 3_841_770,
    "heavy_attack_labeled/stateless_features-heavy_text.csv": 7_506_604,
    "heavy_attack_labeled/stateless_features-heavy_video.csv": 4_015_010,
    "light_attack_labeled/stateless_features-light_audio.csv": 1_861_252,
    "light_attack_labeled/stateless_features-light_compressed.csv": 1_081_568,
    "light_attack_labeled/stateless_features-light_exe.csv": 681_348,
    "light_attack_labeled/stateless_features-light_image.csv": 55_748,
    "light_attack_labeled/stateless_features-light_text.csv": 367_492,
    "light_attack_labeled/stateless_features-light_video.csv": 461_847,
}

MIRROR_BASE = (
    "https://raw.githubusercontent.com/Pinkal-Kumar/"
    "Benign-Attacked-Classification/main/data/labeled"
)

MANUAL_INSTRUCTIONS = f"""\
CIC-Bell-DNS-EXF-2021 could not be fetched automatically (no --from-mirror,
--url or --from-local given).

To get it manually:
  1. Request the dataset from the Canadian Institute for Cybersecurity:
       https://www.unb.ca/cic/datasets/dns-exf-2021.html
     (the page serves it behind a short request form; the data is free for
     research use).
  2. Unpack it so that the three per-class directories hold their CSVs:
       {DEST_DIR}/benign_labeled/
       {DEST_DIR}/heavy_attack_labeled/
       {DEST_DIR}/light_attack_labeled/
     (or run this script again with --from-local pointing at the tree/.zip).

Alternatively, this script knows a public mirror that redistributes the
released per-class CSVs, each one byte-size-verified against the origin:
  python scripts/download_dnsexf2021.py --from-mirror

Then run:
  python scripts/run_dnsexf2021_experiment.py

The experiment runner aborts cleanly if the data is absent, so this is not a
hard failure -- it just means the DNS-EXF2021 experiment can't run until the
CSVs are in place.
"""


def _header_ok(path: Path) -> bool:
    """True if the file's first line is a DNS-EXF-2021 stateless header.

    Guards against saving an error page (or a truncated stub) as if it were
    data: the real header always carries both `sld` and `subdomain`, which no
    HTML error body does.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return False
    return "sld" in first and "subdomain" in first


def _fetch(url: str, dest: Path) -> None:
    """Stream `url` to `dest` (leaving no half-file behind on failure)."""
    req = urllib.request.Request(url, headers={"User-Agent": "kronus-dataset-fetch/1.0"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (dest.name + ".part")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)


def _extract_zip(zip_path: Path, dest: Path) -> int:
    """Extract every *.csv from `zip_path`, preserving its directory part."""
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.endswith(".csv"):
                continue
            parts = Path(member).parts
            # Keep the last two components (<class>_labeled/<file>.csv) when the
            # archive is nested; fall back to the bare filename otherwise.
            rel = Path(*parts[-2:]) if len(parts) >= 2 else Path(parts[-1])
            target = dest / rel
            if target.parent == dest and rel.parent == Path("."):
                # A bare filename carries no class directory; that is fine, the
                # loader's filename fallback handles it.
                pass
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            print(f"  extracted {rel}")
            n += 1
    return n


def _from_mirror(dest: Path) -> int:
    for rel, expect_size in SPLITS.items():
        url = f"{MIRROR_BASE}/{rel}"
        target = dest / rel
        print(f"downloading {rel} ...")
        try:
            _fetch(url, target)
        except Exception as exc:
            print(f"error: could not fetch {url}: {exc}", file=sys.stderr)
            return 1
        got = target.stat().st_size
        if got != expect_size:
            print(
                f"error: {rel} is {got:,} bytes, expected {expect_size:,}. "
                "The mirror no longer matches the released files; not trusting it.",
                file=sys.stderr,
            )
            target.unlink(missing_ok=True)
            return 1
        print(f"  saved {rel} ({got:,} bytes, verified)")
    return 0


def _from_url(url: str, dest: Path) -> int:
    if url.endswith(".zip"):
        tmp_zip = dest / "_dnsexf2021_download.zip"
        print(f"downloading {url} ...")
        try:
            _fetch(url, tmp_zip)
        except Exception as exc:
            print(f"error: could not fetch {url}: {exc}", file=sys.stderr)
            return 1
        n = _extract_zip(tmp_zip, dest)
        tmp_zip.unlink(missing_ok=True)
        if n == 0:
            print("error: the .zip contained no .csv files", file=sys.stderr)
            return 1
        return 0

    base = url.rstrip("/")
    for rel in SPLITS:
        target = dest / rel
        src = f"{base}/{rel}"
        print(f"downloading {src} ...")
        try:
            _fetch(src, target)
        except Exception as exc:
            print(f"error: could not fetch {src}: {exc}", file=sys.stderr)
            return 1
        print(f"  saved {rel} ({target.stat().st_size:,} bytes)")
    return 0


def _from_local(src: Path, dest: Path) -> int:
    if not src.exists():
        print(f"error: --from-local path {src} does not exist", file=sys.stderr)
        return 1
    if src.is_dir():
        n = 0
        for csv in sorted(src.glob("**/*.csv")):
            rel = csv.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv, target)
            print(f"  copied {rel}")
            n += 1
        if n == 0:
            print(f"error: no .csv files found under {src}", file=sys.stderr)
            return 1
        return 0
    if src.suffix == ".zip":
        if _extract_zip(src, dest) == 0:
            print(f"error: no .csv files found in {src}", file=sys.stderr)
            return 1
        return 0
    print(f"error: --from-local expects a .zip or a directory, got {src}", file=sys.stderr)
    return 1


def _report(dest: Path) -> None:
    present = sorted(dest.glob("**/*.csv"))
    print(f"\n{len(present)} CSV(s) under {dest}:")
    for p in present:
        rel = p.relative_to(dest).as_posix()
        note = ""
        if rel in SPLITS:
            expect = SPLITS[rel]
            if p.stat().st_size == expect:
                note = "  (matches released size)"
            else:
                note = f"  (WARNING: {p.stat().st_size:,} bytes, released is {expect:,})"
        if not _header_ok(p):
            note += "  (WARNING: header does not look like a DNS-EXF-2021 stateless CSV)"
        print(f"  {rel}{note}")

    missing = [n for n in SPLITS if not (dest / n).exists()]
    if missing:
        print(
            f"\nWARNING: {len(missing)} of {len(SPLITS)} expected CSVs are still missing: "
            + ", ".join(missing[:4])
            + (" ..." if len(missing) > 4 else ""),
            file=sys.stderr,
        )
    else:
        print("\nRun: python scripts/run_dnsexf2021_experiment.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-mirror",
        action="store_true",
        help="fetch from the public mirror that redistributes the released per-class CSVs",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("KRONUS_DNSEXF2021_URL"),
        help="base URL serving <dir>/<file>.csv paths, or a .zip of the tree",
    )
    parser.add_argument(
        "--from-local",
        type=Path,
        default=None,
        help="path to an already-downloaded directory tree or .zip",
    )
    args = parser.parse_args()

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(DEST_DIR.glob("**/*.csv"))
    if len(existing) >= len(SPLITS):
        print(f"already have {len(existing)} CSV(s) in {DEST_DIR}:")
        for p in existing:
            print(f"  {p.relative_to(DEST_DIR)}")
        print("Delete them to re-download. Nothing to do.")
        return 0

    if args.from_local is not None:
        rc = _from_local(args.from_local, DEST_DIR)
    elif args.from_mirror:
        rc = _from_mirror(DEST_DIR)
    elif args.url:
        rc = _from_url(args.url, DEST_DIR)
    else:
        print(MANUAL_INSTRUCTIONS)
        return 0

    if rc == 0:
        _report(DEST_DIR)
    return rc


if __name__ == "__main__":
    sys.exit(main())
