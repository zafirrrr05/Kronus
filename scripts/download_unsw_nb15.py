#!/usr/bin/env python3
"""Fetches the UNSW-NB15 labelled-flow CSVs into data/external/unsw_nb15/,
the external-validation dataset for scripts/run_unsw_nb15_experiment.py.

UNSW-NB15 (Australian Centre for Cyber Security, UNSW Canberra; Moustafa &
Slay, MilCIS 2015) ships as a pre-split pair of labelled-flow CSVs:

    UNSW_NB15_training-set.csv   175,341 flows
    UNSW_NB15_testing-set.csv     82,332 flows

The canonical source (https://research.unsw.edu.au/projects/unsw-nb15-dataset)
serves them through a gated portal, so -- exactly like
scripts/download_cicids2017.py -- there is no single unauthenticated URL this
script can rely on forever. It therefore supports several modes:

1. --from-mirror: a verified public mirror (a HuggingFace dataset repo), whose
   two CSVs are byte-identical to the official release.

   *** THE MIRROR'S FILENAMES ARE INVERTED. ***
   Its `test.csv` IS the official *training* set and its `train.csv` IS the
   official *testing* set. This is confirmed, not assumed: the per-class row
   counts in each file sum to the published 175,341 / 82,332 split, and the
   official per-attack counts come out exactly right when they are combined
   (e.g. DoS 12,264 + 4,089 = 16,353; Reconnaissance 10,491 + 3,496 = 13,987).
   The mirror's own labels are merely "the big file" and "the small file".
   This script therefore maps each mirror file onto its OFFICIAL name by
   verified byte size (see SPLITS), so the split can never be silently
   swapped -- training on the test set would be a real correctness bug, since
   UNSW-NB15's official test set deliberately omits attack families that the
   training set contains.

2. --url / KRONUS_UNSW_NB15_URL: a base URL whose CSVs already carry the
   official UNSW_NB15_*.csv names, or a .zip of them.
3. --from-local PATH: a .zip, or a directory, you already downloaded.
4. No argument: prints exactly what to download and where to put it, then
   exits 0 -- so the experiment runner's graceful skip is never turned into a
   pipeline failure by a network this script can't reach.

Whichever mode runs, the end state is the same: the two CSVs above in
data/external/unsw_nb15/, ready for:

    python scripts/run_unsw_nb15_experiment.py
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
DEST_DIR = REPO_ROOT / "data" / "external" / "unsw_nb15"

# Official filename -> (expected byte size, the name the same bytes carry on the
# verified mirror). The sizes are the identity check that makes the inverted
# mirror naming safe to handle mechanically.
SPLITS: dict[str, tuple[int, str]] = {
    "UNSW_NB15_training-set.csv": (32_293_018, "test.csv"),
    "UNSW_NB15_testing-set.csv": (15_380_800, "train.csv"),
}

MIRROR_BASE = "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main"

MANUAL_INSTRUCTIONS = f"""\
UNSW-NB15 could not be fetched automatically (no --from-mirror, --url or
--from-local given).

To get it manually:
  1. Visit https://research.unsw.edu.au/projects/unsw-nb15-dataset and use the
     ACCS download links (the portal is gated; the data itself is free for
     research use). You want the two "pre-split" labelled-flow CSVs:
       UNSW_NB15_training-set.csv  (175,341 flows)
       UNSW_NB15_testing-set.csv   ( 82,332 flows)
  2. Put them in:
       {DEST_DIR}
     (or run this script again with --from-local pointing at the .zip/folder).

Alternatively, this script knows a verified mirror whose bytes are identical
to the official release:
  python scripts/download_unsw_nb15.py --from-mirror
(It corrects that mirror's inverted train/test filenames by file size.)

Then run:
  python scripts/run_unsw_nb15_experiment.py

The experiment runner aborts cleanly if the data is absent, so this is not a
hard failure -- it just means the UNSW-NB15 experiment can't run until the
CSVs are in place.
"""


def _header_ok(path: Path) -> bool:
    """True if the file's first line is a UNSW-NB15 pre-split header.

    Guards against saving an error page (or a truncated stub) as if it were
    data: the real header always carries both `attack_cat` and `label`, which
    no HTML error body does.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return False
    return "attack_cat" in first and "label" in first


def _fetch(url: str, dest: Path) -> None:
    """Stream `url` to `dest` (leaving no half-file behind on failure)."""
    req = urllib.request.Request(url, headers={"User-Agent": "kronus-dataset-fetch/1.0"})
    tmp = dest.parent / (dest.name + ".part")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)


def _extract_zip(zip_path: Path, dest: Path) -> int:
    """Extract every *.csv from `zip_path` flat into `dest`."""
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith(".csv"):
                target = dest / Path(member).name
                with zf.open(member) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                print(f"  extracted {target.name}")
                n += 1
    return n


def _from_mirror(dest: Path) -> int:
    """Fetch both splits from MIRROR_BASE, restoring the official names.

    The mirror's `test.csv` is the official training set and vice versa, so
    the mapping comes from SPLITS rather than the remote filename. Each file
    is size-checked afterwards: a mismatch means the mirror changed under us
    and is a hard error, never a silent pass.
    """
    for official_name, (expect_size, mirror_name) in SPLITS.items():
        url = f"{MIRROR_BASE}/{mirror_name}"
        target = dest / official_name
        print(f"downloading {url}  ->  {official_name} ...")
        try:
            _fetch(url, target)
        except Exception as exc:
            print(f"error: could not fetch {url}: {exc}", file=sys.stderr)
            return 1
        got = target.stat().st_size
        if got != expect_size:
            print(
                f"error: {official_name} is {got:,} bytes, expected {expect_size:,}. "
                "The mirror no longer matches the official release; not trusting it.",
                file=sys.stderr,
            )
            target.unlink(missing_ok=True)
            return 1
        print(f"  saved {official_name} ({got:,} bytes, verified)")


def _from_url(url: str, dest: Path) -> int:
    """Fetch from a user-supplied base URL (mirror-named files) or a .zip."""
    if url.endswith(".zip"):
        tmp_zip = dest / "_unsw_nb15_download.zip"
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
    for official_name in SPLITS:
        target = dest / official_name
        src = f"{base}/{official_name}"
        print(f"downloading {src} ...")
        try:
            _fetch(src, target)
        except Exception as exc:
            print(f"error: could not fetch {src}: {exc}", file=sys.stderr)
            return 1
        print(f"  saved {official_name} ({target.stat().st_size:,} bytes)")
    return 0


def _from_local(src: Path, dest: Path) -> int:
    if not src.exists():
        print(f"error: --from-local path {src} does not exist", file=sys.stderr)
        return 1
    if src.is_dir():
        n = 0
        for csv in sorted(src.glob("**/*.csv")):
            shutil.copy2(csv, dest / csv.name)
            print(f"  copied {csv.name}")
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
    """Print what landed, with a size check against the official release."""
    present = sorted(dest.glob("*.csv"))
    print(f"\n{len(present)} CSV(s) in {dest}:")
    for p in present:
        note = ""
        if p.name in SPLITS:
            expect = SPLITS[p.name][0]
            if p.stat().st_size == expect:
                note = "  (matches official size)"
            else:
                note = f"  (WARNING: {p.stat().st_size:,} bytes, official is {expect:,})"
        if not _header_ok(p):
            note += "  (WARNING: header does not look like a UNSW-NB15 pre-split CSV)"
        print(f"  {p.name}{note}")

    missing = [n for n in SPLITS if not (dest / n).exists()]
    if missing:
        print(
            "\nWARNING: the experiment runner reads both splits; still missing: "
            + ", ".join(missing),
            file=sys.stderr,
        )
    else:
        print("\nRun: python scripts/run_unsw_nb15_experiment.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-mirror",
        action="store_true",
        help="fetch from the verified public mirror (corrects its inverted train/test names)",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("KRONUS_UNSW_NB15_URL"),
        help="base URL serving official-named CSVs, or a .zip of them",
    )
    parser.add_argument(
        "--from-local",
        type=Path,
        default=None,
        help="path to an already-downloaded .zip or directory of CSVs",
    )
    args = parser.parse_args()

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(DEST_DIR.glob("*.csv"))
    if existing:
        print(f"already have {len(existing)} CSV(s) in {DEST_DIR}:")
        for p in existing:
            print(f"  {p.name}")
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
