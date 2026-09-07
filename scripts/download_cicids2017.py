#!/usr/bin/env python3
"""Fetches the CIC-IDS2017 labelled-flow CSVs into data/external/cicids2017/,
the external-validation dataset for scripts/validate_cicids2017.py.

Unlike NSL-KDD (scripts/download_data.py, a small public GitHub mirror),
CIC-IDS2017 is ~500 MB of CSVs and its canonical source (the University of
New Brunswick, https://www.unb.ca/cic/datasets/ids-2017.html) is behind a
short registration form, so there is no single unauthenticated URL this
script can rely on forever. It therefore supports three modes, in order of
preference:

1. --url / KRONUS_CICIDS2017_URL: a direct URL to a .zip of the CSVs (e.g.
   an institutional mirror you have access to). Downloaded and extracted.
2. --from-local PATH: a .zip or directory you already downloaded manually
   from UNB. Copied/extracted into place.
3. No argument: prints exactly what to download and where to put it, then
   exits 0 (so it never fails a pipeline that legitimately can't reach the
   data — the validation harness skips gracefully when the data is absent).

Whichever mode runs, the end state is the same: the eight
*.pcap_ISCX.csv files (or the MachineLearningCVE equivalents) sitting in
data/external/cicids2017/, ready for `python scripts/validate_cicids2017.py`.
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
DEST_DIR = REPO_ROOT / "data" / "external" / "cicids2017"

MANUAL_INSTRUCTIONS = f"""\
CIC-IDS2017 could not be fetched automatically (no --url or --from-local given).

To get it manually:
  1. Visit https://www.unb.ca/cic/datasets/ids-2017.html and complete the
     short download form (UNB requires it; the data itself is free for
     research use).
  2. Download the "GeneratedLabelledFlows" (a.k.a. TrafficLabelling) archive
     — the CSVs, not the raw PCAPs. It expands to eight *.pcap_ISCX.csv files.
  3. Put the CSVs in:
       {DEST_DIR}
     (either extract them there directly, or run this script again with
      --from-local pointing at the downloaded .zip or folder.)

Then run:
  python scripts/validate_cicids2017.py

The validation harness skips gracefully if the data is absent, so this is
not a hard failure — it just means the external-dataset SLO can't be
measured until the CSVs are in place.
"""


def _extract_zip(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith(".csv"):
                # Flatten: drop any internal directory structure, land all
                # CSVs directly in dest/.
                target = dest / Path(member).name
                with zf.open(member) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                print(f"  extracted {target.name}")


def _from_local(src: Path, dest: Path) -> int:
    if not src.exists():
        print(f"error: --from-local path {src} does not exist", file=sys.stderr)
        return 1
    if src.is_dir():
        n = 0
        for csv in src.glob("**/*.csv"):
            shutil.copy2(csv, dest / csv.name)
            print(f"  copied {csv.name}")
            n += 1
        if n == 0:
            print(f"error: no .csv files found under {src}", file=sys.stderr)
            return 1
        return 0
    if src.suffix == ".zip":
        _extract_zip(src, dest)
        return 0
    print(f"error: --from-local expects a .zip or a directory, got {src}", file=sys.stderr)
    return 1


def _from_url(url: str, dest: Path) -> int:
    tmp_zip = dest / "_cicids2017_download.zip"
    print(f"downloading {url} ...")
    urllib.request.urlretrieve(url, tmp_zip)
    print(f"  saved {tmp_zip} ({tmp_zip.stat().st_size // (1024 * 1024)} MB); extracting ...")
    _extract_zip(tmp_zip, dest)
    tmp_zip.unlink(missing_ok=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("KRONUS_CICIDS2017_URL"),
                        help="direct URL to a .zip of the CIC-IDS2017 CSVs")
    parser.add_argument("--from-local", type=Path, default=None,
                        help="path to an already-downloaded .zip or directory of CSVs")
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
    elif args.url:
        rc = _from_url(args.url, DEST_DIR)
    else:
        print(MANUAL_INSTRUCTIONS)
        return 0

    if rc == 0:
        n = len(list(DEST_DIR.glob("*.csv")))
        print(f"\nDone — {n} CSV(s) in {DEST_DIR}. Run `python scripts/validate_cicids2017.py`.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
