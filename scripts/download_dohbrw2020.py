#!/usr/bin/env python3
"""Fetches the CIRA-CIC-DoHBrw-2020 per-class CSVs into
data/external/dohbrw2020/, the external-validation dataset for
scripts/run_dohbrw2020_experiment.py.

CIRA-CIC-DoHBrw-2020 (Canadian Institute for Cybersecurity, UNB) is the
DoH / DoH-tunnel capture behind MontazeriShatoori et al., "Detection of DoH
Tunnels using Time-series Classification of Encrypted Traffic", IEEE
CyberSciTech 2020.

*** WHY THIS SCRIPT HAS A HEADER CHECK ***
The officially-published direct links for this dataset are dead, and they
fail in the worst possible way: every one of them still answers HTTP 200,
with Content-Type: text/html and content-length 108784, serving the UNB
"CIC | Datasets" web page. Verified, not assumed:

    http://205.174.165.80/CICDataset/DoHBrw-2020/Dataset/BenignDoH-NonDoH-CSVs.zip
    http://205.174.165.80/CICDataset/DoHBrw-2020/Dataset/MaliciousDoH-CSVs.zip
    http://205.174.165.80/CICDataset/DoHBrw-2020/Dataset/Total-CSVs.zip
    -> all three: status 200, Content-Type: text/html, 108784 bytes

A downloader that trusted the status code would save 108 KB of HTML as a
named .zip and report success. So `--url` refuses a body whose first line is
not a DoHBrw header, and every mode byte-checks against the known sizes.

MODES
1. --from-mirror: a public mirror (a HuggingFace dataset repo) serving the
   four per-class CSVs used here, byte-identical to the sizes recorded in
   SPLITS.
2. --url / KRONUS_DOHBRW2020_URL: a base URL carrying the same filenames, or
   a .zip of them.
3. --from-local PATH: a .zip, or a directory, you already downloaded.
4. No argument: prints what to fetch and where, then exits 0 -- so the
   runner's graceful skip never becomes a pipeline failure here.

DISK
    The four files total ~165 MB (dns2tcp alone is ~102 MB). They are only
    needed while the experiment runs; delete data/external/dohbrw2020/
    afterwards to reclaim the space.

End state: the four CSVs in data/external/dohbrw2020/, ready for

    python scripts/run_dohbrw2020_experiment.py
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
DEST_DIR = REPO_ROOT / "data" / "external" / "dohbrw2020"

# Filename -> expected byte size. The size is the integrity check: a mirror
# that re-splits or re-compresses the data changes the size and is rejected,
# rather than silently feeding the loader different rows.
SPLITS: dict[str, int] = {
    "Benign-DoH.csv": 11_444_070,
    "DNSCat2-DoH.csv": 22_450_979,
    "dns2tcp-DoH.csv": 102_462_664,
    "iodine-DoH.csv": 29_149_735,
}

MIRROR_BASE = "https://huggingface.co/datasets/c01dsnap/DoHTunnelAnalyzer/resolve/main"

MANUAL_INSTRUCTIONS = f"""\
DoHBrw2020 could not be fetched automatically (no --from-mirror, --url or
--from-local given).

To get it manually:
  1. The dataset is CIRA-CIC-DoHBrw-2020 from the Canadian Institute for
     Cybersecurity (UNB). The original direct links are dead; the live
     sources are the UNB dataset page and mirrors of the per-class CSVs.
     You want these four files:
       Benign-DoH.csv    (benign DoH)
       DNSCat2-DoH.csv   (dnscat2 tunnel)
       dns2tcp-DoH.csv   (dns2tcp tunnel)
       iodine-DoH.csv    (iodine tunnel)
  2. Put them in:
       {DEST_DIR}
     (or run this script again with --from-local pointing at the .zip/folder).

Alternatively, this script knows a mirror serving those exact bytes:
  python scripts/download_dohbrw2020.py --from-mirror

Then run:
  python scripts/run_dohbrw2020_experiment.py

The four files total ~165 MB and are only needed while the experiment runs;
delete {DEST_DIR} afterwards.
"""


def _header_ok(path: Path) -> bool:
    """True if the file's first line is a DoHBrw2020 header.

    This is the guard described in the module docstring: the dead official
    links return a 200 status with an HTML body, so a status code proves
    nothing. A real header always carries SourceIP, DestinationIP and
    Duration, which no HTML error page does.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return False
    return "SourceIP" in first and "DestinationIP" in first and "Duration" in first


def _fetch(url: str, dest: Path) -> None:
    """Stream `url` to `dest` (leaving no half-file behind on failure)."""
    req = urllib.request.Request(url, headers={"User-Agent": "kronus-dataset-fetch/1.0"})
    tmp = dest.parent / (dest.name + ".part")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
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
                "Either the source changed or this is not the DoHBrw2020 CSV; "
                "not trusting it.",
                file=sys.stderr,
            )
            target.unlink(missing_ok=True)
            return 1
        if not _header_ok(target):
            print(
                f"error: {name} does not start with a DoHBrw2020 header "
                "(SourceIP/DestinationIP/Duration). An HTML error page saved as "
                "a .csv is the exact failure this check exists to catch.",
                file=sys.stderr,
            )
            target.unlink(missing_ok=True)
            return 1
        print(f"  saved {name} ({got:,} bytes, size + header verified)")
    return 0


def _from_mirror(dest: Path) -> int:
    for name in SPLITS:
        url = f"{MIRROR_BASE}/{name}"
        print(f"downloading {url} ...")
        try:
            _fetch(url, dest / name)
        except Exception as exc:
            print(f"error: could not fetch {url}: {exc}", file=sys.stderr)
            return 1
    return _check_sizes(dest, list(SPLITS))


def _from_url(url: str, dest: Path) -> int:
    if url.endswith(".zip"):
        tmp_zip = dest / "_dohbrw2020_download.zip"
        print(f"downloading {url} ...")
        try:
            _fetch(url, tmp_zip)
        except Exception as exc:
            print(f"error: could not fetch {url}: {exc}", file=sys.stderr)
            return 1
        if not zipfile.is_zipfile(tmp_zip):
            size = tmp_zip.stat().st_size
            tmp_zip.unlink(missing_ok=True)
            print(
                f"error: {url} returned {size:,} bytes that are not a ZIP archive. "
                "The dead CIC links answer 200 with an HTML page — this is that.",
                file=sys.stderr,
            )
            return 1
        n = _extract_zip(tmp_zip, dest)
        tmp_zip.unlink(missing_ok=True)
        if n == 0:
            print("error: the .zip contained no .csv files", file=sys.stderr)
            return 1
        return 0

    base = url.rstrip("/")
    for name in SPLITS:
        src = f"{base}/{name}"
        print(f"downloading {src} ...")
        try:
            _fetch(src, dest / name)
        except Exception as exc:
            print(f"error: could not fetch {src}: {exc}", file=sys.stderr)
            return 1
    return _check_sizes(dest, list(SPLITS))


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
    present = sorted(dest.glob("*.csv"))
    print(f"\n{len(present)} CSV(s) in {dest}:")
    for p in present:
        note = ""
        if p.name in SPLITS:
            expect = SPLITS[p.name]
            if p.stat().st_size == expect:
                note = "  (matches expected size)"
            else:
                note = f"  (WARNING: {p.stat().st_size:,} bytes, expected {expect:,})"
        if not _header_ok(p):
            note += "  (WARNING: header does not look like a DoHBrw2020 CSV)"
        print(f"  {p.name}{note}")

    missing = [n for n in SPLITS if not (dest / n).exists()]
    if missing:
        print(
            "\nWARNING: the runner reads all four per-class files; still missing: "
            + ", ".join(missing),
            file=sys.stderr,
        )
    else:
        print("\nRun: python scripts/run_dohbrw2020_experiment.py")
        print(f"Afterwards, reclaim ~165 MB with: rm -rf {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-mirror",
        action="store_true",
        help="fetch from the public mirror serving the four per-class CSVs",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("KRONUS_DOHBRW2020_URL"),
        help="base URL serving the same filenames, or a .zip of them",
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
