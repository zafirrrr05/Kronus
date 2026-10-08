#!/usr/bin/env python3
"""Fetches the CIC IoT 2023 packet captures into data/external/ciciot2023/,
the external-validation dataset for scripts/run_ciciot2023_experiment.py.

CIC IoT 2023 ("A real-time dataset and benchmark for large-scale attacks in
IoT environment", Canadian Institute for Cybersecurity, UNB) is captured on a
105-device IoT testbed and carries 33 attacks in seven families plus benign
traffic. Official page:
https://www.unb.ca/cic/datasets/iotdataset-2023.html

    Neto, E., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., Ghorbani, A.A.
    "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in
    IoT environment", Sensors, 2023. doi:10.3390/s23135941

*** WHY THIS DOWNLOADS PCAP AND NOT THE PUBLISHED CSVs ***
This is the one dataset here whose *ML-ready CSVs cannot honestly train either
KRONUS lane*, and that is a measured fact about the release, not a preference.

Both CSV variants under CSV/ ship the same 39 columns — Header_Length, Rate,
IAT, Tot sum, Std, Variance, flag counts, protocol indicators — and every one
of them is an aggregate statistic of a flow. Read from the real bytes:

    CSV/CSV/<Family>/<Family>.pcap.csv      39 columns, NO label column
    CSV/MERGED_CSV/Merged01.csv             40 columns, label in `Label`

Neither carries a source IP, a destination IP, a port, or a timestamp. The
Bouncer's feature contract (libs/... FEATURE_NAMES) is event_rate, byte_rate,
dest_port_entropy, unique_dest_count, avg_duration_ms and same_dest_ratio.
Four of those six — everything port- or clock-derived — have nothing real
behind them in those files, so training on them would mean *inventing* the
features and then reporting the resulting score as a detection result. This
repository draws that line explicitly next door, in twin/unsw_nb15.py: hosts
are reconstructed only where they are "grounded in the dataset's own precomputed
connection-rate features rather than invented ones". No such column exists
here. Experiment E (twin/dnsexf2021.py) is the earlier instance of the same
wall and its answer was `verdict: NO_VALID_DETECTION_METRIC`.

The packet captures are the branch that carries real identity and a real
clock — real IPs, real ports, real inter-arrival times — so a flow extractor
over them yields the event shape the lanes actually consume, with each flow's
label taken from the capture's own family folder.
`twin/ciciot2023.py` does that extraction and records exactly what is the
publisher's measurement and what is ours.

*** THIS SOURCE IS FORM-GATED, AND THAT IS NOT WORKED AROUND ***
Like CIC-DDoS2019 (scripts/download_cicddos2019.py), and unlike the
CSE-CIC-IDS2018 S3 bucket next door, these files are served from cicresearch.ca
behind a registration form: browse.php and download.php return 403 without the
session the form issues. This script submits the form for you and downloads
with the cookie it returns. The registration values are NOT hardcoded — this
repository is public and a maintainer's name and email do not belong in it.
Supply your own with the flags below (or the matching environment variables).

*** WHAT IS FETCHED, AND WHY IT IS BOUNDED ***
PCAP is far larger than the CSVs, and the point of this experiment is a bounded
sample, not a corpus. `--families` names the families to take and
`--max-files-per-family` caps each one, so the fetch is planned rather than
unbounded. The default set is the one that lets BOTH lanes train:

    benign        -> normal      (Bouncer negative, Detective BENIGN)
    DDoS-*/DoS-*  -> dos/flood   (Bouncer positive)
    Mirai-*       -> dos/flood   (Bouncer positive — volumetric, distinct shape)
    Recon-*       -> probe       (Detective PORT_SCAN positive)

Every other family (Web-based, Spoofing, Brute Force) is left out: none of them
is volumetric and none is a scan, so neither lane has a contract for them.

MODES
1. --list: browse the dataset tree and print what is there. Makes no download
   and needs no registration beyond the session, so the real folder names are
   read rather than assumed.
2. --register (with --first-name/--last-name/--email/--institution/
   --job-title/--country, or the KRONUS_CIC_* environment variables): submit
   the form and download the selected families.
3. --from-local PATH: a directory already holding the captures.
4. No argument: prints what to fetch and where, then exits 0 — so the runner's
   graceful skip never becomes a pipeline failure here.

DISK
    Captures are large. Budget several GB for a bounded sample and delete
    data/external/ciciot2023/ afterwards to reclaim the space:

        rm -rf data/external/ciciot2023

End state: one subdirectory per family — the family folder IS the label, since
nothing inside a capture says which family it belongs to, so the folder is
preserved rather than flattened:

    data/external/ciciot2023/Benign_Final/BenignTraffic.pcap
    data/external/ciciot2023/DDoS-UDP_Flood/<capture>.pcap

ready for

    python scripts/run_ciciot2023_experiment.py
"""

from __future__ import annotations

import argparse
import http.cookiejar
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "data" / "external" / "ciciot2023"

BASE_URL = "https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023"
FORM_URL = f"{BASE_URL}/insert.php"
BROWSE_URL = f"{BASE_URL}/browse.php"
DOWNLOAD_URL = f"{BASE_URL}/download.php"

# The branch that carries real IPs, ports and a clock. See the module docstring
# for why the CSV branch is deliberately not used.
PCAP_ROOT = "PCAP"

# folder prefix -> KRONUS purpose. Matched case-insensitively against the real
# folder names listed by browse.php, so this stays correct if the publisher
# renames a directory — a mismatch surfaces as "matched no folders" rather than
# as a silently empty download.
FAMILY_PURPOSES: dict[str, str] = {
    "benign": "normal (Bouncer negative, Detective BENIGN)",
    "ddos": "dos/flood (Bouncer positive)",
    "dos-": "dos/flood (Bouncer positive)",
    "mirai": "dos/flood (Bouncer positive, distinct volumetric shape)",
    "recon": "probe/PORT_SCAN (Detective positive)",
    "vulnerabilityscan": "probe/PORT_SCAN (Detective positive)",
}

# The default fetch set: enough families for both lanes, and no more.
DEFAULT_FAMILIES: tuple[str, ...] = (
    "benign", "recon-portscan", "recon-hostdiscovery",
    "ddos-udp_flood", "dos-udp_flood", "mirai-udpplain",
)

# The form's own field names, in the order the page presents them. Required by
# the server; the values come from the caller, never from this file.
FORM_FIELDS: tuple[str, ...] = (
    "first_name", "last_name", "email", "institution", "job_title", "country",
)

_ENV_PREFIX = "KRONUS_CIC_"


def _form_values(args: argparse.Namespace) -> dict[str, str]:
    """Collect the registration fields from flags, then environment.

    Returning only non-empty values lets the caller see exactly which fields
    are still missing instead of posting blanks and reading the server's
    generic rejection.
    """
    flags = {
        "first_name": args.first_name, "last_name": args.last_name,
        "email": args.email, "institution": args.institution,
        "job_title": args.job_title, "country": args.country,
    }
    values: dict[str, str] = {}
    for field, flag_value in flags.items():
        env_value = os.environ.get(_ENV_PREFIX + field.upper(), "")
        value = (flag_value or env_value or "").strip()
        if value:
            values[field] = value
    return values


def _opener() -> urllib.request.OpenerDirector:
    """An opener that carries cookies, which is the whole point.

    The form's response sets an HttpOnly `Token` cookie; browse.php and
    download.php both 403 without it. A bare urlopen would look like the
    dataset is gone rather than gated.
    """
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def _register(opener: urllib.request.OpenerDirector, values: dict[str, str]) -> bool:
    """Submit the access form. Returns True when the server accepts it."""
    body = urllib.parse.urlencode(values).encode()
    req = urllib.request.Request(
        FORM_URL, data=body,
        headers={"User-Agent": "kronus-dataset-fetch/1.0", "Referer": f"{BASE_URL}/",
                 "Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with opener.open(req, timeout=120) as resp:
            payload = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: registration request failed: {exc}", file=sys.stderr)
        return False
    if '"ok":true' in payload.replace(" ", ""):
        print("registration accepted")
        return True
    print(f"error: registration was not accepted: {payload[:200]}", file=sys.stderr)
    return False


def _browse(opener: urllib.request.OpenerDirector, path: str) -> tuple[list[str], list[str]]:
    """List one folder: (subfolder paths, downloadable file paths)."""
    url = f"{BROWSE_URL}?p={urllib.parse.quote(path, safe='')}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "kronus-dataset-fetch/1.0", "Referer": f"{BASE_URL}/"})
    with opener.open(req, timeout=120) as resp:
        html = resp.read().decode("utf-8", "replace")
    folders = [urllib.parse.unquote(m) for m in
               re.findall(r'browse\.php\?p=([^"]+)"', html)]
    files = [urllib.parse.unquote(m) for m in
             re.findall(r'download\.php\?file=([^"]+)"', html)]
    # Drop the "up" link, which points back at the parent.
    folders = [f for f in folders if f.rstrip("/") != path.rstrip("/")]
    return folders, files


def _purpose_for(folder: str) -> str | None:
    """Map a real folder name onto a KRONUS purpose, or None to leave it out."""
    name = Path(folder.rstrip("/")).name.lower()
    for prefix, purpose in FAMILY_PURPOSES.items():
        if name.startswith(prefix) or name == prefix:
            return purpose
    return None


def _list_tree(opener: urllib.request.OpenerDirector) -> int:
    """Print the real tree so folder names are read, never assumed."""
    try:
        root_folders, root_files = _browse(opener, PCAP_ROOT)
    except (urllib.error.URLError, OSError, urllib.error.HTTPError) as exc:
        print(f"error: could not browse {PCAP_ROOT}: {exc}", file=sys.stderr)
        return 1

    print(f"{PCAP_ROOT}/: {len(root_folders)} folders, {len(root_files)} files")
    wanted = 0
    for folder in sorted(root_folders):
        purpose = _purpose_for(folder)
        try:
            _, files = _browse(opener, folder)
        except (urllib.error.URLError, OSError, urllib.error.HTTPError) as exc:
            print(f"  {folder}: unreadable ({exc})")
            continue
        mark = "->" if purpose else "  "
        if purpose:
            wanted += 1
        print(f"  {mark} {folder}: {len(files)} file(s)"
              + (f"  [{purpose}]" if purpose else "  [not used by either lane]"))
    print(f"\n{wanted} folder(s) map onto a lane; the rest are out of scope.")
    print("Every family folder is listed, so the default set in DEFAULT_FAMILIES "
          "can be checked against reality before a fetch.")
    return 0


def _looks_like_pcap(head: bytes) -> bool:
    """libpcap magic, incl. the nanosecond and byte-swapped variants."""
    return head[:4] in (b"\xa1\xb2\xc3\xd4", b"\xd4\xc3\xb2\xa1",
                        b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")


def _fetch(opener: urllib.request.OpenerDirector, name: str, dest: Path) -> int:
    """Stream one capture to `dest`. Returns the byte count, or -1 on failure."""
    url = f"{DOWNLOAD_URL}?file={urllib.parse.quote(name)}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url, headers={"User-Agent": "kronus-dataset-fetch/1.0", "Referer": f"{BASE_URL}/"})
    try:
        with opener.open(req, timeout=3600) as resp, open(tmp, "wb") as out:
            head = resp.read(4)
            out.write(head)
            total = len(head)
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch {name}: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return -1

    # A 403 body would have been written happily by the loop above; an HTML
    # error page is not a capture, so check the magic before keeping it.
    if not _looks_like_pcap(head):
        print(f"error: {name} is not a pcap (wrong session, or not a capture)",
              file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return -1

    tmp.replace(dest)
    return total


def _download_families(
    opener: urllib.request.OpenerDirector, families: list[str], per_family: int
) -> int:
    """Fetch a bounded set of captures, family by family."""
    root_folders, _ = _browse(opener, PCAP_ROOT)
    by_name = {Path(f.rstrip("/")).name.lower(): f for f in root_folders}

    selected: list[str] = []
    for want in families:
        want = want.lower()
        matches = [real for name, real in by_name.items()
                   if name == want or name.startswith(want)]
        if not matches:
            print(f"error: no folder matches {want!r} under {PCAP_ROOT}/ "
                  "(run --list to see the real names)", file=sys.stderr)
            return 1
        for real in sorted(matches):
            if real not in selected:
                selected.append(real)

    if not selected:
        print("error: no families selected", file=sys.stderr)
        return 1

    fetched = 0
    for folder in selected:
        purpose = _purpose_for(folder) or "unmapped"
        _, files = _browse(opener, folder)
        if not files:
            print(f"  {folder}: no downloadable files, skipping")
            continue
        # The family folder IS the label: every capture in DDoS-UDP_Flood is a
        # flood, every capture in Benign_Final is benign, and nothing inside a
        # capture says which it is. Flattening these into one directory would
        # leave the loader inferring the label from a filename, which holds
        # only by coincidence. So the folder is preserved.
        family_dir = DEST_DIR / Path(folder.rstrip("/")).name
        for name in sorted(files)[:per_family]:
            dest = family_dir / Path(name).name
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  {dest.relative_to(DEST_DIR)} already present, skipping")
                continue
            size = _fetch(opener, name, dest)
            if size < 0:
                return 1
            fetched += 1
            print(f"  {dest.relative_to(DEST_DIR)} ({size / 1_048_576:.1f} MB) [{purpose}]")

    if not fetched:
        print("nothing fetched (already complete?)")
    print(f"\n{len(selected)} family folder(s) selected; captures in {DEST_DIR}")
    print("Delete that directory once the experiment has run to reclaim the space.")
    return 0


def _copy_local(source: Path) -> int:
    import shutil

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    found = sorted(p for p in source.rglob("*") if p.suffix in (".pcap", ".cap", ".pcapng"))
    if not found:
        print(f"error: no captures under {source}", file=sys.stderr)
        return 1
    for path in found:
        # Keep each capture's own folder name, for the reason given in
        # _download_families: the folder is the label.
        family = path.parent.name if path.parent != source else ""
        dest = (DEST_DIR / family / path.name) if family else (DEST_DIR / path.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        print(f"  copied {dest.relative_to(DEST_DIR)}")
    return 0


MANUAL_INSTRUCTIONS = f"""\
CIC IoT 2023 could not be fetched automatically (no --register or --from-local).

To get it manually:
  1. The dataset is CIC IoT 2023 from the Canadian Institute for Cybersecurity
     (UNB): https://www.unb.ca/cic/datasets/iotdataset-2023.html
  2. That page links to a registration form at:
       {BASE_URL}/
     Fill it in, then download the packet captures. This loader uses the PCAP
     branch on purpose: the published CSVs carry no IPs, no ports and no
     timestamps, so the Bouncer's port- and clock-derived features would have
     to be invented. See this script's module docstring.
  3. Put them in:
       {DEST_DIR}
     (or run this script again with --from-local pointing at their folder).

The easiest route is simply:
  python scripts/download_ciciot2023.py --register \\
      --first-name <you> --last-name <you> --email <you@example.org> \\
      --institution <org> --job-title <title> --country <country>

Run `python scripts/download_ciciot2023.py --list` first to see the real
family folder names.

Cite the dataset if you publish results from it:
  Neto et al., "CICIoT2023: A real-time dataset and benchmark for large-scale
  attacks in IoT environment", Sensors 23(13):5941, 2023.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the CIC IoT 2023 packet captures (form-gated).",
    )
    parser.add_argument("--list", action="store_true",
                        help="browse the dataset tree and print it; no download")
    parser.add_argument("--register", action="store_true",
                        help="submit the access form and download")
    parser.add_argument("--from-local", type=Path, default=None,
                        help="directory already holding the captures")
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES),
                        help="comma-separated family folders to fetch")
    parser.add_argument("--max-files-per-family", type=int, default=3,
                        help="cap the number of captures taken from each family")
    for field in FORM_FIELDS:
        parser.add_argument("--" + field.replace("_", "-"), default=None,
                            help=f"form field: {field.replace('_', ' ')}")
    args = parser.parse_args(argv)

    if args.from_local is not None:
        return _copy_local(args.from_local)

    if args.list or args.register:
        values = _form_values(args)
        missing = [f for f in FORM_FIELDS if f not in values]
        if missing:
            print("error: this mode needs every form field; missing: "
                  + ", ".join(missing)
                  + "\n       pass them as flags or set "
                  + ", ".join(_ENV_PREFIX + f.upper() for f in missing),
                  file=sys.stderr)
            return 2
        opener = _opener()
        if not _register(opener, values):
            return 1
        if args.list:
            return _list_tree(opener)
        families = [f.strip() for f in args.families.split(",") if f.strip()]
        return _download_families(opener, families, args.max_files_per_family)

    print(MANUAL_INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
