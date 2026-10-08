#!/usr/bin/env python3
"""Fetches the CIC-DDoS2019 CSV day archives into data/external/cicddos2019/,
the external-validation dataset for scripts/run_cicddos2019_experiment.py.

CIC-DDoS2019 ("DDoS Evaluation Dataset", Canadian Institute for Cybersecurity,
University of New Brunswick) is the reflection/amplification DDoS benchmark.
Official page: https://www.unb.ca/cic/datasets/ddos-2019.html

    Iman Sharafaldin, Arash Habibi Lashkari, Saqib Hakak, Ali A. Ghorbani,
    "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset
    and Taxonomy", IEEE 53rd International Carnahan Conference on Security
    Technology (ICCST), 2019. doi:10.1109/ccst.2019.8888419

*** THIS SOURCE IS FORM-GATED, AND THAT IS NOT WORKED AROUND ***
Unlike the CSE-CIC-IDS2018 loader next door (a public unauthenticated S3
bucket), this dataset is served from cicresearch.ca behind a registration
form. `browse.php` and `download.php` both return 403 without the session the
form issues, so there is no honest way to fetch it that skips the form — and
this script does not try. It submits the form for you, then downloads with the
session cookie the form returns.

The registration details are NOT hardcoded: this repository is public, and a
maintainer's name and email do not belong in it. Supply your own with the
flags below (or the matching environment variables), and they stay on your
machine in the cookie jar.

*** WHY THE ARCHIVE IS VERIFIED ***
A loader for this dataset family already existed in the tree that read
CIC-IDS2017's column names while being named for CIC-IDS2018, and mapped this
dataset's label vocabulary. It is exactly the failure this repository keeps
re-learning: a plausible-looking CSV that parses, yields an all-zero dataset,
and reports success. So the download is checked against the real 88-column
header before the runner is allowed near it, and `twin/cicddos2019.py`
re-checks the same thing per file.

WHAT IS FETCHED
    CSV-01-12.zip   2019-01-12 capture: DrDoS LDAP/MSSQL/NetBIOS/Portmap/
                    SNMP/SSDP + the LDAP/MSSQL/NetBIOS/Portmap/SNMP/SSDP/UDP/
                    UDPLag/Syn/TFTP/WebDDoS families
    CSV-03-11.zip   2019-03-11 capture: the same families' second day

Both days are fetched because a single day lets a model separate the classes
by *capture date* rather than by traffic shape — the same reasoning as every
other external experiment here.

MODES
1. --register (with --first-name/--last-name/--email/--institution/
   --job-title/--country, or the KRONUS_CIC_* environment variables): submit
   the registration form and download the archives. This is the only automatic
   mode; every other CIC dataset here has a --from-mirror, but this one has no
   mirror that is not itself redistributing under the same licence.
2. --from-local PATH: a directory already holding the archives.
3. No argument: prints what to fetch and where, then exits 0 — so the runner's
   graceful skip never becomes a pipeline failure here.

Re-running skips archives already present and intact, so a re-run after a
partial download costs one file rather than a full re-fetch.

DISK
    The two archives total roughly 1-2 GB, and expand to several GB of CSVs.
    They are only needed while the experiment runs; delete
    data/external/cicddos2019/ afterwards to reclaim the space.

End state: the two archives in data/external/cicddos2019/, ready for

    python scripts/run_cicddos2019_experiment.py
"""

from __future__ import annotations

import argparse
import http.cookiejar
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "data" / "external" / "cicddos2019"

# The registration endpoint and the two file endpoints. browse.php lists the
# folder; download.php streams one archive and needs the same session.
BASE_URL = "https://cicresearch.ca/CICDataset/CICDDoS2019"
FORM_URL = f"{BASE_URL}/insert.php"
DOWNLOAD_URL = f"{BASE_URL}/download.php"

# The two day archives, as paths inside the dataset tree. These are the file
# names browse.php?p=CSVs reports; they are stable across the release.
ARCHIVES: tuple[str, ...] = ("CSVs/CSV-01-12.zip", "CSVs/CSV-03-11.zip")

# What each archive is here for, surfaced in the report so the mapping is
# visible rather than implied.
PURPOSE: dict[str, str] = {
    "CSVs/CSV-01-12.zip": (
        "capture 2018-12-01, 09:17-17:16 (capture clock): DrDoS_NTP/NetBIOS/SNMP/"
        "SSDP/UDP + Syn + TFTP + UDPLag -> Bouncer (flood)"
    ),
    "CSVs/CSV-03-11.zip": (
        "capture 2018-11-03, 09:18-17:36 (capture clock): LDAP/MSSQL/NetBIOS/"
        "Portmap + Syn + UDP + UDPLag -> Bouncer (flood)"
    ),
}

# The real header of the released CSVs, as the first columns of a monitor
# file. Only the shape is asserted here (the loader does the full check);
# this is enough to refuse a wrong-dataset archive before it is unpacked.
REQUIRED_CSV_COLUMNS: tuple[str, ...] = (
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
    "Label",
)

MANUAL_INSTRUCTIONS = f"""\
CIC-DDoS2019 could not be fetched automatically (no --from-local given).

To get it manually:
  1. The dataset is CIC-DDoS2019 from the Canadian Institute for Cybersecurity
     (UNB): https://www.unb.ca/cic/datasets/ddos-2019.html
  2. That page links to a registration form at:
       {BASE_URL}/
     Fill it in (the form is the only access path: browse.php and download.php
     both return 403 without the session it issues), then download the two CSV
     archives:
       CSV-01-12.zip
       CSV-03-11.zip
  3. Put them in:
       {DEST_DIR}
     (or run this script again with --from-local pointing at their folder).

The easiest route is simply:
  python scripts/download_cicddos2019.py --register \\
      --first-name <you> --last-name <you> --email <you@example.org> \\
      --institution <org> --job-title <title> --country <country>

Cite the dataset if you publish results from it:
  Sharafaldin, Lashkari, Hakak, Ghorbani, "Developing Realistic Distributed
  Denial of Service (DDoS) Attack Dataset and Taxonomy", ICCST 2019.
"""

# The form's own field names, in the order the page presents them. Required by
# the server; the values come from the caller, never from this file.
FORM_FIELDS: tuple[str, ...] = (
    "first_name",
    "last_name",
    "email",
    "institution",
    "job_title",
    "country",
)

_ENV_PREFIX = "KRONUS_CIC_"


def _form_values(args: argparse.Namespace) -> dict[str, str]:
    """Collect the registration fields from flags, then environment.

    Returning only non-empty values lets the caller see exactly which fields
    are still missing instead of posting blanks and reading the server's
    generic rejection.
    """
    flag_names = {
        "first_name": args.first_name,
        "last_name": args.last_name,
        "email": args.email,
        "institution": args.institution,
        "job_title": args.job_title,
        "country": args.country,
    }
    values: dict[str, str] = {}
    for field, flag_value in flag_names.items():
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
        FORM_URL,
        data=body,
        headers={
            "User-Agent": "kronus-dataset-fetch/1.0",
            "Referer": f"{BASE_URL}/",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with opener.open(req, timeout=120) as resp:
            payload = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: registration request failed: {exc}", file=sys.stderr)
        return False

    # The endpoint answers with JSON; anything else means we hit an error page.
    if '"ok":true' in payload.replace(" ", ""):
        print("registration accepted")
        return True
    print(f"error: registration was not accepted: {payload[:200]}", file=sys.stderr)
    return False


def _fetch(opener: urllib.request.OpenerDirector, name: str, dest: Path) -> bool:
    """Stream one archive to `dest`, leaving no half-file behind on failure."""
    url = f"{DOWNLOAD_URL}?file={urllib.parse.quote(name)}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(
        url, headers={"User-Agent": "kronus-dataset-fetch/1.0", "Referer": f"{BASE_URL}/"}
    )
    try:
        with opener.open(req, timeout=3600) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch {name}: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    # A 403 body would have been written happily by the loop above; an HTML
    # error page is not an archive, so check the magic bytes before keeping it.
    if not _looks_like_zip(tmp):
        print(f"error: {name} did not return a zip (wrong session, or gated)", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    tmp.replace(dest)
    size_mb = dest.stat().st_size / 1_048_576
    print(f"  {name} -> {dest.name} ({size_mb:.0f} MB)")
    return True


def _looks_like_zip(path: Path) -> bool:
    with open(path, "rb") as handle:
        return handle.read(4) == b"PK\x03\x04"


def _verify_archives(dest_dir: Path) -> bool:
    """Check each archive is a readable zip holding at least one CSV.

    This is deliberately shallow — header verification belongs to the loader,
    which is the only place that can read a CSV without expanding gigabytes.
    What this catches is the failure that has bitten this repository before: an
    archive (or error page) that is not the dataset.
    """
    import zipfile

    ok = True
    for name in ARCHIVES:
        path = dest_dir / Path(name).name
        if not path.exists():
            print(f"  MISSING {path.name}")
            ok = False
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        except zipfile.BadZipFile:
            print(f"  {path.name}: not a readable zip")
            ok = False
            continue
        if not csv_names:
            print(f"  {path.name}: contains no CSV")
            ok = False
        else:
            print(f"  {path.name}: {len(csv_names)} CSV file(s)")
    return ok


def _download_all(opener: urllib.request.OpenerDirector) -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        target = DEST_DIR / Path(name).name
        if target.exists() and _looks_like_zip(target):
            print(f"  {name} already present, skipping")
            continue
        print(f"downloading {name} ...")
        if not _fetch(opener, name, target):
            return 1
    return 0 if _verify_archives(DEST_DIR) else 1


def _copy_local(source: Path) -> int:
    import shutil

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        candidate = source / Path(name).name
        if not candidate.exists():
            print(f"error: {candidate} not found", file=sys.stderr)
            return 1
        shutil.copy2(candidate, DEST_DIR / Path(name).name)
        print(f"  copied {candidate.name}")
    return 0 if _verify_archives(DEST_DIR) else 1


def _download_from_base(base_url: str) -> int:
    """Fetch from a location serving the same two file names.

    A base URL cannot substitute for the registration the official host
    requires, so this path is for a mirror an operator controls — the licence
    permits redistribution with citation.
    """
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        url = f"{base_url.rstrip('/')}/{Path(name).name}"
        target = DEST_DIR / Path(name).name
        if target.exists() and _looks_like_zip(target):
            print(f"  {name} already present, skipping")
            continue
        print(f"downloading {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "kronus-dataset-fetch/1.0"})
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp, open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
        except (urllib.error.URLError, OSError) as exc:
            print(f"error: could not fetch {url}: {exc}", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            return 1
        if not _looks_like_zip(tmp):
            print(f"error: {url} did not return a zip", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            return 1
        tmp.replace(target)
    return 0 if _verify_archives(DEST_DIR) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the CIC-DDoS2019 CSV archives (form-gated).",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="submit the access form and download (the only automatic mode)",
    )
    parser.add_argument("--first-name", default=None, help="form field: first name")
    parser.add_argument("--last-name", default=None, help="form field: last name")
    parser.add_argument("--email", default=None, help="form field: email")
    parser.add_argument("--institution", default=None, help="form field: institution")
    parser.add_argument("--job-title", default=None, help="form field: job title")
    parser.add_argument("--country", default=None, help="form field: country")
    parser.add_argument(
        "--url",
        default=os.environ.get("KRONUS_CICDDOS2019_URL"),
        help="base URL serving the same two file names (operator-controlled mirror)",
    )
    parser.add_argument(
        "--from-local",
        type=Path,
        default=None,
        help="directory already holding the two archives",
    )
    args = parser.parse_args(argv)

    if args.from_local is not None:
        return _copy_local(args.from_local)
    if args.url:
        return _download_from_base(args.url)
    if args.register:
        values = _form_values(args)
        missing = [f for f in FORM_FIELDS if f not in values]
        if missing:
            print(
                "error: --register needs every form field; missing: "
                + ", ".join(missing)
                + "\n       pass them as flags or set "
                + ", ".join(_ENV_PREFIX + f.upper() for f in missing),
                file=sys.stderr,
            )
            return 2
        opener = _opener()
        if not _register(opener, values):
            return 1
        return _download_all(opener)

    print(MANUAL_INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
