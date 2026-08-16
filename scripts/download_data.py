#!/usr/bin/env python3
"""Fetches the real NSL-KDD train/test split into data/real/.

This is the ONLY real-data source the demo and test suite run against
(see docs/setup.md). The Digital Twin's synthetic generator writes to
data/synthetic/ instead, and is never read by the demo or the test suite —
only by services/drift_watcher's improvement loop.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "data" / "real"

FILES = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
}


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        dest = DEST_DIR / filename
        if dest.exists():
            print(f"already have {dest}")
            continue
        print(f"downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        lines = sum(1 for _ in open(dest))
        print(f"  saved {dest} ({lines} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
