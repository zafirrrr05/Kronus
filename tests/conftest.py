import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
opa_bin = REPO_ROOT / "bin" / "opa"
if opa_bin.exists():
    os.environ["KRONUS_OPA_BINARY"] = str(opa_bin)
    os.environ["PATH"] = f"{REPO_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
