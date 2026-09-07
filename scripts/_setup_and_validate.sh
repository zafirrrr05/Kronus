#!/usr/bin/env bash
# One-shot bring-up + external validation for the CIC-IDS2017 work.
# Idempotent: safe to re-run. Each step guards on whether its output exists.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "== repo: $ROOT =="

echo "== [1/7] python + deps =="
python3 --version
python3 -c "import pandas,numpy,sklearn,xgboost,pydantic,scipy,pytest,pytest_asyncio,autograd,onnx; print('all deps import OK')" \
  || { echo 'MISSING DEPS — installing'; python3 -m pip install -r requirements-dev.txt; }

echo "== [2/7] NSL-KDD data =="
if [ ! -f data/real/KDDTrain+.txt ]; then
  python3 scripts/download_data.py
else
  echo "already have data/real/KDDTrain+.txt"
fi
wc -l data/real/*.txt 2>/dev/null || true

echo "== [3/7] train Bouncer =="
if [ ! -f models/registry/bouncer/bouncer.json ]; then
  python3 -m services.bouncer.train
else
  echo "already trained: models/registry/bouncer/"
fi

echo "== [4/7] train Detective =="
if [ ! -f models/registry/detective/detective.npz ]; then
  python3 -m services.detective.train --limit 20000 --epochs 4
else
  echo "already trained: models/registry/detective/"
fi

echo "== [5/7] full test suite =="
python3 -m pytest tests/ -q 2>&1 | tail -40

echo "== [6/7] CIC-IDS2017 loader unit tests (no big download needed) =="
python3 -m pytest tests/unit/test_cicids2017_loader.py -v 2>&1 | tail -40

echo "== [7/7] CIC-IDS2017 external validation (skips gracefully if data absent) =="
python3 scripts/download_cicids2017.py || true
python3 scripts/validate_cicids2017.py --no-gate || true

echo "== DONE =="
