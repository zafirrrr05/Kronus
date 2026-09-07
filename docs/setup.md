# Setup

Step by step, from a clean environment to a passing test suite and a
completed demo run. Every command below is exactly what was used to build
and verify this repository — nothing here is aspirational.

## 1. Requirements

- Python 3.12 (3.11+ should work; 3.12 is what this repo was built and
  tested against)
- ~500 MB free disk for dependencies, plus a few hundred MB for the real
  NSL-KDD dataset and trained model artifacts
- Internet access to PyPI (dependencies), GitHub (the OPA binary and the
  NSL-KDD dataset mirror)

Nothing else is required for the default configuration. Postgres and a
Redpanda/Kafka broker are optional, production-parity backends — see
[§6](#6-optional-production-backends).

## 2. Clone and install Python dependencies

```bash
cd kronus
pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes everything in `requirements.txt` plus
`pytest`, `pytest-asyncio`, and `ruff`. If you only intend to run the
system (not the tests), `pip install -r requirements.txt` is enough.

If `pip` reports an externally-managed-environment error (common on
recent Debian/Ubuntu-based systems), add `--break-system-packages`, or use
a virtual environment instead:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

## 3. Install the OPA binary

The Response Engine shells out to a real Open Policy Agent binary — there
is no Python re-implementation of Rego in this repo.

```bash
curl -sL -o /usr/local/bin/opa \
  "https://github.com/open-policy-agent/opa/releases/latest/download/opa_linux_amd64_static"
chmod +x /usr/local/bin/opa
opa version
```

(For macOS, download the `opa_darwin_amd64` or `opa_darwin_arm64` asset
from the same [releases page](https://github.com/open-policy-agent/opa/releases)
instead.)

Verify the policy suite passes on its own, independent of anything else
in the repo:

```bash
opa test policy/ -v
```

You should see 22 tests pass, including the seven required cases named in
the spec (confidence boundary, allowlist, gray-zone, circuit-breaker,
malformed-input fail-closed, FR-12 window correlation, and the Decoy's
no-special-case-bypass).

## 4. Download the real dataset

```bash
python scripts/download_data.py
```

Fetches `KDDTrain+.txt` (125,973 rows) and `KDDTest+.txt` (22,544 rows)
into `data/real/`. This is the *only* real-data source the demo and test
suite use — see the README's "What's real in this repository" section.

## 5. Train the models

```bash
python -m services.bouncer.train
python -m services.detective.train --limit 20000 --epochs 4
```

Both scripts train on `data/real/KDDTrain+.txt`, evaluate on the
held-out `data/real/KDDTest+.txt`, print real metrics, and save trained
artifacts (plus a `metrics.json`) under `models/registry/`.

- Bouncer takes well under a second (native XGBoost).
- Detective's `--limit` controls how many real rows are replayed into
  training graph windows; 20,000 trains in roughly 10 seconds on a
  laptop CPU and is what the checked-in metrics reflect. Increase it for
  a larger training set (proportionally longer training time — the
  hand-rolled GAT has no GPU acceleration path).

Both scripts are safe to re-run; they overwrite the previous artifact.

## 6. Run the tests

```bash
python -m pytest tests/ -q
```

All 159 tests should pass. A few notes on what you'll see:

- `tests/unit/` runs fast (well under a minute) and covers each component
  in isolation, including real invariant checks (a numerical gradient
  check for the Detective's backprop, a real tamper-detection test for
  the Logbook's hash chain, real OPA subprocess calls for every Response
  Engine test).
- `tests/integration/test_flow_cases.py` loads the real trained models
  from `models/registry/` and real NSL-KDD rows — if you skip steps 4-5,
  these tests **skip gracefully** with a message telling you what to run
  first, rather than failing.
- If Postgres isn't running, `tests/unit/test_logbook.py`'s Postgres
  parametrization is skipped automatically; the SQLite parametrization
  still covers the same hash-chain logic. See §6 to run against real
  Postgres too.

Run just the policy suite, just the flow cases, or just one file the same
way:

```bash
python -m pytest tests/integration/test_flow_cases.py -v
python -m pytest tests/unit/test_detective_model.py -v
```

## 6b. (Optional) External-dataset validation — CIC-IDS2017

The models are *trained* on NSL-KDD (step 5). CIC-IDS2017 is an independent
dataset — a different network, captured with a different tool (CICFlowMeter)
years later — used to measure cross-dataset generalization, the SLO table's
"External-dataset F1" row. Nothing is retrained here; the NSL-KDD-trained
models score CIC-IDS2017 flows they have never seen.

The dataset (~500 MB of labelled-flow CSVs) is hosted by the University of
New Brunswick behind a short registration form, so it isn't fetched by
default. Get it, then validate:

```bash
# Option A: you have a direct URL to a .zip of the CSVs
python scripts/download_cicids2017.py --url "<your-url>"

# Option B: you downloaded the GeneratedLabelledFlows archive manually from
# https://www.unb.ca/cic/datasets/ids-2017.html
python scripts/download_cicids2017.py --from-local ~/Downloads/GeneratedLabelledFlows.zip

# Option C: just print instructions and where to drop the CSVs
python scripts/download_cicids2017.py

# then, once the CSVs are in data/external/cicids2017/:
python scripts/validate_cicids2017.py
```

`validate_cicids2017.py` replays CIC-IDS2017 flows through the *same* live
featurizer and graph builder used at training time (no train/serve skew),
scores both lanes, writes `models/registry/external_validation_cicids2017.json`,
and exits non-zero if a lane falls below the F1 > 0.85 SLO (pass `--no-gate`
to report without gating, or `--limit N` to cap rows per lane for a quick
run). If the dataset or trained models aren't present, it skips gracefully
with a message rather than failing.

The loader (`twin/cicids2017.py`) is covered by
`tests/unit/test_cicids2017_loader.py`, which runs without the large download
(it builds small CIC-IDS2017-shaped frames in a tmp dir); the real-data test
in that file skips gracefully until the CSVs are in place.

## 7. Run the demo

```bash
python -m demo.run_demo
```

Trains the models first if `models/registry/` is empty, then walks
through all seven flow cases from the design docs against real data,
prints what happened at each step, drains the LLM Explainer's job queue,
and verifies the Logbook's hash chain end to end. A machine-readable
summary is written to `kronus_demo_report.json` in the repo root.

By default the LLM Explainer runs **offline** (a deterministic, templated
narrative — see `docs/instructions.md`), so the demo costs nothing and
needs no API key. To exercise the real Claude API call instead:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m demo.run_demo
```

## 8. (Optional) Run the API server

```bash
python -m api.serve
```

Starts a real `KronusSystem` (in-memory event bus, SQLite logbook by
default) behind a FastAPI app on `http://localhost:8000`, with Prometheus
metrics on `:9100`. See `docs/frontend_guide.md` for the full endpoint
reference and what a frontend needs.

## 9. Optional: production backends

The default configuration (in-memory event bus, SQLite logbook) needs no
infrastructure. To exercise the real Redpanda and Postgres code paths
locally instead:

```bash
docker compose -f infra/docker-compose.yml up
```

This starts Redpanda, Postgres, Prometheus, Grafana, and the API service
wired to all of them. Grafana is at `http://localhost:3000` (anonymous
viewer access enabled for local dev) with the KRONUS dashboard
pre-provisioned; Prometheus is at `http://localhost:9090`.

To point a locally-run (non-Docker) instance at a real Postgres instead
of SQLite, without the full compose stack:

```bash
# once, to create the role/database:
psql -U postgres -c "CREATE USER kronus WITH PASSWORD 'kronus_dev_local';"
psql -U postgres -c "CREATE DATABASE kronus OWNER kronus;"

export KRONUS_LOGBOOK_BACKEND=postgres
export KRONUS_POSTGRES_DSN="dbname=kronus user=kronus password=kronus_dev_local host=localhost"
python -m pytest tests/unit/test_logbook.py -v   # now exercises the Postgres parametrization too
```

## 10. Lint

```bash
ruff check .
```

The codebase is clean under `ruff check . --select F401,F841` (unused
imports/variables) as an unconditional bar; broader style rules
(`E`, `W`, `I`, `UP`, `B`) are also enabled in `pyproject.toml`.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError: data/real/KDDTrain+.txt not found` | Run step 4 |
| `pytest.skip("real trained models not found ...")` | Run step 5 |
| `opa: command not found` | Run step 3, or set `KRONUS_OPA_BINARY` to the binary's full path |
| Postgres tests silently skipped | Expected if Postgres isn't running — see §9 |
| `pip install` fails with an externally-managed-environment error | Add `--break-system-packages` or use a venv (§2) |
