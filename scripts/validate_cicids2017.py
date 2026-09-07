#!/usr/bin/env python3
"""The CIC-IDS2017 external-validation harness the README's SLO table promises
("External-dataset F1 > 0.85 | CIC-IDS2017-compatible validation harness").

This is a *cross-dataset* test, which is a stronger claim than a held-out
split: the Bouncer and Detective are trained on NSL-KDD (services/*/train.py)
and evaluated here on CIC-IDS2017 — a different network, a different capture
tool (CICFlowMeter), 18 years later. The models see CIC data for the first
time at scoring. Nothing is retrained here by default.

What it measures, per lane, kept honest and identical in spirit to the
NSL-KDD training scripts so the numbers are comparable:

- Bouncer: binary flood-vs-not F1. CIC rows are replayed through the SAME
  live FlowFeaturizer (services/bouncer/features.py) used at train time, so
  there is no train/serve feature skew — the 6 rate/entropy features are
  recomputed from replayed TelemetryEvents exactly as in production.
- Detective: macro benign-vs-port_scan F1 over graph windows built by the
  SAME WindowedGraphBuilder, using the same stratified per-category replay
  the training script uses (see services/detective/train.py's
  build_labeled_snapshots and its docstring for why windows are built per
  category rather than by inspecting a mixed window's row mix).

Run:
    python scripts/validate_cicids2017.py                 # scores existing models
    python scripts/validate_cicids2017.py --limit 40000   # cap rows per lane
    python scripts/validate_cicids2017.py --data data/external/cicids2017

Writes models/registry/external_validation_cicids2017.json (machine-readable)
and prints a summary table. Exits non-zero if a lane is present but falls
below the F1 > 0.85 SLO, so CI can gate on it; exits 0 with a skip notice if
the dataset or trained models aren't present (same graceful-skip contract as
tests/integration/test_flow_cases.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from libs.constants import DataOrigin, Label
from services.bouncer.features import FEATURE_NAMES, FlowFeaturizer
from services.bouncer.model import BouncerModel
from services.detective.model import DetectiveModel
from services.graph_builder.builder import WindowedGraphBuilder
from services.telemetry_exporter.converters import flow_row_to_event
from twin.cicids2017 import load_cicids2017

BOUNCER_DIR = "models/registry/bouncer"
DETECTIVE_DIR = "models/registry/detective"
DEFAULT_DATA_DIR = "data/external/cicids2017"
OUT_PATH = "models/registry/external_validation_cicids2017.json"
SLO_F1 = 0.85

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
ROW_SPACING_MS = 20  # matches services/*/train.py replay spacing


# --- Bouncer lane -----------------------------------------------------------

def _bouncer_features_for_stream(rows, label: int):
    """Replay one category's rows through the live featurizer, source-sorted
    so each host's burst is temporally contiguous — identical to
    services/bouncer/train.py's _features_for_stream, so CIC features are
    built the same way NSL-KDD training features were."""
    rows = sorted(rows, key=lambda r: r.source_ip)
    featurizer = FlowFeaturizer()
    X = []
    for i, row in enumerate(rows):
        assert row.origin == DataOrigin.REAL
        event = flow_row_to_event(row, ts=T0 + timedelta(milliseconds=i * ROW_SPACING_MS))
        feats = featurizer.features_for(event)
        X.append([feats[name] for name in FEATURE_NAMES])
    if not X:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,), dtype=int)
    return np.array(X), np.full(len(X), label)


def evaluate_bouncer(rows, limit: int | None) -> dict | None:
    if not Path(BOUNCER_DIR, "bouncer.json").exists():
        return None
    dos_rows = [r for r in rows if r.category == "dos"]
    other_rows = [r for r in rows if r.category != "dos"]
    if limit is not None:
        half = limit // 2
        dos_rows, other_rows = dos_rows[:half], other_rows[:half]
    if not dos_rows or not other_rows:
        return {"skipped": "CIC-IDS2017 subset lacks both flood and non-flood rows"}

    X_dos, y_dos = _bouncer_features_for_stream(dos_rows, label=1)
    X_other, y_other = _bouncer_features_for_stream(other_rows, label=0)
    X = np.vstack([X_dos, X_other])
    y = np.concatenate([y_dos, y_other])

    model = BouncerModel.load(BOUNCER_DIR)
    proba = model.predict_proba(X)
    y_pred = (proba >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    accuracy = float((y_pred == y).mean())
    return {
        "task": "binary flood-vs-not, cross-dataset (trained NSL-KDD, tested CIC-IDS2017)",
        "n": int(len(y)),
        "n_flood": int(y.sum()),
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


# --- Detective lane ---------------------------------------------------------

def _windowed_snapshots(rows, label: Label):
    builder = WindowedGraphBuilder(window_seconds=2.0)
    labeled = []
    for i, row in enumerate(rows):
        assert row.origin == DataOrigin.REAL
        ts = T0 + timedelta(milliseconds=i * ROW_SPACING_MS)
        snapshot = builder.ingest(flow_row_to_event(row, ts=ts))
        if snapshot is not None and snapshot.edges:
            labeled.append((snapshot, label))
    final = builder.flush(end_time=T0 + timedelta(milliseconds=len(rows) * ROW_SPACING_MS))
    if final is not None and final.edges:
        labeled.append((final, label))
    return labeled


def evaluate_detective(rows, limit: int | None) -> dict | None:
    if not Path(DETECTIVE_DIR, "detective.npz").exists():
        return None
    normal_rows = [r for r in rows if r.category == "normal"]
    probe_rows = [r for r in rows if r.category == "probe"]
    if limit is not None:
        half = limit // 2
        normal_rows, probe_rows = normal_rows[:half], probe_rows[:half]
    if not normal_rows or not probe_rows:
        return {"skipped": "CIC-IDS2017 subset lacks both benign and portscan rows"}

    labeled = _windowed_snapshots(normal_rows, Label.BENIGN)
    labeled += _windowed_snapshots(probe_rows, Label.PORT_SCAN)
    if not labeled:
        return {"skipped": "no graph windows with edges were produced"}

    model = DetectiveModel.load(DETECTIVE_DIR)
    y_true, y_pred = [], []
    for snapshot, label in labeled:
        verdict = model.predict_verdict(snapshot, window_id=snapshot.window_id)
        y_true.append(label.value)
        y_pred.append(verdict.label if verdict.label != Label.UNCERTAIN.value else "port_scan")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["benign", "port_scan"], average="macro", zero_division=0
    )
    accuracy = float(np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)]))
    return {
        "task": "macro benign-vs-port_scan on graph windows, cross-dataset "
                "(trained NSL-KDD, tested CIC-IDS2017)",
        "n_windows": len(labeled),
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


# --- driver -----------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA_DIR,
                        help=f"CIC-IDS2017 CSV file or directory (default {DEFAULT_DATA_DIR})")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap rows per lane per class (default: all)")
    parser.add_argument("--no-gate", action="store_true",
                        help="report metrics but always exit 0 (don't fail CI on sub-SLO F1)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"SKIP: {data_path} not present. Run `python scripts/download_cicids2017.py` "
              "or pass --data pointing at the CIC-IDS2017 CSVs (see docs/setup.md §external).")
        return 0
    if not Path(BOUNCER_DIR, "bouncer.json").exists() and \
       not Path(DETECTIVE_DIR, "detective.npz").exists():
        print("SKIP: no trained models in models/registry/. Run services/bouncer/train.py "
              "and services/detective/train.py first (see docs/setup.md).")
        return 0

    print(f"Loading CIC-IDS2017 from {data_path} ...")
    start = time.perf_counter()
    rows = load_cicids2017(data_path)
    load_seconds = time.perf_counter() - start
    cats = {}
    for r in rows:
        cats[r.category] = cats.get(r.category, 0) + 1
    print(f"  loaded {len(rows)} rows in {load_seconds:.1f}s; category counts: {cats}")

    print("Scoring Bouncer (fast lane) on CIC-IDS2017 ...")
    bouncer_metrics = evaluate_bouncer(rows, args.limit)
    print(f"  {bouncer_metrics}")

    print("Scoring Detective (deep lane) on CIC-IDS2017 ...")
    detective_metrics = evaluate_detective(rows, args.limit)
    print(f"  {detective_metrics}")

    report = {
        "dataset": "CIC-IDS2017 (GeneratedLabelledFlows)",
        "data_path": str(data_path),
        "evaluation": "cross-dataset external validation (models trained on NSL-KDD)",
        "slo_f1_threshold": SLO_F1,
        "n_rows_loaded": len(rows),
        "category_counts": cats,
        "bouncer": bouncer_metrics,
        "detective": detective_metrics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_PATH).write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_PATH}")

    # Gate: any lane that produced a real F1 must clear the SLO.
    gated_f1s = [
        m["f1"] for m in (bouncer_metrics, detective_metrics)
        if isinstance(m, dict) and "f1" in m
    ]
    print("\n=== CIC-IDS2017 external validation summary ===")
    for name, m in (("Bouncer", bouncer_metrics), ("Detective", detective_metrics)):
        if isinstance(m, dict) and "f1" in m:
            verdict = "PASS" if m["f1"] >= SLO_F1 else "BELOW SLO"
            print(f"  {name:10s} F1={m['f1']:.4f}  ({verdict}, SLO > {SLO_F1})")
        else:
            print(f"  {name:10s} {m}")

    if not args.no_gate and gated_f1s and any(f < SLO_F1 for f in gated_f1s):
        print("\nAt least one lane is below the F1 > 0.85 SLO.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
