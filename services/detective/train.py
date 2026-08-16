"""Trains the Detective on graph snapshots built from real NSL-KDD data.

Real-data scope, stated plainly: normal and probe categories only. DoS is
the Bouncer's target (see services/bouncer/train.py); r2l/u2r have no
KRONUS label to train toward (see twin/nsl_kdd.py's module docstring);
lateral_movement has no real-data ground truth NSL-KDD's row-independent
schema can provide (see twin/synthetic.py) and is trained via the twin's
improvement loop instead, never claimed on real-data metrics here.

A window's label comes from which category stream it was built from —
normal rows windowed separately from probe rows (see
build_labeled_snapshots) — rather than inspecting each raw window's row
mix, since NSL-KDD's normal and probe rows are densely interleaved in much
of the file (see that function's docstring for what this replaced and why).

Run: python -m services.detective.train [--limit N]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from libs.constants import DataOrigin, Label
from services.detective.model import DetectiveModel
from services.graph_builder.builder import WindowedGraphBuilder
from services.telemetry_exporter.converters import flow_row_to_event
from twin.nsl_kdd import load_nsl_kdd

TRAIN_PATH = "data/real/KDDTrain+.txt"
TEST_PATH = "data/real/KDDTest+.txt"
MODEL_DIR = "models/registry/detective"
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
ROW_SPACING_MS = 20  # synthetic replay spacing — see converters.flow_row_to_event's docstring


def build_labeled_snapshots(rows, limit: int | None = None) -> list[tuple]:
    """Replays rows (normal/probe only) through a real WindowedGraphBuilder
    with increasing timestamps, producing (GraphSnapshot, Label) pairs.

    Normal and probe rows are replayed as two separate streams, each
    windowed on its own, rather than one stream in raw file order. Caught
    for real: NSL-KDD's normal and probe rows are densely interleaved in
    much of the file (every 100-row window in the first 5,000 rows
    contained at least one probe row), so "does this window contain a
    probe row" produced port_scan for every window regardless of any
    proportion threshold tried. Every row used below is still real,
    untouched NSL-KDD data — this only changes which real rows get
    windowed together, which is a legitimate stratified-replay choice
    (raw file order isn't a meaningful clock to begin with; see
    twin/nsl_kdd.py) not a synthetic substitution.
    """
    normal_rows = [r for r in rows if r.category == "normal"]
    probe_rows = [r for r in rows if r.category == "probe"]
    if limit is not None:
        half = limit // 2
        normal_rows, probe_rows = normal_rows[:half], probe_rows[:half]

    labeled: list[tuple] = []
    labeled += _windowed_snapshots(normal_rows, Label.BENIGN)
    labeled += _windowed_snapshots(probe_rows, Label.PORT_SCAN)
    return labeled


def _windowed_snapshots(rows, label: Label) -> list[tuple]:
    builder = WindowedGraphBuilder(window_seconds=2.0)
    labeled = []
    for i, row in enumerate(rows):
        assert row.origin == DataOrigin.REAL  # train/demo/test on real data only, enforced here
        ts = T0 + timedelta(milliseconds=i * ROW_SPACING_MS)
        snapshot = builder.ingest(flow_row_to_event(row, ts=ts))
        if snapshot is not None and snapshot.edges:
            labeled.append((snapshot, label))
    final = builder.flush(end_time=T0 + timedelta(milliseconds=len(rows) * ROW_SPACING_MS))
    if final is not None and final.edges:
        labeled.append((final, label))
    return labeled


def evaluate(model: DetectiveModel, labeled) -> dict:
    y_true, y_pred = [], []
    for snapshot, label in labeled:
        verdict = model.predict_verdict(snapshot, window_id=snapshot.window_id)
        y_true.append(label.value)
        # verdict.label is already a plain str here (KronusModel sets
        # use_enum_values=True), not an enum member — no .value to take.
        # collapse UNCERTAIN into its underlying concrete class for this
        # sweep-level metric — the policy layer, not accuracy reporting,
        # is where "uncertain" earns its own treatment (policy/response.rego)
        y_pred.append(verdict.label if verdict.label != Label.UNCERTAIN.value else "port_scan")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["benign", "port_scan"], average="macro", zero_division=0
    )
    accuracy = float(np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)]))
    return {"accuracy": round(accuracy, 4), "precision": round(float(precision), 4),
            "recall": round(float(recall), 4), "f1": round(float(f1), 4), "n": len(labeled)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20_000,
                         help="rows to replay (subset of 125,973 — see module docstring)")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading real NSL-KDD data (replaying up to {args.limit} rows)...")
    train_rows = load_nsl_kdd(TRAIN_PATH)
    test_rows = load_nsl_kdd(TEST_PATH)

    train_snapshots = build_labeled_snapshots(train_rows, limit=args.limit)
    test_snapshots = build_labeled_snapshots(test_rows, limit=args.limit)
    print(f"  train windows: {len(train_snapshots)}, test windows: {len(test_snapshots)}")
    label_counts = {lbl.value: sum(1 for _, l in train_snapshots if l == lbl)
                     for lbl in (Label.BENIGN, Label.PORT_SCAN)}
    print(f"  train label counts: {label_counts}")

    model = DetectiveModel(rng=np.random.default_rng(42))
    start = time.perf_counter()
    batch_size = 8
    for epoch in range(args.epochs):
        rng = np.random.default_rng(epoch)
        order = rng.permutation(len(train_snapshots))
        epoch_loss = 0.0
        n_batches = 0
        for start_idx in range(0, len(order), batch_size):
            batch_idx = order[start_idx:start_idx + batch_size]
            batch = [train_snapshots[i] for i in batch_idx]
            epoch_loss += model.train_batch(batch, learning_rate=0.02, momentum=0.9)
            n_batches += 1
        print(f"  epoch {epoch + 1}/{args.epochs}  avg_loss={epoch_loss / max(n_batches, 1):.4f}")
    train_seconds = time.perf_counter() - start
    print(f"  trained in {train_seconds:.1f}s (CPU)")

    print("Evaluating on held-out KDDTest+ windows...")
    metrics = evaluate(model, test_snapshots)
    metrics.update({
        "component": "detective",
        "task": "structural benign-vs-port_scan classification on graph windows",
        "trained_on": "NSL-KDD KDDTrain+ (real, normal-only and probe-only replay streams)",
        "evaluated_on": "NSL-KDD KDDTest+ (real, held-out file, unseen attack subtypes)",
        "note": (
            "Windows are built by replaying real normal rows and real probe "
            "rows as two separate streams (see build_labeled_snapshots) so "
            "each window is unambiguously one class — this measures the "
            "graph-structural separability of scan vs. normal topology, "
            "which is what the Detective exists to catch (features.txt "
            "Case 2). It is not a claim about mixed, ambiguous live traffic; "
            "see the integration flow-case tests and demo for end-to-end, "
            "unstratified evaluation of the full pipeline."
        ),
        "train_seconds": round(train_seconds, 2),
    })
    print(f"  {metrics}")

    model.save(MODEL_DIR)
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model + metrics to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
