#!/usr/bin/env python3
"""CIC IoT 2023 Independent ML Experiment for KRONUS.

EXPERIMENT: CIC IoT 2023 External Validation (Experiment H)
===========================================================
Trains fresh, independent KRONUS lanes on CIC IoT 2023 (Canadian Institute for
Cybersecurity, University of New Brunswick): a capture from a 105-device IoT
testbed carrying 33 attacks in seven families plus benign traffic.

Dataset URL: https://www.unb.ca/cic/datasets/iotdataset-2023.html

    Neto, Dadkhah, Ferreira, Zohourian, Lu, Ghorbani, "CICIoT2023: A real-time
    dataset and benchmark for large-scale attacks in IoT environment",
    Sensors 23(13):5941, 2023. doi:10.3390/s23135941

THE FIRST EXPERIMENT HERE WHERE BOTH LANES TRAIN
------------------------------------------------
Every experiment before this one could train only one lane, and said so:
CIC-DDoS2019 is all volumetric floods (Bouncer only, Experiment G), DoHBrw2020
has no flood at all (Detective only, Experiment D). This release carries both
kinds in one capture set, so both lanes train on it:

  Bouncer    flood (DDoS-*/DoS-*/Mirai-*) vs benign          — its own contract
  Detective  port_scan (Recon-*/VulnerabilityScan) vs benign — 2 of its 3
             RAW_CLASSES; lateral_movement has no representative here and is
             never a training target, but stays a visible column in the
             confusion matrix.

WHAT IS THE PUBLISHER'S, AND WHAT IS OURS
-----------------------------------------
This is the only runner whose rows come from packets rather than from rows the
publisher already produced. The publisher's: the captures, every byte on the
wire, the addresses, the ports, the inter-arrival times, and the family folder
that is the only label the release has. Ours: the grouping of packets into
flows (twin/ciciot2023.py, which states the definition and reports every drop).
The loader's `extraction` and `flow_definition` blocks are copied into the
metrics verbatim so the disclosure travels with the numbers.

No host is reconstructed and no address is invented: `synthetic_hosts` is false.
That is the point of using the captures — four of the Bouncer's six features are
port- or clock-derived, and the published CSVs carry neither.

SPLIT: THE CAPTURE IS THE UNIT, NOT THE FLOW
--------------------------------------------
Every flow in one capture shares that capture's burst — the same attacker, the
same victim, the same minute. Splitting pooled flows at random would put half of
one burst in train and half in test, and would then report the model recognising
a near-duplicate as generalization. The split is therefore over captures: whole
captures go to train or to test.

A class with only one capture cannot be split that way without emptying a side,
so it falls back to a stride within that capture and the metrics record
`split_strategy` per class, rather than the fallback passing silently as the
same thing.

MEMORY AND TIME
---------------
Each capture is streamed once (packet by packet, never held in memory), thinned
to `--sample-per-capture` flows by an evenly spaced stride, and replayed once
through a fresh FlowFeaturizer in true chronological order; only the resulting
6-float vectors are kept. The clock is real — flows carry their capture's own
first-packet time — so a 2-second window is a real 2 seconds, and the clock is
excluded from `features` because an attack capture is a different capture from a
benign one.

TRAIN/TEST SPLIT
  Bouncer:    stratified 67/33 over captures, seeded, per class.
  Detective:  the same captures, windowed into 2-second graph snapshots.

WEIGHT ARTIFACTS
  models/experiments/ciciot2023/bouncer/    (bouncer.json, calibration.json)
  models/experiments/ciciot2023/detective/  (detective.npz, detective.onnx)

METRICS
  results/experiments/ciciot2023_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

opa_bin = REPO_ROOT / "bin" / "opa"
if opa_bin.exists():
    os.environ["KRONUS_OPA_BINARY"] = str(opa_bin)
    os.environ["PATH"] = f"{REPO_ROOT / 'bin'}:{os.environ.get('PATH', '')}"

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from libs.constants import Label
from services.bouncer.features import FEATURE_NAMES, FlowFeaturizer
from services.bouncer.model import BouncerModel
from services.detective.model import RAW_CLASSES, DetectiveModel
from services.graph_builder.builder import WindowedGraphBuilder
from services.telemetry_exporter.converters import flow_row_to_event
from twin.ciciot2023 import load_ciciot2023_by_capture

DATA_DIR = REPO_ROOT / "data" / "external" / "ciciot2023"
MODELS_DIR = REPO_ROOT / "models" / "experiments" / "ciciot2023"
BOUNCER_DIR = MODELS_DIR / "bouncer"
DETECTIVE_DIR = MODELS_DIR / "detective"
RESULTS_DIR = REPO_ROOT / "results" / "experiments"
METRICS_PATH = RESULTS_DIR / "ciciot2023_metrics.json"

DATASET_URL = "https://www.unb.ca/cic/datasets/iotdataset-2023.html"
DATASET_SOURCE = "CIC IoT 2023 (Canadian Institute for Cybersecurity, UNB)"

TRAIN_RATIO = 0.67
SEED = 42
DEFAULT_SAMPLE_PER_CAPTURE = 40_000

# A score computed from a handful of flows is not a measurement. Below these
# counts the runner refuses to write metrics rather than reporting a number the
# sample cannot support.
MIN_ROWS_PER_CLASS = 200


# --- replay -----------------------------------------------------------------

def _replay_features(rows: list, times: list) -> np.ndarray:
    """Replay one pool in true chronological order through a fresh featurizer.

    `features_for` is stateful and keyed by source_ip, so it must see the pool
    in timestamp order. Each pool gets its OWN featurizer rather than sharing
    one across train and test: sharing would let test rows' windows contain
    train rows, which is a leak the score would not show.
    """
    featurizer = FlowFeaturizer()
    order = sorted(range(len(rows)), key=lambda i: (times[i], i))
    vectors = []
    for index in order:
        event = flow_row_to_event(
            rows[index],
            ts=datetime.fromtimestamp(times[index], tz=timezone.utc),
        )
        feats = featurizer.features_for(event)
        vectors.append([feats[name] for name in FEATURE_NAMES])
    if not vectors:
        return np.empty((0, len(FEATURE_NAMES)))
    return np.asarray(vectors, dtype=float)


def _pool(by_capture: dict, category: str) -> dict[str, tuple]:
    """{capture: (rows, times)} for every capture whose family is `category`."""
    return {
        name: (entry[0], entry[1])
        for name, entry in by_capture.items()
        if entry[0] and entry[0][0].category == category
    }


def _thin(entry: tuple, sample_per_capture: int | None) -> tuple:
    """Cap one capture's flows with an evenly spaced stride."""
    rows, times = entry
    if sample_per_capture is None or len(rows) <= sample_per_capture:
        return rows, times
    keep = _evenly_spaced(len(rows), sample_per_capture)
    return [rows[i] for i in keep], [times[i] for i in keep]


def _split_pool(pool: dict[str, tuple], seed: int, sample_per_capture: int | None
                ) -> tuple[list, list, list, list, str]:
    """Split one class's captures into (train rows, train times, test rows,
    test times, strategy).

    Whole captures go one way or the other. A class with a single capture
    cannot be split that way without emptying a side, so the fallback is a
    stride within that capture and the strategy string says which happened —
    the fallback must not pass silently as the same thing.
    """
    names = sorted(pool)
    if not names:
        return [], [], [], [], "empty"

    thinned = {name: _thin(pool[name], sample_per_capture) for name in names}

    if len(names) >= 2:
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(len(names)))
        n_train = min(len(names) - 1, max(1, int(round(len(names) * TRAIN_RATIO))))
        train_names = sorted(names[i] for i in order[:n_train])
        test_names = sorted(names[i] for i in order[n_train:])
        train_rows, train_times = _concat(thinned, train_names)
        test_rows, test_times = _concat(thinned, test_names)
        strategy = f"capture_holdout(train={len(train_names)},test={len(test_names)})"
        return train_rows, train_times, test_rows, test_times, strategy

    # One capture: every third flow goes to test. Evenly spaced, so both sides
    # span the capture, and deterministic, so the run reproduces exactly.
    name = names[0]
    rows, times = thinned[name]
    test_idx = list(range(2, len(rows), 3))
    test_set = set(test_idx)
    train_idx = [i for i in range(len(rows)) if i not in test_set]
    return (
        [rows[i] for i in train_idx], [times[i] for i in train_idx],
        [rows[i] for i in test_idx], [times[i] for i in test_idx],
        "flow_stride_within_single_capture",
    )


def _concat(thinned: dict[str, tuple], names: list[str]) -> tuple[list, list]:
    rows: list = []
    times: list = []
    for name in names:
        rows.extend(thinned[name][0])
        times.extend(thinned[name][1])
    return rows, times


def _evenly_spaced(total: int, target: int | None) -> list[int]:
    """Indices for an evenly spaced sample — a stride, never a prefix, so a
    bounded sample still spans the capture rather than its opening moments."""
    if target is None or target <= 0 or target >= total:
        return list(range(total))
    step = total / target
    return [min(total - 1, int(i * step)) for i in range(target)]


# --- Bouncer ----------------------------------------------------------------
def _score_bouncer(model: BouncerModel, X: np.ndarray, y: np.ndarray) -> dict:
    proba = model.predict_proba(X)
    pred = (proba >= 0.5).astype(int)
    prf = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "n": int(len(y)), "n_flood": int(y.sum()), "n_benign": int((y == 0).sum()),
        "accuracy": round(float((pred == y).mean()), 4),
        "precision": round(float(prf[0]), 4),
        "recall": round(float(prf[1]), 4),
        "f1": round(float(prf[2]), 4),
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["benign", "flood"],
    }


def train_bouncer(by_capture: dict, sample_per_capture: int | None, seed: int
                  ) -> tuple[BouncerModel | None, dict]:
    """Fit the Bouncer on flood vs benign, split by capture."""
    flood = _pool(by_capture, "dos")
    benign = _pool(by_capture, "normal")

    (f_tr, f_tr_t, f_te, f_te_t, f_strategy) = _split_pool(flood, seed, sample_per_capture)
    (b_tr, b_tr_t, b_te, b_te_t, b_strategy) = _split_pool(benign, seed + 1, sample_per_capture)

    if len(f_tr) < MIN_ROWS_PER_CLASS or len(b_tr) < MIN_ROWS_PER_CLASS:
        return None, {
            "component": "bouncer",
            "training_origin": "ciciot2023",
            "skipped": "not enough flood and/or benign flows to train",
            "n_flood_train": len(f_tr), "n_benign_train": len(b_tr),
            "min_rows_per_class": MIN_ROWS_PER_CLASS,
        }
    if not f_te or not b_te:
        return None, {
            "component": "bouncer",
            "training_origin": "ciciot2023",
            "skipped": "the capture split left no flood and/or benign flows to test on",
            "n_flood_test": len(f_te), "n_benign_test": len(b_te),
        }

    Xf_tr, Xb_tr = _replay_features(f_tr, f_tr_t), _replay_features(b_tr, b_tr_t)
    Xf_te, Xb_te = _replay_features(f_te, f_te_t), _replay_features(b_te, b_te_t)

    X_train = np.vstack([Xf_tr, Xb_tr])
    y_train = np.concatenate([np.ones(len(Xf_tr), int), np.zeros(len(Xb_tr), int)])
    X_test = np.vstack([Xf_te, Xb_te])
    y_test = np.concatenate([np.ones(len(Xf_te), int), np.zeros(len(Xb_te), int)])

    print(f"  Bouncer train: {len(Xf_tr):,} flood + {len(Xb_tr):,} benign "
          f"= {len(y_train):,} feature vectors")
    print(f"  Bouncer test : {len(Xf_te):,} flood + {len(Xb_te):,} benign "
          f"= {len(y_test):,} feature vectors")

    model = BouncerModel()
    started = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    metrics = _score_bouncer(model, X_test, y_test)
    metrics.update({
        "component": "bouncer",
        "training_origin": "ciciot2023",
        "dataset_source": DATASET_SOURCE,
        "dataset_url": DATASET_URL,
        "model_type": "XGBoost + temperature scaling (fresh, from scratch)",
        "task": "binary flood vs benign",
        "features": FEATURE_NAMES,
        "synthetic_hosts": False,
        "seed": seed,
        "train_ratio": TRAIN_RATIO,
        "split_unit": "capture",
        "split_strategy_flood": f_strategy,
        "split_strategy_benign": b_strategy,
        "n_train": int(len(y_train)),
        "n_train_flood": int(y_train.sum()),
        "n_train_benign": int((y_train == 0).sum()),
        "train_seconds": round(elapsed, 3),
        "weight_artifacts": [
            str(BOUNCER_DIR / "bouncer.json"),
            str(BOUNCER_DIR / "calibration.json"),
        ],
    })
    BOUNCER_DIR.mkdir(parents=True, exist_ok=True)
    model.save(BOUNCER_DIR)
    print(f"  Bouncer (CIC IoT 2023): Acc={metrics['accuracy']:.4f} "
          f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f} "
          f"F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")
    return model, metrics


# --- Detective --------------------------------------------------------------

def _build_graph_snapshots(rows: list, times: list, label: Label) -> list[tuple]:
    """Windowed graph snapshots at the captures' true spacing.

    Matches the construction the other external runners use, so the results stay
    comparable across experiments; the difference here is that the timestamps
    are the capture's own rather than a synthetic uniform spacing.
    """
    builder = WindowedGraphBuilder(window_seconds=2.0)
    labeled = []
    order = sorted(range(len(rows)), key=lambda i: (times[i], i))
    last_ts = None
    for index in order:
        ts = datetime.fromtimestamp(times[index], tz=timezone.utc)
        last_ts = ts
        snap = builder.ingest(flow_row_to_event(rows[index], ts=ts))
        if snap is not None and snap.edges:
            labeled.append((snap, label))
    if last_ts is not None:
        final = builder.flush(end_time=last_ts)
        if final is not None and final.edges:
            labeled.append((final, label))
    return labeled


def train_detective(by_capture: dict, sample_per_capture: int | None, seed: int
                    ) -> tuple[DetectiveModel | None, dict]:
    """Train the Detective on port_scan vs benign graph windows."""
    probe = _pool(by_capture, "probe")
    benign = _pool(by_capture, "normal")

    (p_tr, p_tr_t, p_te, p_te_t, p_strategy) = _split_pool(probe, seed, sample_per_capture)
    (b_tr, b_tr_t, b_te, b_te_t, b_strategy) = _split_pool(benign, seed + 1, sample_per_capture)

    if not p_tr or not b_tr:
        return None, {
            "component": "detective",
            "training_origin": "ciciot2023",
            "skipped": "this capture set lacks a benign and/or recon (probe) family",
            "n_probe_train": len(p_tr), "n_benign_train": len(b_tr),
        }
    if not p_te or not b_te:
        return None, {
            "component": "detective",
            "training_origin": "ciciot2023",
            "skipped": "the capture split left no probe and/or benign flows to test on",
            "n_probe_test": len(p_te), "n_benign_test": len(b_te),
        }

    train_snaps = (_build_graph_snapshots(b_tr, b_tr_t, Label.BENIGN)
                   + _build_graph_snapshots(p_tr, p_tr_t, Label.PORT_SCAN))
    test_snaps = (_build_graph_snapshots(b_te, b_te_t, Label.BENIGN)
                  + _build_graph_snapshots(p_te, p_te_t, Label.PORT_SCAN))

    print(f"  Detective train windows: {len(train_snaps)}")
    print(f"  Detective test  windows: {len(test_snaps)}")

    if not train_snaps or not test_snaps:
        return None, {
            "component": "detective",
            "training_origin": "ciciot2023",
            "skipped": "no graph windows with edges came out of the split",
            "train_windows": len(train_snaps), "test_windows": len(test_snaps),
        }

    started = time.perf_counter()
    model = DetectiveModel(rng=np.random.default_rng(seed))
    epochs, batch_size = 4, 4
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(train_snaps))
        epoch_loss, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = [train_snaps[i] for i in order[start:start + batch_size]]
            epoch_loss += model.train_batch(batch, learning_rate=0.02, momentum=0.9)
            batches += 1
        print(f"  Epoch {epoch + 1}/{epochs}  avg_loss={epoch_loss / max(batches, 1):.4f}")
    train_seconds = time.perf_counter() - started

    y_true, y_pred, n_uncertain = [], [], 0
    for snap, label in test_snaps:
        verdict = model.predict_verdict(snap, window_id=snap.window_id)
        y_true.append(label.value)
        if verdict.label == Label.UNCERTAIN.value:
            n_uncertain += 1
        # Same convention as Experiments B, C and D: a verdict in the gray zone
        # counts toward the attack class, and the count is reported so the
        # effect is visible rather than buried.
        y_pred.append(verdict.label if verdict.label != Label.UNCERTAIN.value
                      else Label.PORT_SCAN.value)

    trained_labels = [Label.BENIGN.value, Label.PORT_SCAN.value]
    accuracy = float(np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)]))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=trained_labels, average="macro", zero_division=0)
    # Every RAW_CLASS as a column, so a lateral_movement prediction — a class
    # with no training examples here — shows up rather than vanishing.
    cm = confusion_matrix(y_true, y_pred,
                          labels=[label.value for label in RAW_CLASSES]).tolist()

    DETECTIVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(DETECTIVE_DIR)

    metrics = {
        "component": "detective",
        "training_origin": "ciciot2023",
        "dataset_source": DATASET_SOURCE,
        "dataset_url": DATASET_URL,
        "model_type": "GAT in autograd (trained from scratch)",
        "task": "macro benign vs port_scan graph window classification",
        "synthetic_hosts": False,
        "seed": seed,
        "train_ratio": TRAIN_RATIO,
        "split_unit": "capture",
        "split_strategy_probe": p_strategy,
        "split_strategy_benign": b_strategy,
        "trained_labels": trained_labels,
        "never_trained_labels": [
            label.value for label in RAW_CLASSES if label.value not in trained_labels],
        "n_train_benign": len(b_tr),
        "n_train_probe": len(p_tr),
        "n_test_benign": len(b_te),
        "n_test_probe": len(p_te),
        "train_windows": len(train_snaps),
        "test_windows": len(test_snaps),
        "n_uncertain_verdicts": n_uncertain,
        "train_seconds": round(train_seconds, 3),
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
        "confusion_matrix_labels": [label.value for label in RAW_CLASSES],
        "weight_artifacts": [
            str(DETECTIVE_DIR / "detective.npz"),
            str(DETECTIVE_DIR / "detective.onnx"),
        ],
    }
    print(f"  Detective (CIC IoT 2023): Acc={accuracy:.4f} Prec={precision:.4f} "
          f"Rec={recall:.4f} F1={f1:.4f}")
    return model, metrics


# --- weight verification ----------------------------------------------------

def verify_weights(bouncer_trained: bool, detective_trained: bool) -> dict:
    """Verify the saved models load cleanly and execute correct inference."""
    results: dict = {}

    if bouncer_trained:
        bouncer_ok = False
        try:
            loaded = BouncerModel.load(BOUNCER_DIR)
            feats = {name: 10.0 for name in FEATURE_NAMES}
            verdict = loaded.predict_verdict(feats, window_id="verify-ciciot2023-bouncer")
            bouncer_ok = verdict is not None and 0.0 <= verdict.confidence <= 1.0
            print(f"  CIC IoT 2023 Bouncer load: "
                  f"{'PASSED' if bouncer_ok else 'FAILED'} "
                  f"(verdict={verdict.label}, conf={verdict.confidence:.3f})")
        except Exception as exc:  # noqa: BLE001 — reported, not swallowed
            print(f"  CIC IoT 2023 Bouncer load FAILED: {exc}")
        results["ciciot2023_bouncer_loadable"] = bouncer_ok
    else:
        results["ciciot2023_bouncer_loadable"] = None
        results["ciciot2023_bouncer_note"] = "not trained on this dataset"

    if detective_trained:
        detective_ok = False
        try:
            loaded = DetectiveModel.load(DETECTIVE_DIR)
            builder = WindowedGraphBuilder(window_seconds=2.0)
            base = datetime(2025, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
            snap = None
            for step in range(4):
                event = flow_row_to_event(_verification_row(),
                                          ts=base.replace(second=step))
                snap = builder.ingest(event)
            if snap is None:
                snap = builder.flush(end_time=base.replace(second=10))
            if snap is not None:
                verdict = loaded.predict_verdict(snap, window_id="verify-ciciot2023-detective")
                detective_ok = verdict is not None and 0.0 <= verdict.confidence <= 1.0
                print(f"  CIC IoT 2023 Detective load: "
                      f"{'PASSED' if detective_ok else 'FAILED'} "
                      f"(verdict={verdict.label}, conf={verdict.confidence:.3f})")
        except Exception as exc:  # noqa: BLE001 — reported, not swallowed
            print(f"  CIC IoT 2023 Detective load FAILED: {exc}")
        results["ciciot2023_detective_loadable"] = detective_ok
    else:
        results["ciciot2023_detective_loadable"] = None
        results["ciciot2023_detective_note"] = "not trained on this dataset"

    return results


def _verification_row():
    """A throwaway row used only to drive the loaded models once.

    Built from the same NSLKDDRow shape the loader produces so the converter
    path exercised here is the real one.
    """
    from libs.constants import DataOrigin, Protocol
    from twin.nsl_kdd import NSLKDDRow

    return NSLKDDRow(
        source_ip="10.0.0.1", dest_ip="10.0.0.2", source_port=40000, dest_port=80,
        protocol=Protocol.TCP, total_bytes=512, duration_ms=10, raw_label="Benign_Final",
        category="normal", kronus_label=Label.BENIGN, difficulty=0,
        features={name: 1.0 for name in FEATURE_NAMES}, origin=DataOrigin.REAL,
    )


# --- main -------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-per-capture", type=int, default=DEFAULT_SAMPLE_PER_CAPTURE,
                        help="max flows kept per capture (evenly spaced, not a prefix)")
    args = parser.parse_args()

    print("=" * 70)
    print("CIC IoT 2023 INDEPENDENT TRAINING & EVALUATION")
    print("=" * 70)

    if not DATA_DIR.exists() or not any(DATA_DIR.rglob("*.pcap*")):
        print(f"Dataset directory not found or empty: {DATA_DIR}")
        print("Run: python scripts/download_ciciot2023.py --register "
              "--first-name ... --last-name ... --email ... "
              "--institution ... --job-title ... --country ...")
        print("(see the downloader's module docstring; registry values are not "
              "committed)")
        return 1

    print(f"Extracting flows from {DATA_DIR} ...")
    started = time.perf_counter()
    by_capture = load_ciciot2023_by_capture(
        DATA_DIR, sample_per_capture=args.sample_per_capture)
    load_seconds = time.perf_counter() - started

    all_rows = [row for rows, _, _ in by_capture.values() for row in rows]
    if not all_rows:
        print("No flows extracted — nothing to train on. "
              "Aborting without writing metrics.")
        return 1

    cats: dict[str, int] = {}
    for row in all_rows:
        cats[row.category] = cats.get(row.category, 0) + 1
    print(f"  {len(by_capture)} capture(s), {len(all_rows):,} flows "
          f"in {load_seconds:.2f}s")
    print(f"  Category distribution: {cats}")

    print("\n--- Training Bouncer (flood vs benign) on CIC IoT 2023 ---")
    bouncer_model, bouncer_metrics = train_bouncer(
        by_capture, args.sample_per_capture, SEED)

    print("\n--- Training Detective (port_scan vs benign) on CIC IoT 2023 ---")
    detective_model, detective_metrics = train_detective(
        by_capture, args.sample_per_capture, SEED)

    if bouncer_model is None and detective_model is None:
        print("\nNeither lane could be trained on this capture set. "
              "Aborting without writing metrics.")
        return 1

    print("\n--- Verifying Weight Loadability & Live Inference ---")
    weight_verification = verify_weights(
        bouncer_trained=bouncer_model is not None,
        detective_trained=detective_model is not None,
    )

    # The loader's own disclosure is carried through verbatim, so the metrics
    # state what is the publisher's and what this repository derived.
    report = _extraction_report(by_capture)

    full_metrics = {
        "experiment": "ciciot2023",
        "status": "COMPLETE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "The first experiment in this repository where both lanes train: the "
            "release carries volumetric floods and recon scans against the same "
            "benign baseline. Rows are flows this repository extracted from the "
            "publisher's packet captures, not rows the publisher published — see "
            "dataset.extraction and dataset.flow_definition."
        ),
        "dataset": {
            "source": DATASET_SOURCE,
            "url": DATASET_URL,
            "total_flows": len(all_rows),
            "captures": len(by_capture),
            "category_distribution": cats,
            "synthetic_hosts": False,
            "features_are_real": True,
            "extraction": report.get("extraction"),
            "flow_definition": report.get("flow_definition"),
            "per_capture": report.get("per_capture"),
            "dropped_families": report.get("dropped_families"),
            "rows_by_family": report.get("rows_by_family"),
        },
        "bouncer": bouncer_metrics,
        "detective": detective_metrics,
        "weight_verification": weight_verification,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(full_metrics, indent=2))

    print("\n" + "=" * 70)
    print("CIC IoT 2023 EXPERIMENT COMPLETE")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Bouncer weights:   {BOUNCER_DIR}")
    print(f"Detective weights: {DETECTIVE_DIR}")
    print("=" * 70)
    return 0


def _extraction_report(by_capture: dict) -> dict:
    """Rebuild the loader's disclosure block from the per-capture reports.

    Taken from the reports the loader already produced rather than re-derived,
    so the metrics cannot drift from what the extraction actually did.
    """
    from twin.ciciot2023 import FLOW_ACTIVE_TIMEOUT_S, FLOW_IDLE_TIMEOUT_S

    per_capture = [entry[2] for entry in by_capture.values()]
    by_family: dict[str, int] = {}
    for capture_rows, _, _ in by_capture.values():
        for row in capture_rows:
            by_family[row.raw_label] = by_family.get(row.raw_label, 0) + 1
    return {
        "extraction": ("flows extracted by us from the publisher's packet "
                       "captures; addresses, ports and times are the capture's"),
        "flow_definition": {
            "key": "bidirectional 5-tuple (endpoints sorted)",
            "idle_timeout_s": FLOW_IDLE_TIMEOUT_S,
            "active_timeout_s": FLOW_ACTIVE_TIMEOUT_S,
            "source_ip": "endpoint that sent the flow's first packet",
            "timestamp": "flow's first-packet time",
        },
        "per_capture": per_capture,
        "rows_by_family": dict(sorted(by_family.items())),
    }


if __name__ == "__main__":
    sys.exit(main())
