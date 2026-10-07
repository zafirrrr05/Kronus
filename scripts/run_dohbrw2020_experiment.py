#!/usr/bin/env python3
"""CIRA-CIC-DoHBrw-2020 Independent ML Experiment for KRONUS.

EXPERIMENT: DoHBrw2020 External Dataset (DoH tunnel detection)
=============================================================
This script trains a fresh, independent KRONUS Detective using the
CIRA-CIC-DoHBrw-2020 capture (Canadian Institute for Cybersecurity, UNB).

Dataset: https://www.unb.ca/cic/datasets/dohbrw-2020.html
Paper:   MontazeriShatoori, Davidson, Dharmawansa, Habibi Lashkari,
         "Detection of DoH Tunnels using Time-series Classification of
         Encrypted Traffic", IEEE CyberSciTech 2020.

WHY ONLY THE DETECTIVE TRAINS HERE
----------------------------------
KRONUS's two lanes have fixed contracts. The Bouncer is strictly binary
flood-vs-benign: services/bouncer/model.py::predict_verdict can only emit
Label.FLOOD or Label.BENIGN, and its six features are volumetric
(event_rate, byte_rate, dest_port_entropy, unique_dest_count,
avg_duration_ms, same_dest_ratio). This dataset contains no DoS/flood
traffic at all — every capture is either benign DoH or a DNS tunnel — and a
DNS tunnel is not a flood. Training the Bouncer here would mean labelling
tunnel rows `flood`, which is simply false and would corrupt what the model
means. So the Bouncer is NOT trained on this dataset, and the metrics record
that as an explicit skip rather than an empty section.

The Detective's contract fits: RAW_CLASSES is already
[benign, port_scan, lateral_movement] (services/detective/model.py), and a
tunnelled/exfiltrating flow is exactly `lateral_movement`. So this is the
first experiment in the repo that exercises the Detective's third class.

LABEL MAP
---------
  Benign-DoH.csv   benign DoH        -> normal           -> Label.BENIGN
  DNSCat2-DoH.csv  dnscat2 tunnel    -> lateral_movement -> Label.LATERAL_MOVEMENT
  dns2tcp-DoH.csv  dns2tcp tunnel    -> lateral_movement -> Label.LATERAL_MOVEMENT
  iodine-DoH.csv   iodine tunnel     -> lateral_movement -> Label.LATERAL_MOVEMENT

NO RECONSTRUCTION (unlike Experiments A and C)
---------------------------------------------
This dataset carries real source/destination IPs, real ports and a real
capture clock, so nothing is synthesized: graph topology here is the
capture's own. The metrics record "synthetic_hosts": false. Rows are replayed
in the dataset's true chronological order (the loader sorts by TimeStamp) on
the same synthetic clock the other external experiments use.

EXPERIMENT DESIGN
-----------------
Models are trained INDEPENDENTLY from scratch -- zero weight inheritance.
Feature extraction and graph building use the identical production telemetry
pipeline (FlowFeaturizer / WindowedGraphBuilder) as every other experiment.

TRAIN/TEST SPLIT:
  Stratified 67/33 split by class, seeded for reproducibility.

WEIGHT ARTIFACTS:
  models/experiments/dohbrw2020/detective/ (detective.npz, detective.onnx)

METRICS:
  results/experiments/dohbrw2020_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

opa_bin = REPO_ROOT / "bin" / "opa"
if opa_bin.exists():
    os.environ["KRONUS_OPA_BINARY"] = str(opa_bin)
    os.environ["PATH"] = f"{REPO_ROOT / 'bin'}:{os.environ.get('PATH', '')}"

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from libs.constants import Label
from libs.schemas import GraphEdge, GraphNode, GraphSnapshot
from services.bouncer.features import FEATURE_NAMES
from services.detective.model import RAW_CLASSES, DetectiveModel
from services.graph_builder.builder import WindowedGraphBuilder
from services.telemetry_exporter.converters import flow_row_to_event
from twin.dohbrw2020 import load_dohbrw2020

DATA_DIR = REPO_ROOT / "data" / "external" / "dohbrw2020"
MODELS_DIR = REPO_ROOT / "models" / "experiments" / "dohbrw2020"
BOUNCER_DIR = MODELS_DIR / "bouncer"
DETECTIVE_DIR = MODELS_DIR / "detective"
RESULTS_DIR = REPO_ROOT / "results" / "experiments"
METRICS_PATH = RESULTS_DIR / "dohbrw2020_metrics.json"

DATASET_URL = "https://www.unb.ca/cic/datasets/dohbrw-2020.html"
DATASET_SOURCE = "CIRA-CIC-DoHBrw-2020 (Canadian Institute for Cybersecurity, UNB)"

T0 = datetime(2025, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
ROW_SPACING_MS = 20
TRAIN_RATIO = 0.67
SEED = 42

# The two classes the Detective is trained on here. port_scan is in the
# model's RAW_CLASSES but has no representatives in this dataset, so it is
# never a training target — it can still be *predicted*, and the confusion
# matrix below keeps it as a visible column rather than hiding it.
TRAINED_LABELS = ["benign", "lateral_movement"]

BOUNCER_SKIP_REASON = (
    "DoHBrw2020 contains no DoS/flood traffic — every flow is benign DoH or a "
    "DNS tunnel. The Bouncer's contract is strictly binary flood-vs-benign "
    "(predict_verdict emits only Label.FLOOD or Label.BENIGN), so training it "
    "here would require labelling tunnels as floods, which is false. Skipped "
    "deliberately; this experiment exercises the Detective's lateral_movement "
    "class instead."
)


def _build_graph_snapshots(rows: list, label: Label) -> list[tuple]:
    """Build windowed graph snapshots for GAT training.

    Matches the construction used by the existing external-dataset runners
    (scripts/run_unsw_nb15_experiment.py::_build_graph_snapshots) so results
    are directly comparable across experiments.
    """
    builder = WindowedGraphBuilder(window_seconds=2.0)
    labeled = []
    for i, row in enumerate(rows):
        ts = T0 + timedelta(milliseconds=i * ROW_SPACING_MS)
        snap = builder.ingest(flow_row_to_event(row, ts=ts))
        if snap is not None and snap.edges:
            labeled.append((snap, label))
    final_ts = T0 + timedelta(milliseconds=len(rows) * ROW_SPACING_MS)
    final = builder.flush(end_time=final_ts)
    if final is not None and final.edges:
        labeled.append((final, label))
    return labeled


def _stratified_split(items: list, ratio: float, seed: int) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items)).tolist()
    n_train = int(len(idx) * ratio)
    train_idx = set(idx[:n_train])
    train = [items[i] for i in range(len(items)) if i in train_idx]
    test = [items[i] for i in range(len(items)) if i not in train_idx]
    return train, test


def _bounded_sample(items: list, limit: int | None) -> list:
    """Cap `items` at `limit` by taking an evenly spaced stride, not a prefix.

    A prefix would be actively misleading on this dataset: the tunnel class is
    three separate captures concatenated in file order (DNSCat2, dns2tcp,
    iodine), so `[:limit]` would hand back one tool's traffic and call it
    "tunnel traffic". A stride spans every source file and the whole capture
    timeline within each, so the bounded sample stays representative. It is
    deterministic — no RNG — so the experiment reproduces exactly.
    """
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def skip_bouncer(rows: list) -> dict:
    """Record, honestly, why the Bouncer lane does not train on this dataset."""
    n_flood = sum(1 for r in rows if r.category == "dos")
    return {
        "component": "bouncer",
        "training_origin": "dohbrw2020",
        "skipped": BOUNCER_SKIP_REASON,
        "n_flood_rows": n_flood,
        "n_rows_considered": len(rows),
        "features": FEATURE_NAMES,
    }


def train_detective(rows: list, seed: int, limit: int | None = 20_000) -> tuple[DetectiveModel | None, dict]:
    """Train the Detective on DoHBrw2020 graph windows: benign vs lateral_movement."""
    benign_rows = [r for r in rows if r.category == "normal"]
    tunnel_rows = [r for r in rows if r.category == "lateral_movement"]

    if not benign_rows or not tunnel_rows:
        return None, {
            "component": "detective",
            "training_origin": "dohbrw2020",
            "skipped": "DoHBrw2020 subset lacks both benign and tunnel rows",
            "n_benign": len(benign_rows),
            "n_tunnels": len(tunnel_rows),
        }

    if limit is not None and limit > 0:
        benign_rows = _bounded_sample(benign_rows, limit)
        tunnel_rows = _bounded_sample(tunnel_rows, limit)

    benign_train, benign_test = _stratified_split(benign_rows, TRAIN_RATIO, seed)
    tunnel_train, tunnel_test = _stratified_split(tunnel_rows, TRAIN_RATIO, seed + 1)

    train_snaps = (
        _build_graph_snapshots(benign_train, Label.BENIGN) +
        _build_graph_snapshots(tunnel_train, Label.LATERAL_MOVEMENT)
    )
    test_snaps = (
        _build_graph_snapshots(benign_test, Label.BENIGN) +
        _build_graph_snapshots(tunnel_test, Label.LATERAL_MOVEMENT)
    )

    print(f"  Detective train windows: {len(train_snaps)}")
    print(f"  Detective test  windows: {len(test_snaps)}")

    if not train_snaps:
        return None, {
            "component": "detective",
            "training_origin": "dohbrw2020",
            "skipped": "No graph windows produced from DoHBrw2020 train rows",
        }

    t0 = time.perf_counter()
    model = DetectiveModel(rng=np.random.default_rng(seed))
    epochs, batch_size = 4, 4
    for ep in range(epochs):
        rng = np.random.default_rng(seed + ep)
        order = rng.permutation(len(train_snaps))
        ep_loss, nb = 0.0, 0
        for si in range(0, len(order), batch_size):
            batch = [train_snaps[i] for i in order[si : si + batch_size]]
            ep_loss += model.train_batch(batch, learning_rate=0.02, momentum=0.9)
            nb += 1
        print(f"  Epoch {ep + 1}/{epochs}  avg_loss={ep_loss / max(nb, 1):.4f}")
    train_time = time.perf_counter() - t0

    if not test_snaps:
        return model, {
            "component": "detective",
            "training_origin": "dohbrw2020",
            "skipped": "No graph windows produced from DoHBrw2020 test rows",
        }

    y_true, y_pred, n_uncertain = [], [], 0
    for snap, lbl in test_snaps:
        v = model.predict_verdict(snap, window_id=snap.window_id)
        y_true.append(lbl.value)
        if v.label == Label.UNCERTAIN.value:
            n_uncertain += 1
        # Same convention as Experiments B and C: a verdict in the gray zone
        # is counted toward the attack class. The count is reported so the
        # effect is visible rather than buried.
        y_pred.append(v.label if v.label != Label.UNCERTAIN.value else "lateral_movement")

    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)]))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=TRAINED_LABELS, average="macro", zero_division=0
    )
    # All three RAW_CLASSES as columns, so a port_scan prediction (a class with
    # no training examples here) shows up in the matrix instead of vanishing.
    cm = confusion_matrix(y_true, y_pred, labels=[label.value for label in RAW_CLASSES]).tolist()

    DETECTIVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(DETECTIVE_DIR)

    metrics = {
        "component": "detective",
        "training_origin": "dohbrw2020",
        "dataset_source": DATASET_SOURCE,
        "dataset_url": DATASET_URL,
        "model_type": "GAT in autograd (trained from scratch)",
        "task": "macro benign vs lateral_movement graph window classification",
        "synthetic_hosts": False,
        "seed": seed,
        "train_ratio": TRAIN_RATIO,
        "n_train_benign": len(benign_train),
        "n_train_tunnel": len(tunnel_train),
        "n_test_benign": len(benign_test),
        "n_test_tunnel": len(tunnel_test),
        "train_windows": len(train_snaps),
        "test_windows": len(test_snaps),
        "n_uncertain_verdicts": n_uncertain,
        "train_seconds": round(train_time, 3),
        "accuracy": round(acc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
        "confusion_matrix_labels": [label.value for label in RAW_CLASSES],
        "weight_artifacts": [
            str(DETECTIVE_DIR / "detective.npz"),
            str(DETECTIVE_DIR / "detective.onnx"),
        ],
    }
    print(f"  Detective (DoHBrw2020): Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    return model, metrics


def verify_weights(bouncer_trained: bool) -> dict:
    """Verify saved models load cleanly and execute correct inference."""
    results = {}

    if not bouncer_trained:
        results["dohbrw2020_bouncer_loadable"] = None
        results["dohbrw2020_bouncer_note"] = "not trained on this dataset — see bouncer.skipped"
        print("  DoHBrw2020 Bouncer load: SKIPPED (not trained on this dataset)")
    else:
        bouncer_ok = False
        try:
            from services.bouncer.model import BouncerModel

            b = BouncerModel.load(BOUNCER_DIR)
            feats = {n: 10.0 for n in FEATURE_NAMES}
            v = b.predict_verdict(feats, window_id="verify-dohbrw-bouncer")
            bouncer_ok = v is not None and 0.0 <= v.confidence <= 1.0
            print(f"  DoHBrw2020 Bouncer load: {'PASSED' if bouncer_ok else 'FAILED'} "
                  f"(verdict={v.label}, conf={v.confidence:.3f})")
        except Exception as exc:
            print(f"  DoHBrw2020 Bouncer load FAILED: {exc}")
        results["dohbrw2020_bouncer_loadable"] = bouncer_ok

    detective_ok = False
    try:
        d = DetectiveModel.load(DETECTIVE_DIR)
        now = datetime.now(timezone.utc)
        # Shaped like the real capture here: an internal host reaching a public
        # resolver over 443 (the topology twin/dohbrw2020.py actually produces).
        snap = GraphSnapshot(
            window_id="verify-dohbrw-detective",
            window_start=now,
            window_end=now,
            nodes=[
                GraphNode(node_id="192.168.20.207", degree_in=0.0, degree_out=2.0,
                          bytes_total=500.0, unique_ports_contacted=2),
                GraphNode(node_id="9.9.9.11", degree_in=2.0, degree_out=0.0,
                          bytes_total=500.0, unique_ports_contacted=0),
            ],
            edges=[
                GraphEdge(src="192.168.20.207", dst="9.9.9.11", bytes=500.0,
                          flow_count=2, port_entropy=0.5, duration_mean_ms=15.0),
            ],
        )
        v = d.predict_verdict(snap, window_id="verify-dohbrw-detective")
        detective_ok = v is not None and 0.0 <= v.confidence <= 1.0
        print(f"  DoHBrw2020 Detective load: {'PASSED' if detective_ok else 'FAILED'} "
              f"(verdict={v.label}, conf={v.confidence:.3f})")
    except Exception as exc:
        print(f"  DoHBrw2020 Detective load FAILED: {exc}")
    results["dohbrw2020_detective_loadable"] = detective_ok
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10_000,
                        help="max rows per class for the Detective window pools")
    parser.add_argument("--sample-per-file", type=int, default=15_000,
                        help="max rows read per CSV (streamed; bounds RAM)")
    args = parser.parse_args()

    print("=" * 70)
    print("CIRA-CIC-DoHBrw-2020 INDEPENDENT TRAINING & EVALUATION")
    print("=" * 70)

    if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.csv")):
        print(f"Dataset directory not found or empty: {DATA_DIR}")
        print("Run: python scripts/download_dohbrw2020.py --from-mirror")
        return 1

    print(f"Loading DoHBrw2020 flows from {DATA_DIR} ...")
    t0 = time.perf_counter()
    rows = load_dohbrw2020(DATA_DIR, sample_per_file=args.sample_per_file)
    load_time = time.perf_counter() - t0

    cats: dict[str, int] = {}
    for r in rows:
        cats[r.category] = cats.get(r.category, 0) + 1
    print(f"  Loaded {len(rows):,} flows in {load_time:.2f}s")
    print(f"  Category distribution: {cats}")

    if not rows:
        print("No rows loaded — nothing to train on. Aborting without writing metrics.")
        return 1

    print("\n--- Bouncer on DoHBrw2020: deliberately SKIPPED ---")
    print(f"  {BOUNCER_SKIP_REASON}")
    bouncer_metrics = skip_bouncer(rows)

    print("\n--- Training Detective (GAT in autograd) on DoHBrw2020 ---")
    detective_model, detective_metrics = train_detective(rows, seed=SEED, limit=args.limit)

    print("\n--- Verifying Weight Loadability & Live Inference ---")
    weight_verification = verify_weights(bouncer_trained=False)

    full_metrics = {
        "experiment": "dohbrw2020",
        "status": "COMPLETE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Detective-only experiment by design: DoHBrw2020 has no DoS/flood "
            "class, so the Bouncer's binary flood-vs-benign contract cannot be "
            "honestly trained here. See bouncer.skipped."
        ),
        "dataset": {
            "source": DATASET_SOURCE,
            "url": DATASET_URL,
            "total_flows": len(rows),
            "category_distribution": cats,
            "synthetic_hosts": False,
            "synthetic_hosts_note": (
                "No reconstruction: this capture carries real source/destination "
                "IPs and real ports, so graph topology is the capture's own. Rows "
                "are replayed in the dataset's true chronological order."
            ),
            "features_are_real": True,
        },
        "bouncer": bouncer_metrics,
        "detective": detective_metrics,
        "weight_verification": weight_verification,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(full_metrics, indent=2))

    print("\n" + "=" * 70)
    print("DoHBrw2020 EXPERIMENT COMPLETE")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Detective weights: {DETECTIVE_DIR}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
