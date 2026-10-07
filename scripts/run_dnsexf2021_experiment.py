#!/usr/bin/env python3
"""CIC-Bell-DNS-EXF-2021 Independent ML Experiment for KRONUS.

EXPERIMENT: DNS-EXF2021 External Dataset (DNS exfiltration / tunnelling)
=======================================================================
This script trains a fresh, independent KRONUS Detective using the
CIC-Bell-DNS-EXF-2021 corpus (Canadian Institute for Cybersecurity, UNB).

Dataset: https://www.unb.ca/cic/datasets/dns-exf-2021.html
Paper:   Samaneh Mahdavifar, Ali A. Ghorbani, "DeNAs: Deep Network Attack
         Signature and its Application to DNS Exfiltration Detection", 2021.

WHAT THIS EXPERIMENT CONCLUDES
------------------------------
No valid detection metric can be produced from this dataset, and the metrics
file says exactly that rather than quoting a number. Two measured facts:

1. THE LABELS ARE NOT RECOVERABLE FROM THE FEATURES. 536,138 rows collapse to
   46,462 distinct feature vectors (91.3% exact duplicates), and 64 vectors
   carry two different KRONUS classes, involving 390,215 rows. The hard ceiling
   on accuracy for ANY model reading these features is 0.8210 — below the
   repository's own external-dataset SLO of F1 > 0.85. The same base32
   exfiltration payload FHEPFCELEHFCEPFFFACACACACACACABN carries ONE identical
   feature vector under both labels. The cause is capture-level labelling:
   "this capture contained an exfiltration run" is stamped onto every row of
   the capture, including the ordinary lookups the machine made while the
   attack ran.

2. THE GUARD LEAVES ALMOST NO ATTACK CLASS. Of 294,353 exfiltration rows,
   294,204 — 99.95% — sit on a vector that also carries a benign label.
   Removing those (which keeps the experiment from measuring label noise)
   leaves the attack class with 149 rows carrying only 32 distinct signatures,
   against 46,366 signatures of benign traffic. De-duplication then collapses
   those 149 rows to the 32 the experiment actually trains on, against 46,366
   benign rows.

So the pipeline IS run — it trains, saves, and is verified loadable — but its
score is marked `reportable_as_detection: false`. On a 32-signature minority
class it measures a lookup table, not detection, and this repository will not
present it as performance. `label_quality` carries the measurements, and
`verdict` is NO_VALID_DETECTION_METRIC.

There is a second, independent problem: the dataset has NO IPs, PORTS, BYTES
OR DURATIONS. Every quantity the KRONUS lanes consume is therefore
reconstructed, and the one genuinely discriminating real signal (query-name
entropy) is not something either lane reads. What reaches the Detective is:
destination identity rebuilt from the real queried domain (`sld`), byte counts
derived from the real query-name length, and a constant source and port.

TWO PRECONDITIONS THAT KEEP THE EXPERIMENT FROM BEING ACTIVELY FALSE
-------------------------------------------------------------------
`twin/dnsexf2021.py` drops every feature vector carrying more than one class,
then keeps one row per distinct vector. Without the first, the score would
partly measure the label noise itself; without the second, 57,469 copies of a
single observation would be weighted as 57,469 observations and identical rows
would land in both splits. Both are default-on and both are disclosed here.

WHY ONLY THE DETECTIVE TRAINS
-----------------------------
Same reasoning as scripts/run_dohbrw2020_experiment.py: the Bouncer's contract
is strictly binary flood-vs-benign, and this dataset contains no DoS/flood
traffic at all (every row is either benign DNS or an exfiltration channel).
Labelling exfiltration as a flood to make the Bouncer train would be false, so
it is skipped and the metrics record why.

LABEL MAP
---------
  benign_labeled/        -> normal           -> Label.BENIGN
  heavy_attack_labeled/  -> lateral_movement -> Label.LATERAL_MOVEMENT
  light_attack_labeled/  -> lateral_movement -> Label.LATERAL_MOVEMENT

EXPERIMENT DESIGN
-----------------
Models are trained INDEPENDENTLY from scratch -- zero weight inheritance.
Feature extraction and graph building use the identical production telemetry
pipeline (WindowedGraphBuilder) as every other experiment. Rows are replayed
in the dataset's true chronological order on the same synthetic clock.

TRAIN/TEST SPLIT:
  Stratified 67/33 split by class, seeded for reproducibility.

WEIGHT ARTIFACTS:
  models/experiments/dnsexf2021/detective/ (detective.npz, detective.onnx)

METRICS:
  results/experiments/dnsexf2021_metrics.json
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
from twin.dnsexf2021 import ambiguity_report, load_dnsexf2021

DATA_DIR = REPO_ROOT / "data" / "external" / "dnsexf2021"
MODELS_DIR = REPO_ROOT / "models" / "experiments" / "dnsexf2021"
BOUNCER_DIR = MODELS_DIR / "bouncer"
DETECTIVE_DIR = MODELS_DIR / "detective"
RESULTS_DIR = REPO_ROOT / "results" / "experiments"
METRICS_PATH = RESULTS_DIR / "dnsexf2021_metrics.json"

DATASET_URL = "https://www.unb.ca/cic/datasets/dns-exf-2021.html"
DATASET_SOURCE = "CIC-Bell-DNS-EXF-2021 (Canadian Institute for Cybersecurity, UNB)"

T0 = datetime(2025, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
ROW_SPACING_MS = 20
TRAIN_RATIO = 0.67
SEED = 42

# The repository's own SLO for an external dataset. Stated here so the result
# is compared against it explicitly rather than in a footnote.
EXTERNAL_SLO_F1 = 0.85

# The floor below which this repository will not quote a detection metric: a
# class with fewer distinct feature signatures than this cannot distinguish
# "the model learned the traffic" from "the model memorised a short list". On
# this dataset the ambiguity guard leaves the attack class far below both.
MIN_REPORTABLE_MINORITY_VECTORS = 100
MIN_REPORTABLE_MINORITY_ROWS = 500

TRAINED_LABELS = ["benign", "lateral_movement"]

BOUNCER_SKIP_REASON = (
    "DNS-EXF2021 contains no DoS/flood traffic — every row is either benign DNS "
    "or an exfiltration channel. The Bouncer's contract is strictly binary "
    "flood-vs-benign (predict_verdict emits only Label.FLOOD or Label.BENIGN), so "
    "training it here would require labelling exfiltration as a flood, which is "
    "false. Skipped deliberately; the Detective's lateral_movement class is what "
    "this dataset can exercise."
)

RECONSTRUCTION_NOTE = (
    "The dataset carries no IP address, no port, no byte count and no flow "
    "duration, so all four are reconstructed: source_ip is a single constant "
    "monitored client (the dataset records no client identity, and a "
    "class-varying source would be a label proxy); dest_ip is a deterministic "
    "synthetic IPv4 derived from the row's REAL queried domain (`sld`), so "
    "distinct-destination counts in a window are the dataset's own "
    "distinct-domain counts; total_bytes is derived from the REAL query-name "
    "length (`len`) via a documented DNS packet-size formula; duration_ms is 0 "
    "because no duration is recorded. dest_port=53 and protocol=UDP are the "
    "dataset's subject matter, not invented measurements."
)


def _build_graph_snapshots(rows: list, label: Label) -> list[tuple]:
    """Build windowed graph snapshots for GAT training.

    Matches the construction used by the other external-dataset runners
    (scripts/run_dohbrw2020_experiment.py::_build_graph_snapshots) so results
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
    """Cap `items` at `limit` by an evenly spaced stride, not a prefix.

    A prefix would bias the sample toward whichever capture file happens to
    sort first, so the stride spans every source file and the whole capture
    timeline. Deterministic — no RNG — so the experiment reproduces exactly.
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
        "training_origin": "dnsexf2021",
        "skipped": BOUNCER_SKIP_REASON,
        "n_flood_rows": n_flood,
        "n_rows_considered": len(rows),
        "features": FEATURE_NAMES,
    }


def train_detective(rows: list, seed: int, limit: int | None = 10_000) -> tuple[DetectiveModel | None, dict]:
    """Train the Detective on DNS-EXF2021 graph windows: benign vs lateral_movement."""
    benign_rows = [r for r in rows if r.category == "normal"]
    tunnel_rows = [r for r in rows if r.category == "lateral_movement"]

    if not benign_rows or not tunnel_rows:
        return None, {
            "component": "detective",
            "training_origin": "dnsexf2021",
            "skipped": "DNS-EXF2021 subset lacks both benign and exfiltration rows",
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
            "training_origin": "dnsexf2021",
            "skipped": "No graph windows produced from DNS-EXF2021 train rows",
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
            "training_origin": "dnsexf2021",
            "skipped": "No graph windows produced from DNS-EXF2021 test rows",
        }

    y_true, y_pred, n_uncertain = [], [], 0
    for snap, lbl in test_snaps:
        v = model.predict_verdict(snap, window_id=snap.window_id)
        y_true.append(lbl.value)
        if v.label == Label.UNCERTAIN.value:
            n_uncertain += 1
        # Same convention as Experiments B, C and D: a verdict in the gray zone
        # is counted toward the attack class, and the count is reported so the
        # effect is visible rather than buried.
        y_pred.append(v.label if v.label != Label.UNCERTAIN.value else "lateral_movement")

    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)]))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=TRAINED_LABELS, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[label.value for label in RAW_CLASSES]).tolist()

    DETECTIVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(DETECTIVE_DIR)

    metrics = {
        "component": "detective",
        "training_origin": "dnsexf2021",
        "dataset_source": DATASET_SOURCE,
        "dataset_url": DATASET_URL,
        "model_type": "GAT in autograd (trained from scratch)",
        "task": "macro benign vs lateral_movement graph window classification",
        "synthetic_hosts": True,
        "synthetic_hosts_note": RECONSTRUCTION_NOTE,
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
    print(f"  Detective (DNS-EXF2021): Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    return model, metrics


def verify_weights(bouncer_trained: bool) -> dict:
    """Verify saved models load cleanly and execute correct inference."""
    results = {}

    if not bouncer_trained:
        results["dnsexf2021_bouncer_loadable"] = None
        results["dnsexf2021_bouncer_note"] = "not trained on this dataset — see bouncer.skipped"
        print("  DNS-EXF2021 Bouncer load: SKIPPED (not trained on this dataset)")
    else:
        bouncer_ok = False
        try:
            from services.bouncer.model import BouncerModel

            b = BouncerModel.load(BOUNCER_DIR)
            feats = {n: 10.0 for n in FEATURE_NAMES}
            v = b.predict_verdict(feats, window_id="verify-dnsexf-bouncer")
            bouncer_ok = v is not None and 0.0 <= v.confidence <= 1.0
            print(f"  DNS-EXF2021 Bouncer load: {'PASSED' if bouncer_ok else 'FAILED'} "
                  f"(verdict={v.label}, conf={v.confidence:.3f})")
        except Exception as exc:
            print(f"  DNS-EXF2021 Bouncer load FAILED: {exc}")
        results["dnsexf2021_bouncer_loadable"] = bouncer_ok

    detective_ok = False
    try:
        d = DetectiveModel.load(DETECTIVE_DIR)
        now = datetime.now(timezone.utc)
        # Shaped like the real capture here: the monitored client querying
        # distinct domains (the topology twin/dnsexf2021.py actually produces).
        snap = GraphSnapshot(
            window_id="verify-dnsexf-detective",
            window_start=now,
            window_end=now,
            nodes=[
                GraphNode(node_id="10.30.0.1", degree_in=0.0, degree_out=2.0,
                          bytes_total=500.0, unique_ports_contacted=1),
                GraphNode(node_id="10.40.97.11", degree_in=2.0, degree_out=0.0,
                          bytes_total=500.0, unique_ports_contacted=0),
            ],
            edges=[
                GraphEdge(src="10.30.0.1", dst="10.40.97.11", bytes=500.0,
                          flow_count=2, port_entropy=0.0, duration_mean_ms=0.0),
            ],
        )
        v = d.predict_verdict(snap, window_id="verify-dnsexf-detective")
        detective_ok = v is not None and 0.0 <= v.confidence <= 1.0
        print(f"  DNS-EXF2021 Detective load: {'PASSED' if detective_ok else 'FAILED'} "
              f"(verdict={v.label}, conf={v.confidence:.3f})")
    except Exception as exc:
        print(f"  DNS-EXF2021 Detective load FAILED: {exc}")
    results["dnsexf2021_detective_loadable"] = detective_ok
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10_000,
                        help="max rows per class for the Detective window pools")
    parser.add_argument("--sample-per-file", type=int, default=None,
                        help="optional cap on rows read per CSV (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("CIC-Bell-DNS-EXF-2021 INDEPENDENT TRAINING & EVALUATION")
    print("=" * 70)

    if not DATA_DIR.exists() or not list(DATA_DIR.glob("**/*.csv")):
        print(f"Dataset directory not found or empty: {DATA_DIR}")
        print("Run: python scripts/download_dnsexf2021.py --from-mirror")
        return 1

    print(f"Measuring label quality in {DATA_DIR} (this is the disclosure) ...")
    t0 = time.perf_counter()
    label_quality = ambiguity_report(DATA_DIR, args.sample_per_file)
    print(f"  rows={label_quality.get('rows', 0):,}  "
          f"distinct_vectors={label_quality.get('distinct_feature_vectors', 0):,}  "
          f"ceiling={label_quality.get('deterministic_ceiling')}")
    print(f"  {label_quality.get('ambiguous_vectors', 0)} vectors carry >1 class, "
          f"involving {label_quality.get('ambiguous_row_fraction', 0.0) * 100:.1f}% of rows")

    print(f"\nLoading DNS-EXF2021 flows from {DATA_DIR} ...")
    rows = load_dnsexf2021(DATA_DIR, sample_per_file=args.sample_per_file)
    load_time = time.perf_counter() - t0

    cats: dict[str, int] = {}
    for r in rows:
        cats[r.category] = cats.get(r.category, 0) + 1
    print(f"  Loaded {len(rows):,} flows in {load_time:.2f}s")
    print(f"  Category distribution: {cats}")

    if not rows:
        print("No rows loaded — nothing to train on. Aborting without writing metrics.")
        return 1

    print("\n--- Bouncer on DNS-EXF2021: deliberately SKIPPED ---")
    print(f"  {BOUNCER_SKIP_REASON}")
    bouncer_metrics = skip_bouncer(rows)

    print("\n--- Training Detective (GAT in autograd) on DNS-EXF2021 ---")
    detective_model, detective_metrics = train_detective(rows, seed=SEED, limit=args.limit)

    print("\n--- Verifying Weight Loadability & Live Inference ---")
    weight_verification = verify_weights(bouncer_trained=False)

    ceiling = label_quality.get("deterministic_ceiling")
    f1 = detective_metrics.get("f1")

    # The guard in twin/dnsexf2021.py removes every feature vector that carries
    # two labels. On this dataset that removes 99.95% of the attack rows, so
    # what survives cannot support a detection claim — say so from measurement
    # rather than from opinion.
    minority_rows = label_quality.get("rows_after_guard_by_category", {}).get("lateral_movement", 0)
    minority_vectors = label_quality.get(
        "distinct_vectors_after_guard_by_category", {}
    ).get("lateral_movement", 0)
    reportable = (
        minority_vectors >= MIN_REPORTABLE_MINORITY_VECTORS
        and minority_rows >= MIN_REPORTABLE_MINORITY_ROWS
    )

    if reportable:
        verdict = "DETECTION_METRIC_VALID"
        finding = (
            f"The exfiltration class retained {minority_vectors:,} distinct feature "
            f"signatures across {minority_rows:,} rows after the ambiguity guard, "
            "which is enough to report a bounded detection metric."
        )
    else:
        verdict = "NO_VALID_DETECTION_METRIC"
        ambiguous_pct = label_quality.get("ambiguous_row_fraction", 0) * 100
        attack_ambiguous_pct = (
            label_quality.get("ambiguous_row_fraction_by_category", {}).get(
                "lateral_movement", 0
            )
            * 100
        )
        # The loaded rows are de-duplicated as well as guarded, so the class
        # balance the model actually trains on is the surviving-row counts, not
        # the pre-dedupe `rows_after_guard_by_category` — and the benign side is
        # further bounded to keep this runnable on a laptop. Read both off the
        # artifacts rather than restating them here, so the text cannot drift.
        benign_vectors = label_quality.get(
            "distinct_vectors_after_guard_by_category", {}
        ).get("normal", 0)
        trained_attack = detective_metrics.get("n_train_tunnel", 0) + detective_metrics.get(
            "n_test_tunnel", 0
        )
        trained_benign = detective_metrics.get("n_train_benign", 0) + detective_metrics.get(
            "n_test_benign", 0
        )
        finding = (
            "This dataset cannot support a detection claim, and the metrics below "
            f"say so rather than quoting a number. Measured: {ambiguous_pct:.2f}% "
            "of all rows sit on a feature vector that carries two different labels, "
            f"including {attack_ambiguous_pct:.2f}% "
            "of the exfiltration rows. Removing those leaves the attack class with "
            f"{minority_rows} rows carrying only {minority_vectors} distinct "
            f"signatures, against {benign_vectors:,} distinct signatures of benign "
            "traffic; after de-duplication the experiment trains and evaluates on "
            f"{trained_attack} attack rows against a bounded {trained_benign:,}-row "
            "benign sample. The trained pipeline's score is recorded below and "
            "explicitly marked non-reportable: on a "
            f"{minority_vectors}-signature minority class it measures a lookup "
            "table, not detection. The hard bound for any model on the full corpus "
            f"is {ceiling}, which is below the repository's own external-dataset "
            f"SLO of {EXTERNAL_SLO_F1}."
        )

    detective_metrics["reportable_as_detection"] = reportable
    detective_metrics["why_not_reportable"] = None if reportable else (
        "The attack class retains only "
        f"{minority_vectors} distinct feature signatures / {minority_rows} rows after "
        "removing label-ambiguous vectors, below the "
        f"{MIN_REPORTABLE_MINORITY_VECTORS}-signature / {MIN_REPORTABLE_MINORITY_ROWS}-row "
        "floor this repository requires before quoting a detection metric. The "
        "numbers in this block are a pipeline smoke test: they confirm the model "
        "trains, saves and loads, and they must NOT be cited as detection performance."
    )

    full_metrics = {
        "experiment": "dnsexf2021",
        "status": "COMPLETE",
        "verdict": verdict,
        "finding": finding,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Detective-only experiment, and a deliberately negative result. The "
            "dataset's labels are capture-level and not recoverable from its "
            "features, and it carries no IPs, ports, bytes or durations — so every "
            "input the lanes consume is reconstructed. No valid detection metric "
            "can be produced; see `finding` and `label_quality` for the measurements "
            "behind that, and `detective.reportable_as_detection` for the pipeline "
            "output's status. The Bouncer is skipped: this dataset has no flood "
            "traffic to train its binary contract on."
        ),
        "dataset": {
            "source": DATASET_SOURCE,
            "url": DATASET_URL,
            "total_flows": len(rows),
            "total_flows_before_guard": label_quality.get("rows"),
            "category_distribution": cats,
            "synthetic_hosts": True,
            "synthetic_hosts_note": RECONSTRUCTION_NOTE,
            "features_are_real": False,
            "features_are_real_note": (
                "The DNS query statistics are real. `total_bytes` is derived from "
                "the real query-name length, and every other quantity the lanes "
                "consume is reconstructed — see synthetic_hosts_note."
            ),
        },
        "label_quality": label_quality,
        "label_quality_note": (
            "All values are measured from the downloaded files, not assumed. "
            "`deterministic_ceiling` is a hard upper bound on accuracy for ANY "
            "model reading this dataset's features: where one feature vector "
            "carries two classes, no function of those features can be right on "
            "more than the majority count. It is computed on the RAW corpus, so it "
            "stays an honest bound on the source rather than on the guarded subset."
        ),
        "slo": {
            "target_f1": EXTERNAL_SLO_F1,
            "measured_f1": f1,
            "ceiling": ceiling,
            "target_reachable": bool(ceiling is not None and ceiling >= EXTERNAL_SLO_F1),
            "meets_slo": bool(reportable and f1 is not None and f1 >= EXTERNAL_SLO_F1),
            "note": (
                "The repository's external-dataset SLO (F1 > 0.85) is unreachable on "
                "this dataset by construction, because the measured ceiling is below "
                "it. `meets_slo` is therefore False however the smoke test scores."
            ),
        },
        "bouncer": bouncer_metrics,
        "detective": detective_metrics,
        "weight_verification": weight_verification,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(full_metrics, indent=2))

    print("\n" + "=" * 70)
    print("CIC-Bell-DNS-EXF-2021 EXPERIMENT COMPLETE")
    print(f"VERDICT: {verdict}")
    print(f"  {finding}")
    print("-" * 70)
    print(f"  hard ceiling for any model on this data : {ceiling}")
    print(f"  repo external-dataset SLO               : >{EXTERNAL_SLO_F1} (unreachable here)")
    print(f"  pipeline smoke-test F1 (NOT reportable) : {f1}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Detective weights: {DETECTIVE_DIR}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
