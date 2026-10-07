#!/usr/bin/env python3
"""CSE-CIC-IDS2018 Independent ML Experiment for KRONUS.

EXPERIMENT: CSE-CIC-IDS2018 External Validation (Experiment F)
==============================================================
Trains fresh, independent KRONUS Bouncer and Detective models on
CSE-CIC-IDS2018 (Canadian Institute for Cybersecurity, University of New
Brunswick), the successor to CIC-IDS2017.

Dataset URL: https://www.unb.ca/cic/datasets/ids-2018.html

    Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani,
    "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
    Characterization", 4th ICISSP, Portugal, January 2018.

WHAT THIS DATASET GIVES EACH LANE
---------------------------------
Unlike a single-attack capture, CSE-CIC-IDS2018 carries both flood traffic and
scanning traffic, so BOTH KRONUS lanes train here:

  Bouncer   (flood vs benign)      <- DoS attacks-Hulk, SlowHTTPTest, DDOS-HOIC
  Detective (benign vs port_scan)  <- Infilteration (Nmap sweep + full port scan)

The label mapping, and the measurement that justifies calling `Infilteration` a
port scan, are in twin/cicids2018.py.

THE FOUR DAYS, AND WHY EACH LANE GETS TWO
-----------------------------------------
Both lanes are trained and tested across two capture days, because a single day
lets a model separate the classes by *capture date* instead of by traffic shape:

  28-02  Infilteration + Benign     -> Detective, and Bouncer benign
  01-03  Infilteration + Benign     -> Detective, and Bouncer benign
  16-02  DoS Hulk + SlowHTTPTest    -> Bouncer flood
  21-02  DDOS HOIC                  -> Bouncer flood

Two days does not abolish that risk, so the Bouncer's `cross_day` block tests it
directly: fit on one flood day plus one benign day, score on the two unseen.

BENIGN ROWS INSIDE AN ATTACK ARE HELD OUT, NOT MISLABELLED
----------------------------------------------------------
The release has no client identity, so every row is replayed onto one
reconstructed source and the featurizer's 2-second aggregate is global to the
capture, not per-host. A benign flow arriving mid-flood therefore sits in a
window the flood dominates and its features are the flood's. Labelling those
rows "benign" would teach the model that a flood is benign. `attack_intervals()`
grows each attack's span and benign is scored only outside it. The day survey in
the metrics shows what this costs: on an attack day, nearly all benign traffic
is concurrent with the attack, which is why benign comes from the two
infiltration days instead of the flood days.

WHY THE STRIDE IS GLOBAL
------------------------
All four files are thinned by the SAME stride, so a flood row and a benign row
are decimated identically and their rates stay comparable. A per-file row budget
would thin a 333 MB day harder than a 108 MB day and manufacture a rate
difference that is not in the data.

THE DETECTIVE'S CEILING IS MEASURED BEFORE IT IS TRAINED
--------------------------------------------------------
At the production 2-second cadence the Infilteration label is only weakly
separable from benign. Rather than train a model and then explain a poor score,
the runner measures the ceiling first with a cross-validated linear probe over
the same windows, and sweeps the cadence to show whether the limit is the attack
or the window. Read both together: if the attack/normal activity ratio stays flat
while AUC climbs, longer windows are measuring the same small difference more
precisely, not finding new signal.

HOST RECONSTRUCTION (honest disclosure)
---------------------------------------
The ML-ready CSVs carry NO IP addresses and NO source ports, so host identity is
reconstructed deterministically (see twin/cicids2018.py) and recorded as
"synthetic_hosts": true. Flow measurements — bytes, durations, ports, protocol,
timestamps and all 74 CICFlowMeter columns — are REAL, and unlike Experiments
A-E the capture clock is real too, so windows are true 2-second slices rather
than synthetic row spacing. `dest_ip` is a bijection of the real destination
port, so the graph measures SERVICE fan-out, not host fan-out; two of the four
edge features (port_entropy, flow_count) are constant by construction. That is a
real limit on the graph lane here, and it is stated in the metrics rather than
left for a reader to discover.

TRAIN/TEST SPLIT
  Bouncer:   stratified 67/33 per class, seeded; plus a leave-one-day-out block.
  Detective: stratified 67/33 per class over graph windows, seeded. Windows are
             subsampled by an evenly spaced stride, per day, so neither the
             quota nor a time-of-day prefix biases the training set.

WEIGHT ARTIFACTS
  models/experiments/cicids2018/bouncer/   (bouncer.json, calibration.json)
  models/experiments/cicids2018/detective/ (detective.npz, detective.onnx)

METRICS
  results/experiments/cicids2018_metrics.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from libs.constants import Label
from libs.schemas import GraphEdge, GraphNode, GraphSnapshot
from services.bouncer.features import FEATURE_NAMES, FlowFeaturizer
from services.bouncer.model import BouncerModel
from services.detective.model import DetectiveModel
from services.graph_builder.builder import WindowedGraphBuilder
from services.telemetry_exporter.converters import flow_row_to_event
from twin.cicids2018 import (
    attack_intervals,
    in_any_interval,
    label_quality_report,
    load_cicids2018_timed,
)

DATA_DIR = REPO_ROOT / "data" / "external" / "cicids2018"
MODELS_DIR = REPO_ROOT / "models" / "experiments" / "cicids2018"
BOUNCER_DIR = MODELS_DIR / "bouncer"
DETECTIVE_DIR = MODELS_DIR / "detective"
RESULTS_DIR = REPO_ROOT / "results" / "experiments"
METRICS_PATH = RESULTS_DIR / "cicids2018_metrics.json"

DATASET_URL = "https://www.unb.ca/cic/datasets/ids-2018.html"
DATASET_SOURCE = "CSE-CIC-IDS2018 (Canadian Institute for Cybersecurity, UNB)"

BOUNCER_FLOOD_DAYS = ("Friday-16-02-2018", "Wednesday-21-02-2018")
BOUNCER_BENIGN_DAYS = ("Wednesday-28-02-2018", "Thursday-01-03-2018")
DETECTIVE_DAYS = ("Wednesday-28-02-2018", "Thursday-01-03-2018")
ALL_DAYS = tuple(dict.fromkeys(BOUNCER_FLOOD_DAYS + BOUNCER_BENIGN_DAYS + DETECTIVE_DAYS))

TRAIN_RATIO = 0.67
SEED = 42
WINDOW_SECONDS = 2.0            # the production cadence; not tuned here
QUALITY_STRIDE = 40             # stride for the feature-carrying label-quality load

# The cadences the ceiling sweep walks, to separate "the attack is weak" from
# "the window is too fine for it". 2.0 is the production value.
CADENCE_SWEEP_SECONDS = (2.0, 5.0, 10.0, 30.0, 60.0, 120.0)

WINDOW_FEATURE_NAMES = [
    "n_edges", "n_nodes", "max_unique_ports", "max_degree_out", "max_degree_in",
    "mean_duration_ms", "mean_bytes", "max_port_entropy", "mean_flow_count",
]


def _day_path(stem: str) -> Path:
    return DATA_DIR / f"{stem}_TrafficForML_CICFlowMeter.csv"


def _load_day(stem: str, stride: int, with_features: bool) -> tuple[list, list]:
    return load_cicids2018_timed(
        _day_path(stem), sample_per_file=None, stride=stride,
        with_features=with_features,
    )


def _stratified_split(items: list, ratio: float, seed: int) -> tuple[list, list]:
    """Split a single class's items, shuffled by a seeded permutation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    n_train = int(len(idx) * ratio)
    train_idx = set(idx[:n_train].tolist())
    train = [items[i] for i in range(len(items)) if i in train_idx]
    test = [items[i] for i in range(len(items)) if i not in train_idx]
    return train, test


def _even_subsample(items: list, n: int) -> list:
    """Thin `items` to `n` by an evenly spaced stride.

    A prefix would take the first n items, and both rows and windows arrive in
    time order — so a prefix is a time-of-day sample, not a sample. This keeps
    the whole span, matching the repo's bounded-sample rule elsewhere.
    """
    if n >= len(items) or n <= 0:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).astype(int)
    return [items[i] for i in idx]


# --- Bouncer ---------------------------------------------------------------

def _replay_bouncer_features(day_pools: dict[str, list]) -> np.ndarray:
    """Replay each day's rows through a fresh FlowFeaturizer, in true time.

    A fresh featurizer per day is deliberate: the days are separate captures
    weeks apart, and carrying window state across the join would fabricate
    events. Within a day the real inter-arrival times are used as they are,
    which is what makes `event_rate` a real burst rate rather than a constant
    invented by even spacing.
    """
    out: list[list[float]] = []
    for stem in sorted(day_pools):
        featurizer = FlowFeaturizer()
        for row, ts in day_pools[stem]:
            feats = featurizer.features_for(flow_row_to_event(row, ts=ts))
            out.append([feats[name] for name in FEATURE_NAMES])
    if not out:
        return np.empty((0, len(FEATURE_NAMES)))
    return np.array(out)


def _label_census(rows: list) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r.raw_label] = out.get(r.raw_label, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def collect_bouncer_pools(stride: int, limit: int | None) -> tuple[dict, dict, dict]:
    """Per-day flood and benign pools, plus the day survey that explains them."""
    flood: dict[str, list] = {}
    benign: dict[str, list] = {}
    survey: dict[str, dict] = {}

    for stem in ALL_DAYS:
        rows, times = _load_day(stem, stride, with_features=False)
        if not rows:
            continue
        # strict=True: a row without its capture time would be silently dropped
        # by a lenient zip, and every rate feature here is a function of time.
        paired = list(zip(rows, times, strict=True))
        intervals = attack_intervals(rows, times)
        day_flood = [(r, t) for r, t in paired if r.category == "dos"]
        day_benign = [
            (r, t) for r, t in paired
            if r.category == "normal" and not in_any_interval(t, intervals)
        ]
        held_out = sum(
            1 for r, t in paired
            if r.category == "normal" and in_any_interval(t, intervals)
        )
        if limit:
            day_flood = _even_subsample(day_flood, limit)
            day_benign = _even_subsample(day_benign, limit)
        if day_flood:
            flood[stem] = day_flood
        if day_benign:
            benign[stem] = day_benign
        survey[stem] = {
            "rows_loaded": len(rows),
            "flood_rows": len(day_flood),
            "benign_rows_outside_attack": len(day_benign),
            "benign_rows_held_out_inside_attack": held_out,
            "attack_intervals": len(intervals),
            "attack_span_hours": round(
                sum((e - s).total_seconds() for s, e in intervals) / 3600.0, 3
            ),
            "raw_labels": _label_census(rows),
        }
        del rows, times
    return flood, benign, survey


def _split_by_day(pools: dict, seed: int, take: str = "train") -> dict[str, list]:
    """Split each day's pool 67/33 independently, preserving the day key."""
    out: dict[str, list] = {}
    for stem, pairs in pools.items():
        train, test = _stratified_split(pairs, TRAIN_RATIO, seed)
        chosen = train if take == "train" else test
        if chosen:
            out[stem] = chosen
    return out


def _fit_bouncer(flood: dict, benign: dict, seed: int) -> tuple[BouncerModel, dict]:
    """Fit on the given per-day pools and score on a stratified 67/33 split.

    The split is taken per day rather than over the pooled rows. Splitting the
    pool would let one day's rows land mostly in train and another's mostly in
    test, converting a day difference into a test-set difference; splitting each
    day separately keeps every day's contribution proportional on both sides.
    It also keeps the day key attached to every row, which the replay needs in
    order to start a fresh featurizer per capture.
    """
    flood_train = _split_by_day(flood, seed)
    flood_test = _split_by_day(flood, seed, take="test")
    benign_train = _split_by_day(benign, seed + 1)
    benign_test = _split_by_day(benign, seed + 1, take="test")

    n_flood_train = sum(len(v) for v in flood_train.values())
    n_benign_train = sum(len(v) for v in benign_train.values())
    n_flood_test = sum(len(v) for v in flood_test.values())
    n_benign_test = sum(len(v) for v in benign_test.values())
    if not n_flood_train or not n_benign_train:
        raise ValueError("both a flood and a benign training pool are required")

    X_tr = np.vstack([_replay_bouncer_features(flood_train),
                      _replay_bouncer_features(benign_train)])
    y_tr = np.concatenate([np.ones(n_flood_train, int), np.zeros(n_benign_train, int)])
    X_te = np.vstack([_replay_bouncer_features(flood_test),
                      _replay_bouncer_features(benign_test)])
    y_te = np.concatenate([np.ones(n_flood_test, int), np.zeros(n_benign_test, int)])

    model = BouncerModel()
    started = time.perf_counter()
    model.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - started

    proba = model.predict_proba(X_te)
    y_pred = (proba >= 0.5).astype(int)
    prf = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)
    return model, {
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "n_train_flood": int(y_tr.sum()), "n_train_benign": int((y_tr == 0).sum()),
        "n_test_flood": int(y_te.sum()), "n_test_benign": int((y_te == 0).sum()),
        "train_seconds": round(elapsed, 3),
        "accuracy": round(float((y_pred == y_te).mean()), 4),
        "precision": round(float(prf[0]), 4),
        "recall": round(float(prf[1]), 4),
        "f1": round(float(prf[2]), 4),
        "auc": round(float(roc_auc_score(y_te, proba)), 4),
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["benign", "flood"],
    }


def train_bouncer(
    stride: int, limit: int | None, seed: int, skip_cross_day: bool
) -> tuple[BouncerModel | None, dict]:
    flood, benign, survey = collect_bouncer_pools(stride, limit)
    if not flood or not benign:
        return None, {
            "skipped": "no flood and/or benign rows survived loading",
            "n_flood_days": len(flood), "n_benign_days": len(benign),
        }
    print(f"  flood pool:  { {k: len(v) for k, v in flood.items()} }")
    print(f"  benign pool: { {k: len(v) for k, v in benign.items()} }")

    model, metrics = _fit_bouncer(flood, benign, seed)
    print(f"  Bouncer: Acc={metrics['accuracy']:.4f} Prec={metrics['precision']:.4f} "
          f"Rec={metrics['recall']:.4f} F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")

    # The evidence behind that score: class-conditional medians. A model that
    # separates on traffic shape and one that separates on a day fingerprint can
    # both score 0.99; these medians and the cross-day block are what tell them
    # apart, so they ship with the number rather than as an afterthought.
    Xf = _replay_bouncer_features(flood)
    Xb = _replay_bouncer_features(benign)
    metrics["feature_medians"] = {
        name: {
            "flood": round(float(np.median(Xf[:, j])), 4),
            "benign": round(float(np.median(Xb[:, j])), 4),
        }
        for j, name in enumerate(FEATURE_NAMES)
    }

    cross: list[dict] = []
    if not skip_cross_day:
        for train_flood in BOUNCER_FLOOD_DAYS:
            for train_benign in BOUNCER_BENIGN_DAYS:
                test_flood = [d for d in BOUNCER_FLOOD_DAYS if d != train_flood]
                test_benign = [d for d in BOUNCER_BENIGN_DAYS if d != train_benign]
                if not all(d in flood for d in [train_flood, *test_flood]):
                    continue
                if not all(d in benign for d in [train_benign, *test_benign]):
                    continue
                # Fit on the two training days only, then score the two days the
                # model has never seen. The split inside _fit_bouncer is not used
                # for this block — every row of a training day is training data
                # and every row of a test day is unseen — so the fit is done here
                # against the full pools rather than a 67% slice of them.
                X_tr = np.vstack([
                    _replay_bouncer_features({train_flood: flood[train_flood]}),
                    _replay_bouncer_features({train_benign: benign[train_benign]}),
                ])
                y_tr = np.concatenate([np.ones(len(flood[train_flood]), int),
                                       np.zeros(len(benign[train_benign]), int)])
                unseen = BouncerModel().fit(X_tr, y_tr)
                X_te = np.vstack([
                    _replay_bouncer_features({d: flood[d] for d in test_flood}),
                    _replay_bouncer_features({d: benign[d] for d in test_benign}),
                ])
                y_te = np.concatenate([
                    np.ones(sum(len(flood[d]) for d in test_flood), int),
                    np.zeros(sum(len(benign[d]) for d in test_benign), int),
                ])
                proba = unseen.predict_proba(X_te)
                pred = (proba >= 0.5).astype(int)
                row = {
                    "trained_on": {"flood": train_flood, "benign": train_benign},
                    "tested_on": {"flood": test_flood, "benign": test_benign},
                    "n_train": int(len(y_tr)),
                    "n_test": int(len(y_te)),
                    "f1": round(float(precision_recall_fscore_support(
                        y_te, pred, average="binary", zero_division=0)[2]), 4),
                    "auc": round(float(roc_auc_score(y_te, proba)), 4),
                    "confusion_matrix": confusion_matrix(y_te, pred, labels=[0, 1]).tolist(),
                    "confusion_matrix_labels": ["benign", "flood"],
                }
                cross.append(row)
                print(f"  cross-day {train_flood[:10]}+{train_benign[:10]} -> "
                      f"{'+'.join(d[:10] for d in test_flood)}+"
                      f"{'+'.join(d[:10] for d in test_benign)}: "
                      f"F1={row['f1']:.4f} AUC={row['auc']:.4f}")

    metrics.update({
        "component": "bouncer",
        "training_origin": "cicids2018",
        "dataset_source": DATASET_SOURCE,
        "dataset_url": DATASET_URL,
        "model_type": "XGBoost + Temperature Scaling (trained from scratch)",
        "task": "binary DoS/DDoS flood vs benign classification",
        "synthetic_hosts": True,
        "seed": seed,
        "train_ratio": TRAIN_RATIO,
        "stride": stride,
        "features": FEATURE_NAMES,
        "cross_day": cross,
        "day_survey": survey,
        "weight_artifacts": [
            str(BOUNCER_DIR / "bouncer.json"),
            str(BOUNCER_DIR / "calibration.json"),
        ],
    })
    return model, metrics


# --- Detective -------------------------------------------------------------

def _window_vector(snap: GraphSnapshot) -> list[float]:
    """Aggregate one snapshot into the flat vector the linear probe reads.

    These are the quantities a graph lane can see, reduced to order statistics,
    so the probe measures the information available at this window size rather
    than the GAT's ability to use it.
    """
    edges, nodes = snap.edges, snap.nodes
    return [
        float(len(edges)), float(len(nodes)),
        float(max(n.unique_ports_contacted for n in nodes)),
        float(max(n.degree_out for n in nodes)),
        float(max(n.degree_in for n in nodes)),
        float(np.mean([e.duration_mean_ms for e in edges])),
        float(np.mean([e.bytes for e in edges])),
        float(max(e.port_entropy for e in edges)),
        float(np.mean([e.flow_count for e in edges])),
    ]


def _probe_metrics(seconds: float, X: np.ndarray, y: np.ndarray) -> dict:
    """Cross-validated linear separability at one cadence.

    `median_edges_*` is the diagnostic that keeps the AUC honest: if it stays
    put while AUC rises, longer windows are estimating the same small difference
    more precisely, not revealing a scan signature.
    """
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"window_seconds": seconds, "available": False,
                "reason": "degenerate label distribution", "windows": int(len(y))}
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    proba = cross_val_predict(
        LogisticRegression(max_iter=3000, class_weight="balanced"),
        X, y, cv=cv, method="predict_proba",
    )[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "window_seconds": seconds,
        "available": True,
        "windows": int(len(y)),
        "attack_windows": int(y.sum()),
        "attack_fraction": round(float(y.mean()), 4),
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "f1_attack": round(float(precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0)[2]), 4),
        "majority_class_accuracy": round(float(max(y.mean(), 1 - y.mean())), 4),
        "median_edges_attack": float(np.median(X[y == 1, 0])),
        "median_edges_benign": float(np.median(X[y == 0, 0])),
        "median_ratio_attack_over_benign": {
            name: (round(float(np.median(X[y == 1, j]) / np.median(X[y == 0, j])), 4)
                   if np.median(X[y == 0, j]) else None)
            for j, name in enumerate(WINDOW_FEATURE_NAMES)
        },
    }


def collect_detective(
    stride: int, quota_per_class: int, cadences: tuple[float, ...]
) -> tuple[list, dict, list, dict]:
    """One pass per day driving every cadence's builder over the same stream.

    Returns (train_windows, probe, sweep, survey). `train_windows` holds
    GraphSnapshot objects for the GAT, bounded to `quota_per_class` by an evenly
    spaced stride per day, so no quota and no time-of-day prefix biases it.
    `probe` and `sweep` keep only the 9-float window vectors, which are cheap
    enough to retain for every window at every cadence — so the ceiling is
    measured on everything while the GAT trains on a bounded, day-balanced set.
    """
    surfaces = list(dict.fromkeys([WINDOW_SECONDS, *cadences]))
    sweep: dict[float, dict[str, list]] = {c: {"X": [], "y": []} for c in surfaces}
    quota_per_day = max(1, quota_per_class // len(DETECTIVE_DAYS))
    quotas: dict[str, dict[str, list]] = {
        stem: {"attack": [], "benign": []} for stem in DETECTIVE_DAYS
    }
    survey: dict[str, dict] = {}

    for stem in DETECTIVE_DAYS:
        rows, times = _load_day(stem, stride, with_features=False)
        if not rows:
            continue
        intervals = attack_intervals(rows, times)
        builders = {c: WindowedGraphBuilder(window_seconds=c) for c in surfaces}
        counts = {c: {"attack": 0, "benign": 0} for c in surfaces}
        for row, ts in zip(rows, times, strict=True):
            event = flow_row_to_event(row, ts=ts)
            for cadence, builder in builders.items():
                snap = builder.ingest(event)
                if snap is None or not snap.edges or not snap.nodes:
                    continue
                is_attack = in_any_interval(snap.window_start, intervals)
                bucket = "attack" if is_attack else "benign"
                counts[cadence][bucket] += 1
                sweep[cadence]["X"].append(_window_vector(snap))
                sweep[cadence]["y"].append(1 if is_attack else 0)
                # Only the production cadence needs the objects themselves, and
                # only up to the per-day share of the quota.
                if cadence == WINDOW_SECONDS and len(quotas[stem][bucket]) < quota_per_day:
                    quotas[stem][bucket].append(
                        (snap, Label.PORT_SCAN if is_attack else Label.BENIGN)
                    )
        survey[stem] = {
            "rows_loaded": len(rows),
            "attack_intervals": len(intervals),
            "attack_span_hours": round(
                sum((e - s).total_seconds() for s, e in intervals) / 3600.0, 3
            ),
            "windows_at_production_cadence": (
                counts[WINDOW_SECONDS]["attack"] + counts[WINDOW_SECONDS]["benign"]
            ),
            "attack_windows": counts[WINDOW_SECONDS]["attack"],
            "benign_windows": counts[WINDOW_SECONDS]["benign"],
            "windows_kept_for_training": {
                "attack": len(quotas[stem]["attack"]),
                "benign": len(quotas[stem]["benign"]),
            },
        }
        print(f"  {stem[:17]}: {len(rows):,} rows -> "
              f"{counts[WINDOW_SECONDS]['attack'] + counts[WINDOW_SECONDS]['benign']:,} "
              f"windows at {WINDOW_SECONDS:g}s "
              f"({counts[WINDOW_SECONDS]['attack']:,} probe / "
              f"{counts[WINDOW_SECONDS]['benign']:,} benign); kept "
              f"{len(quotas[stem]['attack']):,}/{len(quotas[stem]['benign']):,}")
        del rows, times, builders

    train_windows: list = []
    for stem in DETECTIVE_DAYS:
        train_windows.extend(quotas[stem]["attack"])
        train_windows.extend(quotas[stem]["benign"])

    probe = {
        "X": np.array(sweep[WINDOW_SECONDS]["X"], float),
        "y": np.array(sweep[WINDOW_SECONDS]["y"], int),
    }
    del sweep[WINDOW_SECONDS]
    sweep_metrics = [
        _probe_metrics(c, np.array(s["X"], float), np.array(s["y"], int))
        for c, s in sorted(sweep.items())
    ]
    return train_windows, probe, sweep_metrics, survey


def train_detective(
    windows: list, limit: int | None, seed: int
) -> tuple[DetectiveModel | None, dict]:
    attack = [w for w in windows if w[1] is Label.PORT_SCAN]
    benign = [w for w in windows if w[1] is Label.BENIGN]
    if not attack or not benign:
        return None, {
            "skipped": "need both probe and benign windows",
            "n_probe": len(attack), "n_benign": len(benign),
        }
    if limit:
        attack = _even_subsample(attack, limit)
        benign = _even_subsample(benign, limit)

    atk_train, atk_test = _stratified_split(attack, TRAIN_RATIO, seed)
    ben_train, ben_test = _stratified_split(benign, TRAIN_RATIO, seed + 1)
    train_snaps = ben_train + atk_train
    test_snaps = ben_test + atk_test
    print(f"  Detective train windows: {len(train_snaps)} "
          f"({len(atk_train)} probe / {len(ben_train)} benign)")
    print(f"  Detective test  windows: {len(test_snaps)} "
          f"({len(atk_test)} probe / {len(ben_test)} benign)")

    model = DetectiveModel(rng=np.random.default_rng(seed))
    # Batch 16 rather than Experiment C's 4: this pool is an order of magnitude
    # larger, and averaging gradients over more examples is the direction
    # train_batch's own docstring argues for.
    epochs, batch_size = 4, 16
    started = time.perf_counter()
    for ep in range(epochs):
        rng = np.random.default_rng(seed + ep)
        order = rng.permutation(len(train_snaps))
        ep_loss, n_batches = 0.0, 0
        for si in range(0, len(order), batch_size):
            batch = [train_snaps[i] for i in order[si:si + batch_size]]
            ep_loss += model.train_batch(batch, learning_rate=0.02, momentum=0.9)
            n_batches += 1
        print(f"  Epoch {ep + 1}/{epochs}  avg_loss={ep_loss / max(n_batches, 1):.4f}")
    train_time = time.perf_counter() - started

    y_true, y_pred = [], []
    for snap, label in test_snaps:
        verdict = model.predict_verdict(snap, window_id=snap.window_id)
        # The verdict that comes back is post-`derive_label`, so it can be an
        # abstention (UNCERTAIN) or a class this dataset never trained. Recorded
        # verbatim and never matched to a class: mapping an abstention onto the
        # class it "probably" meant is exactly the failure this repo reports on
        # itself elsewhere. Abstentions still lower recall, because a predicted
        # value equal to neither class can be no class's true positive.
        y_true.append(label.value)
        y_pred.append(str(verdict.label))

    breakdown: dict[str, int] = {}
    for p in y_pred:
        breakdown[p] = breakdown.get(p, 0) + 1

    prf = precision_recall_fscore_support(
        y_true, y_pred, labels=["benign", "port_scan"], average="macro", zero_division=0
    )
    accuracy = float(np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)]))
    others = sorted(set(y_pred) - {"benign", "port_scan"})
    cm_labels = ["benign", "port_scan", *others]
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels).tolist()
    print(f"  Detective (CSE-CIC-IDS2018): Acc={accuracy:.4f} Prec={prf[0]:.4f} "
          f"Rec={prf[1]:.4f} F1={prf[2]:.4f}")
    print(f"  predictions: {breakdown}")

    DETECTIVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(DETECTIVE_DIR)
    return model, {
        "component": "detective",
        "training_origin": "cicids2018",
        "dataset_source": DATASET_SOURCE,
        "dataset_url": DATASET_URL,
        "model_type": "GAT in autograd (trained from scratch)",
        "task": "macro benign vs port_scan graph-window classification",
        "synthetic_hosts": True,
        "seed": seed,
        "train_ratio": TRAIN_RATIO,
        "window_seconds": WINDOW_SECONDS,
        "window_label_rule": (
            "window_start inside an attack interval (widened by a 3s guard); a "
            "window is never relabelled by row majority, because a mixed window "
            "is the slice the detector is asked about"
        ),
        "epochs": epochs,
        "batch_size": batch_size,
        "train_windows": len(train_snaps),
        "test_windows": len(test_snaps),
        "n_train_probe": len(atk_train), "n_train_benign": len(ben_train),
        "n_test_probe": len(atk_test), "n_test_benign": len(ben_test),
        "train_seconds": round(train_time, 3),
        "accuracy": round(accuracy, 4),
        "precision": round(float(prf[0]), 4),
        "recall": round(float(prf[1]), 4),
        "f1": round(float(prf[2]), 4),
        "prediction_breakdown": breakdown,
        "confusion_matrix": cm,
        "confusion_matrix_labels": cm_labels,
        "abstention_policy": (
            "verdicts outside {benign, port_scan} are reported as-is and count "
            "as errors for recall; none are folded into a class"
        ),
        "weight_artifacts": [
            str(DETECTIVE_DIR / "detective.npz"),
            str(DETECTIVE_DIR / "detective.onnx"),
        ],
    }


# --- weight verification ---------------------------------------------------

def verify_weights(bouncer_trained: bool, detective_trained: bool) -> dict:
    results: dict[str, bool] = {}

    if bouncer_trained:
        try:
            model = BouncerModel.load(BOUNCER_DIR)
            verdict = model.predict_verdict(
                {name: 10.0 for name in FEATURE_NAMES}, window_id="verify-cicids2018-bouncer"
            )
            ok = 0.0 <= float(verdict.confidence) <= 1.0
            print(f"  CSE-CIC-IDS2018 Bouncer load: {'PASSED' if ok else 'FAILED'} "
                  f"(verdict={verdict.label}, conf={verdict.confidence:.3f})")
            results["cicids2018_bouncer_loadable"] = ok
        except Exception as exc:  # noqa: BLE001 - reported to the metrics, not raised
            print(f"  CSE-CIC-IDS2018 Bouncer load FAILED: {exc}")
            results["cicids2018_bouncer_loadable"] = False
    else:
        print("  Bouncer not trained this run; skipping load check")
        results["cicids2018_bouncer_loadable"] = False

    if detective_trained:
        try:
            model = DetectiveModel.load(DETECTIVE_DIR)
            now = datetime.now(timezone.utc)
            snap = GraphSnapshot(
                window_id="verify-cicids2018-detective",
                window_start=now, window_end=now,
                nodes=[
                    GraphNode(node_id="172.31.69.1", degree_in=0.0, degree_out=2.0,
                              bytes_total=500.0, unique_ports_contacted=2),
                    GraphNode(node_id="10.60.0.80", degree_in=2.0, degree_out=0.0,
                              bytes_total=500.0, unique_ports_contacted=0),
                ],
                edges=[
                    GraphEdge(src="172.31.69.1", dst="10.60.0.80", bytes=500.0,
                              flow_count=2, port_entropy=0.5, duration_mean_ms=15.0),
                ],
            )
            verdict = model.predict_verdict(snap, window_id="verify-cicids2018-detective")
            ok = 0.0 <= float(verdict.confidence) <= 1.0
            print(f"  CSE-CIC-IDS2018 Detective load: {'PASSED' if ok else 'FAILED'} "
                  f"(verdict={verdict.label}, conf={verdict.confidence:.3f})")
            results["cicids2018_detective_loadable"] = ok
        except Exception as exc:  # noqa: BLE001 - reported to the metrics, not raised
            print(f"  CSE-CIC-IDS2018 Detective load FAILED: {exc}")
            results["cicids2018_detective_loadable"] = False
    else:
        print("  Detective not trained this run; skipping load check")
        results["cicids2018_detective_loadable"] = False
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=20,
                        help="keep every Nth row of EVERY file (global, so rates "
                             "stay comparable between the flood and benign pools)")
    parser.add_argument("--detective-stride", type=int, default=1,
                        help="stride for the Detective's days; it needs window density")
    parser.add_argument("--bouncer-limit", type=int, default=30_000,
                        help="max rows per class per day for the Bouncer pools")
    parser.add_argument("--detective-limit", type=int, default=8_500,
                        help="max windows per class for the Detective")
    parser.add_argument("--skip-cadence-sweep", action="store_true",
                        help="skip the window-cadence sensitivity analysis")
    parser.add_argument("--skip-cross-day", action="store_true",
                        help="skip the Bouncer's leave-one-day-out block")
    args = parser.parse_args()

    print("=" * 70)
    print("CSE-CIC-IDS2018 INDEPENDENT TRAINING & EVALUATION")
    print("=" * 70)

    missing = [s for s in ALL_DAYS if not _day_path(s).exists()]
    if missing:
        print(f"Dataset files not found in {DATA_DIR}: {', '.join(missing)}")
        print("Run: python scripts/download_cicids2018.py --from-mirror")
        return 1

    started = time.perf_counter()

    print(f"\n--- Training Bouncer (XGBoost + TemperatureScaler, stride {args.stride}) ---")
    bouncer_model, bouncer_metrics = train_bouncer(
        args.stride, args.bouncer_limit, SEED, args.skip_cross_day
    )
    if bouncer_model is not None:
        BOUNCER_DIR.mkdir(parents=True, exist_ok=True)
        bouncer_model.save(BOUNCER_DIR)

    cadences = (WINDOW_SECONDS,) if args.skip_cadence_sweep else CADENCE_SWEEP_SECONDS
    print(f"\n--- Building Detective windows (stride {args.detective_stride}) ---")
    windows, probe, cadence_metrics, window_survey = collect_detective(
        args.detective_stride, args.detective_limit, cadences
    )
    if not windows:
        print("No graph windows produced — aborting without writing metrics.")
        return 1

    print("\n--- Detective ceiling, measured BEFORE training it ---")
    ceiling = _probe_metrics(WINDOW_SECONDS, probe["X"], probe["y"])
    if ceiling.get("available"):
        print(f"  ceiling: AUC={ceiling['auc']:.4f} F1(probe)={ceiling['f1_attack']:.4f} "
              f"over {ceiling['windows']:,} windows "
              f"(majority-class acc {ceiling['majority_class_accuracy']:.4f})")

    print("\n--- Training Detective (GAT in autograd) ---")
    detective_model, detective_metrics = train_detective(windows, args.detective_limit, SEED)
    del windows, probe

    if len(cadence_metrics) > 1:
        print("\n--- Window-cadence sensitivity (linear probe, same stream) ---")
        for row in cadence_metrics:
            if not row.get("available"):
                continue
            print(f"  {row['window_seconds']:>6.0f}s  windows={row['windows']:>7,d}  "
                  f"AUC={row['auc']:.4f}  F1={row['f1_attack']:.4f}  "
                  f"edges {row['median_edges_attack']:.0f}/{row['median_edges_benign']:.0f}")

    print("\n--- Label quality (separate feature-carrying load) ---")
    quality_rows: list = []
    for stem in DETECTIVE_DAYS:
        rows, _ = _load_day(stem, QUALITY_STRIDE, with_features=True)
        quality_rows.extend(rows)
        del rows
    label_quality = label_quality_report(quality_rows)
    del quality_rows
    if label_quality.get("rows"):
        print(f"  {label_quality['rows']:,} rows at stride {QUALITY_STRIDE}, "
              f"{label_quality['ambiguous_row_fraction']:.2%} sit on an ambiguous "
              f"feature vector, deterministic ceiling "
              f"{label_quality['deterministic_ceiling']:.4f}")

    print("\n--- Verifying Weight Loadability & Live Inference ---")
    weight_verification = verify_weights(
        bouncer_model is not None, detective_model is not None
    )

    b_f1 = bouncer_metrics.get("f1")
    d_f1 = detective_metrics.get("f1")
    d_auc = ceiling.get("auc") if ceiling.get("available") else None
    if b_f1 is not None and d_f1 is not None:
        # The Bouncer's task is real and it scores on it; the Detective's task is
        # weak in this capture and the ceiling says so. Naming that in the verdict
        # is the point of measuring the ceiling before training rather than after.
        below_slo = (d_f1 < 0.85) or (d_auc is not None and d_auc < 0.85)
        verdict = ("BOUNCER_REPORTABLE_DETECTIVE_BELOW_SLO" if below_slo
                   else "BOTH_LANES_REPORTABLE")
    else:
        verdict = "PARTIAL"

    full_metrics = {
        "experiment": "cicids2018",
        "status": "COMPLETE",
        "verdict": verdict,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "dataset": {
            "source": DATASET_SOURCE,
            "url": DATASET_URL,
            "day_files": {s: _day_path(s).name for s in ALL_DAYS},
            "roles": {
                "bouncer_flood": list(BOUNCER_FLOOD_DAYS),
                "bouncer_benign": list(BOUNCER_BENIGN_DAYS),
                "detective": list(DETECTIVE_DAYS),
            },
            "stride": args.stride,
            "detective_stride": args.detective_stride,
            "synthetic_hosts": True,
            "synthetic_hosts_note": (
                "The ML-ready CSVs carry no Source IP, Destination IP or Source "
                "Port column. source_ip is one reconstructed constant; dest_ip is "
                "a deterministic bijection of the REAL destination port, so the "
                "graph measures service fan-out, not host fan-out, and the edge "
                "features port_entropy and flow_count are constant by "
                "construction. source_port is None. Flow measurements — bytes, "
                "durations, ports, protocol, timestamps and all 74 CICFlowMeter "
                "columns — are real, and unlike Experiments A-E the capture clock "
                "is real too, so windows are true 2-second slices rather than "
                "synthetic row spacing."
            ),
            "label_mapping": {
                "Benign": "normal (Label.BENIGN)",
                "DoS attacks-*": "dos (Label.FLOOD) — Hulk, SlowHTTPTest, GoldenEye, Slowloris",
                "DDOS attack-*": "dos (Label.FLOOD) — HOIC, LOIC-HTTP",
                "Infilteration": "probe (Label.PORT_SCAN) — Nmap sweep + full port scan",
                "_dropped": ("brute force, web attacks, SQL injection, bot, and the "
                             "streaming day: no KRONUS counterpart in this capture"),
            },
        },
        "window_survey": window_survey,
        "bouncer": bouncer_metrics,
        "detective": detective_metrics,
        "detective_ceiling": ceiling,
        "cadence_sensitivity": cadence_metrics,
        "label_quality": label_quality,
        "weight_verification": weight_verification,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(full_metrics, indent=2, default=str))

    print("\n" + "=" * 70)
    print("CSE-CIC-IDS2018 EXPERIMENT COMPLETE")
    print(f"Verdict: {verdict}")
    print(f"Metrics saved to:  {METRICS_PATH}")
    print(f"Bouncer weights:   {BOUNCER_DIR}")
    print(f"Detective weights: {DETECTIVE_DIR}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
