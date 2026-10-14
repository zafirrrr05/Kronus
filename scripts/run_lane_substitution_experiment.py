#!/usr/bin/env python3
"""Lane-substitution experiment: send the Detective's data to the trained Bouncer.

THE QUESTION
============
KRONUS runs two detection lanes. The Bouncer is the fast lane: XGBoost over six
per-source rate features, read straight off the Event Stream. The Detective is
the deep lane: a hand-rolled GAT over 2-second graph windows, and the only lane
that costs a graph construction per verdict.

Every existing experiment in this repository trains and evaluates each lane on
its OWN traffic (Bouncer on DoS-vs-normal, Detective on benign-vs-port_scan) and
then prints the two scores side by side. That is a *parallel* comparison: the two
numbers describe two different workloads, so the gap between them says nothing
about whether one lane could do the other's job.

This experiment asks the substitution question directly:

    Take the exact data that was routed to the Detective, and send it to the
    already-trained Bouncer instead. How does the fast lane score on the deep
    lane's own test set, measured against the deep lane's own ground truth?

Nothing is retrained for the primary result. The Bouncer artifact is loaded from
disk exactly as served; the Detective artifact is loaded from disk exactly as
served. Both are scored on the identical windows.

BASELINE SELECTION (why these two datasets)
===========================================
Of the external datasets, only Experiments B and C have a trained Bouncer AND a
trained Detective AND a Detective F1 at or near the top of the range:

  Experiment C  UNSW-NB15       Detective F1 1.0000   Bouncer F1 0.9963
  Experiment B  CIC-IDS2017     Detective F1 0.9846   Bouncer F1 0.9987

Experiments D and E carry the next-highest Detective scores (0.9697 and a
withheld 1.0000) but train no Bouncer at all — there is no flood class in either
capture, so a substitution has no trained fast lane to substitute. Experiments F
and H have weaker Detective scores; G is Bouncer-only. So B and C are the two
external datasets with the best Detective *and* a Bouncer to hand it to.

WHAT "THE DETECTIVE'S DATA" MEANS HERE
======================================
The runners build the Detective's test set as:

    test_snaps = _build_graph_snapshots(norm_test, Label.BENIGN)
               + _build_graph_snapshots(probe_test, Label.PORT_SCAN)

so the answerable question is well-defined: those windows, in that order, with
the timestamps the runners already assign them (T0 + i * 20 ms).

Each class's stream is replayed on its own, with its own featurizer — matching
the runners, which build per-class streams and reset the Bouncer's per-source
sliding window for each. Sharing one featurizer across the two classes would be
an artefact of this harness: both streams start at the same T0, so the second
class's early events would be scored against the first class's future history.

Window boundaries come from `_window_index_groups`, which mirrors
`WindowedGraphBuilder.ingest` exactly — including its detail that the event that
closes a window is NOT part of it, it opens the next one — and snapshots are
built with the production `build_snapshot` pure function. So the snapshots handed
to the Detective here are the snapshots it was scored on originally, and the
events inside each one are the events that produced them.

The Bouncer is then handed exactly that event stream, in exactly that order.
Deliberately NOT re-sorted: the runners' Bouncer path sorts rows by source_ip to
make a host's traffic contiguous, but re-ordering the stream would be *changing
the data* rather than substituting the model. `--sorted-sensitivity` measures
what that difference is worth instead of quietly applying it.

WINDOW-LEVEL SCORING
====================
The Bouncer is a per-event model; the Detective's label lives on a window. The
bridge is the window's event set: the Bouncer scores every event in the window
and the window's score is the aggregate. Two aggregates ship:

  max   — the window is flagged if ANY event in it is flagged. This is the
          substitution's most favourable reading (an attack that touches a
          window once should light it up) and is the primary number.
  mean  — the window's average P(flood).

Both are reported against the same window labels the Detective is scored on, so
the two lanes' numbers are directly comparable.

SECONDARY ANALYSIS (why it is here)
===================================
If the shipped Bouncer scores poorly on the deep lane's traffic, that has two
possible readings — "the fast features cannot see this attack" and "the shipped
Bouncer was trained for a different task and was never asked to". They are told
apart by training a fresh Bouncer on the Detective's own training rows and
scoring it on the same test windows. That variant is labelled `retrained` and
kept in its own block so it can never be confused with the substitution result.

Output: results/experiments/lane_substitution_metrics.json
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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from libs.constants import Label
from services.bouncer.features import FEATURE_NAMES, FlowFeaturizer
from services.bouncer.model import BouncerModel
from services.detective.model import DetectiveModel
from services.graph_builder.builder import build_snapshot
from services.telemetry_exporter.converters import flow_row_to_event
from twin.cicids2017 import load_cicids2017
from twin.unsw_nb15 import load_unsw_nb15

RESULTS_DIR = REPO_ROOT / "results" / "experiments"
METRICS_PATH = RESULTS_DIR / "lane_substitution_metrics.json"

TRAIN_RATIO = 0.67
SEED = 42
ROW_SPACING_MS = 20
WINDOW_SECONDS = 2.0
BOUNCER_THRESHOLD = 0.5

# The Detective's own two-class frame, in the runners' order and naming.
BENIGN, ATTACK = "benign", "port_scan"


# --------------------------------------------------------------------------
# Dataset descriptors — every parameter here is copied from the experiment
# runner the baseline metrics came from, so the reproduced test set is the
# one those metrics describe.
# --------------------------------------------------------------------------
DATASETS = {
    "unsw_nb15": {
        "runner": "scripts/run_unsw_nb15_experiment.py",
        "detective_f1_baseline": 1.0,
        "bouncer_f1_baseline": 0.9963,
        "data_dir": REPO_ROOT / "data" / "external" / "unsw_nb15",
        "models_dir": REPO_ROOT / "models" / "experiments" / "unsw_nb15",
        # Baseline run used: run_unsw_nb15_experiment.py --limit 10000 (detective)
        # with the default --bouncer-limit 20000. Only the Detective's own
        # --limit matters here; the Bouncer artifact is loaded, never retrained.
        "detective_limit": 10_000,
        "t0": datetime(2025, 2, 14, 12, 0, 0, tzinfo=timezone.utc),
        "dataset_source": "UNSW-NB15 (Australian Centre for Cyber Security, UNSW Canberra)",
    },
    "cic_ids2017": {
        "runner": "scripts/run_cic_experiment.py",
        "detective_f1_baseline": 0.9846,
        "bouncer_f1_baseline": 0.9987,
        "data_dir": REPO_ROOT / "data" / "external" / "cicids2017",
        "models_dir": REPO_ROOT / "models" / "experiments" / "cic_ids2017",
        # run_cic_experiment.py defaults: --limit None, --detective-limit 20000
        "detective_limit": 20_000,
        "t0": datetime(2017, 7, 3, tzinfo=timezone.utc),
        "dataset_source": "CIC-IDS2017 (GeneratedLabelledFlows/MachineLearningCSV)",
    },
}


# --------------------------------------------------------------------------
# Reproduction of the runners' window construction
# --------------------------------------------------------------------------
def _events_for_rows(rows: list, t0: datetime) -> list:
    """Replay rows into events with the runners' exact timestamp spacing."""
    return [
        flow_row_to_event(row, ts=t0 + timedelta(milliseconds=i * ROW_SPACING_MS))
        for i, row in enumerate(rows)
    ]


def _window_index_groups(events: list, window_seconds: float) -> list[list[int]]:
    """Index groups for the tumbling windows `WindowedGraphBuilder` would emit.

    Mirrors `ingest` exactly: a window closes when an event's timestamp is at
    least `window_seconds` past the window's start, and that closing event opens
    the next window rather than joining the one it closed. The trailing group is
    the builder's `flush`.
    """
    groups: list[list[int]] = []
    cur: list[int] = []
    window_start = None
    for i, ev in enumerate(events):
        if window_start is None:
            window_start = ev.ts
        if (ev.ts - window_start).total_seconds() >= window_seconds and cur:
            groups.append(cur)
            cur = [i]
            window_start = ev.ts
        else:
            cur.append(i)
    if cur:
        groups.append(cur)
    return groups


def _windows_for_stream(events: list, label: str) -> list[tuple]:
    """(snapshot, event_indices, label) for each window carrying at least one edge.

    The `if snap.edges` filter is the runners' own — an edgeless window never
    reaches the Detective, so it must not reach the Bouncer either.
    """
    out = []
    for idxs in _window_index_groups(events, WINDOW_SECONDS):
        evs = [events[i] for i in idxs]
        snap = build_snapshot(evs, window_start=evs[0].ts, window_end=evs[-1].ts)
        if snap.edges:
            out.append((snap, idxs, label))
    return out


def _stratified_split(items: list, ratio: float, seed: int) -> tuple[list, list]:
    """Byte-for-byte the runners' split, so the test rows are the same rows."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items)).tolist()
    n_train = int(len(idx) * ratio)
    train_idx = set(idx[:n_train])
    train = [items[i] for i in range(len(items)) if i in train_idx]
    test = [items[i] for i in range(len(items)) if i not in train_idx]
    return train, test


def _bouncer_features(events: list) -> np.ndarray:
    """Features for one class's stream, replayed through the live FlowFeaturizer.

    A fresh featurizer per stream, exactly as the runners do: the featurizer is
    stateful per source_ip over a 2-second sliding window, and the two classes
    both start at T0, so sharing one would score each class against the other's
    future. Same code path as production and as the runners' own builder.
    """
    featurizer = FlowFeaturizer()
    X = []
    for ev in events:
        feats = featurizer.features_for(ev)
        X.append([feats[name] for name in FEATURE_NAMES])
    return np.array(X) if X else np.empty((0, len(FEATURE_NAMES)))


def _aggregate(scores: np.ndarray, groups: list[list[int]], how: str) -> np.ndarray:
    """Collapse per-event scores onto windows by index group."""
    fn = np.max if how == "max" else np.mean
    return np.array([float(fn(scores[idxs])) if idxs else 0.0 for _, idxs, _ in groups])


def _binary_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Same protocol as the runners' Detective scoring: macro over
    {benign, port_scan}, plus the confusion matrix in that label order."""
    labels = [BENIGN, ATTACK]
    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)])) if y_true else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="binary", pos_label=ATTACK, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4),
        "f1_macro": round(float(f1), 4),
        f"precision_{ATTACK}": round(float(p_pos), 4),
        f"recall_{ATTACK}": round(float(r_pos), 4),
        f"f1_{ATTACK}": round(float(f1_pos), 4),
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm,
    }


METRIC_KEYS = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
               f"precision_{ATTACK}", f"recall_{ATTACK}", f"f1_{ATTACK}"]


def _delta(candidate: dict, baseline: dict) -> dict:
    return {k: round(candidate[k] - baseline[k], 4) for k in METRIC_KEYS}


def _threshold_metrics(y_true: list[str], scores: np.ndarray, thr: float = BOUNCER_THRESHOLD) -> tuple:
    """Binary metrics at `thr`, plus AUC and the best achievable F1 over all
    thresholds — so a poor score can be attributed to the threshold OR to the
    score carrying no signal at all."""
    y_pred = [ATTACK if s >= thr else BENIGN for s in scores]
    metrics = _binary_metrics(y_true, y_pred)

    y_bin = np.array([1 if t == ATTACK else 0 for t in y_true])
    auc = round(float(roc_auc_score(y_bin, scores)), 4) if len(set(y_bin.tolist())) > 1 else None

    best_f1, best_thr = 0.0, None
    for cand in np.unique(scores):
        pred = [ATTACK if s >= cand else BENIGN for s in scores]
        _, _, f1_pos, _ = precision_recall_fscore_support(
            y_true, pred, labels=[BENIGN, ATTACK], average="binary",
            pos_label=ATTACK, zero_division=0,
        )
        if f1_pos > best_f1:
            best_f1, best_thr = float(f1_pos), float(cand)
    return metrics, auc, {"f1": round(best_f1, 4),
                          "threshold": round(best_thr, 6) if best_thr is not None else None}


def _score_stats(scores: np.ndarray, name: str) -> dict:
    """Distribution of a lane's raw score over one class — the evidence behind an
    AUC, so a claim like 'the fast lane is inverted here' can be checked directly."""
    if len(scores) == 0:
        return {"class": name, "n": 0}
    return {
        "class": name,
        "n": int(len(scores)),
        "median": round(float(np.median(scores)), 4),
        "p05": round(float(np.percentile(scores, 5)), 4),
        "p95": round(float(np.percentile(scores, 95)), 4),
        "mean": round(float(np.mean(scores)), 4),
    }


def _score_detective(model: DetectiveModel, windows: list) -> tuple[list[str], float]:
    y_pred = []
    t0 = time.perf_counter()
    for snap, _, _ in windows:
        v = model.predict_verdict(snap, window_id=snap.window_id)
        y_pred.append(v.label if v.label != Label.UNCERTAIN.value else ATTACK)
    return y_pred, time.perf_counter() - t0


# --------------------------------------------------------------------------
# Per-dataset run
# --------------------------------------------------------------------------
def run_dataset(name: str, spec: dict, sorted_sensitivity: bool, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    log("=" * 72)
    log(f"LANE SUBSTITUTION — {name}")
    log("=" * 72)

    data_dir: Path = spec["data_dir"]
    models_dir: Path = spec["models_dir"]
    t0 = spec["t0"]

    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        return {"dataset": name, "status": "NOT_RUN",
                "reason": f"dataset CSVs not present in {data_dir}", "runner": spec["runner"]}

    # ---- load rows exactly as the runner did -----------------------------
    loader = load_cicids2017 if name == "cic_ids2017" else load_unsw_nb15
    t_load = time.perf_counter()
    rows = loader(data_dir)
    load_seconds = time.perf_counter() - t_load
    cats: dict[str, int] = {}
    for r in rows:
        cats[r.category] = cats.get(r.category, 0) + 1
    log(f"  loaded {len(rows):,} flows in {load_seconds:.1f}s  categories={cats}")

    # ---- reproduce the Detective's split --------------------------------
    det_limit = spec["detective_limit"]
    normal_rows = [r for r in rows if r.category == "normal"][:det_limit]
    probe_rows = [r for r in rows if r.category == "probe"][:det_limit]
    if not normal_rows or not probe_rows:
        return {"dataset": name, "status": "NOT_RUN",
                "reason": "dataset lacks a benign or a scan class"}

    norm_train, norm_test = _stratified_split(normal_rows, TRAIN_RATIO, SEED)
    probe_train, probe_test = _stratified_split(probe_rows, TRAIN_RATIO, SEED + 1)

    # ---- the two class streams, each with its own featurizer -------------
    norm_test_events = _events_for_rows(norm_test, t0)
    probe_test_events = _events_for_rows(probe_test, t0)
    norm_train_events = _events_for_rows(norm_train, t0)
    probe_train_events = _events_for_rows(probe_train, t0)

    benign_windows = _windows_for_stream(norm_test_events, BENIGN)
    attack_windows = _windows_for_stream(probe_test_events, ATTACK)
    windows = benign_windows + attack_windows
    y_true = [lbl for _, _, lbl in windows]
    log(f"  detective test windows: {len(windows)} "
        f"(benign={len(benign_windows)}, {ATTACK}={len(attack_windows)})")
    if not windows:
        return {"dataset": name, "status": "NOT_RUN",
                "reason": "no graph windows produced from the test split"}

    result: dict = {
        "experiment": "lane_substitution",
        "dataset": name,
        "dataset_source": spec["dataset_source"],
        "status": "COMPLETE",
        "runner_baseline": spec["runner"],
        "question": (
            "Route the exact event stream that formed the Detective's test windows "
            "to the already-trained Bouncer and score both lanes on those same windows."
        ),
        "method": {
            "split": "identical to the runner (per-class stratified, seed 42, ratio 0.67)",
            "detective_limit_per_class": det_limit,
            "row_spacing_ms": ROW_SPACING_MS,
            "window_seconds": WINDOW_SECONDS,
            "window_rule": "mirrors WindowedGraphBuilder.ingest (the closing event opens the next window)",
            "featurizer": "one FlowFeaturizer per class stream, as the runners do",
            "bouncer_stream_order": "the Detective's own replay order (NOT re-sorted by source_ip)",
            "window_aggregate_primary": "max",
            "window_aggregate_secondary": "mean",
            "bouncer_threshold": BOUNCER_THRESHOLD,
            "scores_are_frozen_artifacts": True,
        },
        "test_set": {
            "windows": len(windows),
            "windows_benign": len(benign_windows),
            "windows_attack": len(attack_windows),
            "events": len(norm_test) + len(probe_test),
            "events_benign": len(norm_test),
            "events_attack": len(probe_test),
        },
        "baseline_reported_by_runner": {
            "detective_f1_macro": spec["detective_f1_baseline"],
            "bouncer_f1_own_task": spec["bouncer_f1_baseline"],
            "note": (
                "The runner's Bouncer number is measured on the Bouncer's OWN "
                "flood-vs-benign test set, NOT on the Detective's windows. That is "
                "the parallel comparison this experiment replaces."
            ),
        },
    }

    # ---- lane A: the Detective, frozen artifact, same windows ------------
    detective_dir = models_dir / "detective"
    if not detective_dir.exists():
        result |= {"status": "NOT_RUN", "reason": f"no trained Detective at {detective_dir}"}
        return result
    detective = DetectiveModel.load(detective_dir)
    det_pred, det_seconds = _score_detective(detective, windows)
    det_metrics = _binary_metrics(y_true, det_pred)
    log(f"  Detective (frozen):  acc={det_metrics['accuracy']}  "
        f"f1_macro={det_metrics['f1_macro']}  cm={det_metrics['confusion_matrix']}")

    # ---- lane B: the shipped Bouncer, frozen, on the Detective's stream ---
    bouncer_dir = models_dir / "bouncer"
    if not bouncer_dir.exists():
        result |= {"status": "NOT_RUN", "reason": f"no trained Bouncer at {bouncer_dir}"}
        return result
    bouncer = BouncerModel.load(bouncer_dir)

    t_f = time.perf_counter()
    benign_scores = bouncer.predict_proba(_bouncer_features(norm_test_events))
    attack_scores = bouncer.predict_proba(_bouncer_features(probe_test_events))
    bouncer_seconds = time.perf_counter() - t_f
    event_scores = np.concatenate([benign_scores, attack_scores])

    max_scores = np.concatenate([
        _aggregate(benign_scores, benign_windows, "max"),
        _aggregate(attack_scores, attack_windows, "max"),
    ])
    mean_scores = np.concatenate([
        _aggregate(benign_scores, benign_windows, "mean"),
        _aggregate(attack_scores, attack_windows, "mean"),
    ])
    max_metrics, max_auc, max_best = _threshold_metrics(y_true, max_scores)
    mean_metrics, mean_auc, mean_best = _threshold_metrics(y_true, mean_scores)
    flow_metrics, flow_auc, flow_best = _threshold_metrics(
        [BENIGN] * len(norm_test) + [ATTACK] * len(probe_test), event_scores
    )
    log(f"  Bouncer (frozen,max):acc={max_metrics['accuracy']}  "
        f"f1_macro={max_metrics['f1_macro']}  auc={max_auc}  cm={max_metrics['confusion_matrix']}")

    n_events = len(event_scores)
    result["lanes"] = {
        "detective": {
            "artifact": str(detective_dir),
            "granularity": "one verdict per 2-second graph window",
            "metrics": det_metrics,
            "inference_seconds_total": round(det_seconds, 4),
            "inference_seconds_per_window_ms": round(det_seconds / max(len(windows), 1) * 1000, 4),
        },
        "bouncer_substitution": {
            "artifact": str(bouncer_dir),
            "granularity": "one score per event, aggregated to the window",
            "aggregate_primary": "max",
            "metrics": max_metrics,
            "auc": max_auc,
            "best_threshold": max_best,
            "aggregate_secondary": "mean",
            "metrics_mean": mean_metrics,
            "auc_mean": mean_auc,
            "best_threshold_mean": mean_best,
            "events_scored": int(n_events),
            "inference_seconds_total": round(bouncer_seconds, 4),
            "inference_seconds_per_event_ms": round(bouncer_seconds / max(n_events, 1) * 1000, 6),
            "flow_level": {
                "note": "each test flow scored directly (no windowing) — the finest grain available",
                "metrics": flow_metrics,
                "auc": flow_auc,
                "best_threshold": flow_best,
            },
            "score_distribution_per_class": [
                _score_stats(max_scores[: len(benign_windows)], BENIGN),
                _score_stats(max_scores[len(benign_windows):], ATTACK),
            ],
            "event_score_distribution_per_class": [
                _score_stats(benign_scores, f"{BENIGN}_events"),
                _score_stats(attack_scores, f"{ATTACK}_events"),
            ],
        },
    }

    result["substitution_delta_vs_detective"] = {
        "note": "bouncer_substitution minus detective, on identical windows",
        "max_aggregate": _delta(max_metrics, det_metrics),
        "mean_aggregate": _delta(mean_metrics, det_metrics),
        "flow_level": _delta(flow_metrics, det_metrics),
    }
    result["cost"] = {
        "detective_ms_per_window": round(det_seconds / max(len(windows), 1) * 1000, 4),
        "bouncer_ms_per_event": round(bouncer_seconds / max(n_events, 1) * 1000, 6),
        "events_per_window_observed": round(n_events / max(len(windows), 1), 2),
        "note": (
            "The Detective's cost is per window and includes graph inference; the "
            "Bouncer's is per event and needs no graph. Both are wall-clock on this "
            "machine over the whole test set, CPU only."
        ),
    }

    # ---- secondary: a Bouncer retrained on the Detective's own rows -------
    X_tr = np.vstack([_bouncer_features(norm_train_events),
                      _bouncer_features(probe_train_events)])
    y_tr = np.array([0] * len(norm_train) + [1] * len(probe_train))
    fresh = BouncerModel()
    t_fit = time.perf_counter()
    fresh.fit(X_tr, y_tr)
    retrain_seconds = time.perf_counter() - t_fit

    fresh_scores = np.concatenate([
        fresh.predict_proba(_bouncer_features(norm_test_events)),
        fresh.predict_proba(_bouncer_features(probe_test_events)),
    ])
    fresh_max = np.concatenate([
        _aggregate(fresh_scores[: len(norm_test)], benign_windows, "max"),
        _aggregate(fresh_scores[len(norm_test):], attack_windows, "max"),
    ])
    fresh_metrics, fresh_auc, fresh_best = _threshold_metrics(y_true, fresh_max)
    fresh_flow_metrics, fresh_flow_auc, fresh_flow_best = _threshold_metrics(
        [BENIGN] * len(norm_test) + [ATTACK] * len(probe_test), fresh_scores
    )
    log(f"  Bouncer (retrained): acc={fresh_metrics['accuracy']}  "
        f"f1_macro={fresh_metrics['f1_macro']}  auc={fresh_auc}  cm={fresh_metrics['confusion_matrix']}")

    result["retrained_variant"] = {
        "note": (
            "SEPARATE from the substitution result above. A fresh Bouncer with the same "
            "six features and the same XGBoost recipe, trained on the Detective's own "
            "training rows (benign vs port_scan) and scored on the same test windows. "
            "This isolates 'the fast features cannot see a scan' from 'the shipped "
            "Bouncer was never asked to'."
        ),
        "train_rows": int(len(y_tr)),
        "train_attack": int(y_tr.sum()),
        "train_benign": int((y_tr == 0).sum()),
        "fit_seconds": round(retrain_seconds, 3),
        "metrics": fresh_metrics,
        "auc": fresh_auc,
        "best_threshold": fresh_best,
        "flow_level": {"metrics": fresh_flow_metrics, "auc": fresh_flow_auc,
                       "best_threshold": fresh_flow_best},
        "delta_vs_detective": _delta(fresh_metrics, det_metrics),
    }

    # ---- optional sensitivity: the Bouncer's native (source_ip-sorted) order
    if sorted_sensitivity:
        s_norm = sorted(norm_test, key=lambda r: r.source_ip)
        s_probe = sorted(probe_test, key=lambda r: r.source_ip)
        s_norm_events = _events_for_rows(s_norm, t0)
        s_probe_events = _events_for_rows(s_probe, t0)
        s_benign = _windows_for_stream(s_norm_events, BENIGN)
        s_attack = _windows_for_stream(s_probe_events, ATTACK)
        s_windows = s_benign + s_attack
        s_true = [lbl for _, _, lbl in s_windows]

        s_scores = np.concatenate([
            bouncer.predict_proba(_bouncer_features(s_norm_events)),
            bouncer.predict_proba(_bouncer_features(s_probe_events)),
        ])
        s_max = np.concatenate([
            _aggregate(s_scores[: len(s_norm)], s_benign, "max"),
            _aggregate(s_scores[len(s_norm):], s_attack, "max"),
        ])
        s_metrics, s_auc, s_best = _threshold_metrics(s_true, s_max)
        s_det_pred, _ = _score_detective(detective, s_windows)
        s_det_metrics = _binary_metrics(s_true, s_det_pred)
        result["sorted_order_sensitivity"] = {
            "note": (
                "ABLATION, not a result. The Bouncer's own training path sorts rows by "
                "source_ip so a host's traffic is contiguous; the substitution above "
                "deliberately does not, because re-ordering is changing the data. This "
                "block re-runs BOTH lanes over the source_ip-sorted stream so the size of "
                "that ordering effect is measured rather than assumed."
            ),
            "windows": len(s_windows),
            "bouncer_metrics": s_metrics,
            "bouncer_auc": s_auc,
            "bouncer_best_threshold": s_best,
            "detective_metrics": s_det_metrics,
            "bouncer_delta_vs_detective": _delta(s_metrics, s_det_metrics),
        }

    log("")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS),
                        choices=list(DATASETS), help="which datasets to run")
    parser.add_argument("--cic-dir", type=Path, default=None,
                        help="override the CIC-IDS2017 CSV directory")
    parser.add_argument("--sorted-sensitivity", action="store_true",
                        help="also run the source_ip-sorted-stream ablation")
    args = parser.parse_args()

    if args.cic_dir is not None:
        DATASETS["cic_ids2017"]["data_dir"] = args.cic_dir
    else:
        # The repo-local path is preferred, but this machine keeps the 1.1 GB
        # capture elsewhere — use it rather than copying a gigabyte around.
        fallback = Path.home() / "Desktop" / "Kronus" / "data" / "external" / "cicids2017"
        if not list(DATASETS["cic_ids2017"]["data_dir"].glob("*.csv")) and fallback.exists():
            DATASETS["cic_ids2017"]["data_dir"] = fallback

    print("=" * 72)
    print("LANE SUBSTITUTION EXPERIMENT — the Detective's data, sent to the Bouncer")
    print("=" * 72)

    results = {name: run_dataset(name, DATASETS[name], args.sorted_sensitivity)
               for name in args.datasets}

    payload = {
        "experiment": "lane_substitution",
        "status": "COMPLETE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Routes the exact event stream that formed each baseline experiment's "
            "Detective test windows to the already-trained Bouncer, scores both lanes "
            "on those identical windows, and reports the delta."
        ),
        "datasets": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nMetrics saved to: {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
