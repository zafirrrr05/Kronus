#!/usr/bin/env python3
"""CIC-DDoS2019 Independent ML Experiment for KRONUS.

EXPERIMENT: CIC-DDoS2019 External Validation (Experiment G)
===========================================================
Trains a fresh, independent KRONUS Bouncer on CIC-DDoS2019 (Canadian Institute
for Cybersecurity, University of New Brunswick): the reflection/amplification
DDoS benchmark.

Dataset URL: https://www.unb.ca/cic/datasets/ddos-2019.html

    Sharafaldin, Lashkari, Hakak, Ghorbani, "Developing Realistic Distributed
    Denial of Service (DDoS) Attack Dataset and Taxonomy", ICCST 2019.
    doi:10.1109/ccst.2019.8888419

WHY THIS IS A BOUNCER-ONLY EXPERIMENT
-------------------------------------
Every labelled attack family here is volumetric — reflection and amplification
floods (LDAP, MSSQL, NetBIOS, Portmap, SNMP, SSDP, DNS, NTP, TFTP) and direct
floods (UDP, UDPLag, Syn, WebDDoS). There is no port-scan class and no
lateral-movement class anywhere in it. The Bouncer's contract is flood-vs-benign
and this dataset is made of exactly that; the Detective has nothing to train on,
so it is skipped and the metrics record why.

This is the mirror image of Experiment D (DoHBrw2020), which was Detective-only
for the opposite reason: a tunnel capture has no volumetric flood. A reader
comparing the two experiments is looking at the same decision made twice from
opposite evidence, which is the point of running both.

WHAT THIS DATASET ADDS OVER EXPERIMENT F
----------------------------------------
Experiment F (CSE-CIC-IDS2018) had to reconstruct host identity, and its benign
traffic was concurrent with the flood, so a benign holdout was forced. This
release carries the REAL Source IP, Source Port, Destination IP, Destination
Port and a real microsecond clock. The Bouncer featurizer keys its 2-second
window by `source_ip`, so here a benign flow's window contains that host's own
traffic and the whole contamination class is absent. That is a property of the
data, not a choice made below — `host_overlap_report` measures it rather than
asserting it, and its result ships in the metrics.

THE TWO DAYS, AND WHY BOTH ARE LOADED
-------------------------------------
  01-12   2019-01-12 capture: the DrDoS families' first day
  03-11   2019-03-11 capture: the same families' second day

A single day lets a model separate the classes by *capture date* rather than by
traffic shape. Two days does not abolish that risk either, so the `cross_day`
block tests it directly: fit on one day, score on the other, in both
directions, and report what that costs.

WHY THE STRIDE IS GLOBAL, AND WHY IT IS THE ONLY ROW KNOB
---------------------------------------------------------
Every file in both archives is thinned by the SAME stride, so a flood row and a
benign row are decimated identically and their rates stay comparable. The
Bouncer's `event_rate` and `byte_rate` are functions of the replay stream's
density, so a per-file row budget would thin a large attack file harder than a
small benign one and manufacture a rate difference that is not in the data.
That is why `--stride` exists and `sample_per_file` is not used here; the
distinction is documented on `_iter_chunks` in twin/cicddos2019.py.

BENIGN IS SCARCE HERE, AND THAT IS LOADED FOR, NOT ASSUMED
----------------------------------------------------------
CIC-DDoS2019 is overwhelmingly attack traffic; benign is a small minority. A
uniform stride thins benign along with everything else, so the stride cannot be
chosen from the row count alone. The runner therefore measures the surviving
benign count and REFUSES to write metrics if either class came back too thin to
score — rather than reporting a number computed from a handful of rows.

MEMORY: ONE STREAMING PASS PER ARCHIVE
--------------------------------------
Each archive is loaded once, replayed once through a fresh FlowFeaturizer in
true chronological order, and only the resulting 6-float feature vectors are
retained — not the rows. That keeps peak memory proportional to the bounded
sample rather than to the archive, which matters on a laptop, and it means the
train/test split partitions already-computed features. Computing features per
split (the Experiment F arrangement) would featurize the training rows at 67%
density and the test rows at 33%, which is a real, if uniform, distortion of
every rate feature; doing it once avoids that entirely.

TRAIN/TEST SPLIT
  Bouncer: stratified 67/33 per class, seeded, taken per archive so no day's
           rows land mostly on one side; plus the leave-one-day-out block.

WEIGHT ARTIFACTS
  models/experiments/cicddos2019/bouncer/   (bouncer.json, calibration.json)

METRICS
  results/experiments/cicddos2019_metrics.json
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

from services.bouncer.features import FEATURE_NAMES, FlowFeaturizer
from services.bouncer.model import BouncerModel
from services.telemetry_exporter.converters import flow_row_to_event
from twin.cicddos2019 import (
    host_overlap_report,
    label_quality_report,
    load_cicddos2019_timed,
)

DATA_DIR = REPO_ROOT / "data" / "external" / "cicddos2019"
MODELS_DIR = REPO_ROOT / "models" / "experiments" / "cicddos2019"
BOUNCER_DIR = MODELS_DIR / "bouncer"
RESULTS_DIR = REPO_ROOT / "results" / "experiments"
METRICS_PATH = RESULTS_DIR / "cicddos2019_metrics.json"

DATASET_URL = "https://www.unb.ca/cic/datasets/ddos-2019.html"
DATASET_SOURCE = "CIC-DDoS2019 (Canadian Institute for Cybersecurity, UNB)"

# The two capture days, one archive each. Both are loaded for every lane: a
# single day would let the model separate flood from benign by capture date.
DAY_ARCHIVES: tuple[str, ...] = ("CSV-01-12.zip", "CSV-03-11.zip")
DAY_STEMS: tuple[str, ...] = tuple(Path(a).stem for a in DAY_ARCHIVES)

TRAIN_RATIO = 0.67
SEED = 42
QUALITY_STRIDE = 40             # stride for the feature-carrying label-quality load

# A score computed from a handful of rows is not a measurement. Below these
# counts the runner refuses to write metrics rather than reporting a number the
# sample cannot support.
MIN_ROWS_PER_CLASS_PER_DAY = 200


def _archive_path(name: str) -> Path:
    return DATA_DIR / name


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

    A prefix would take the first n items, and the vectors arrive in time order
    — so a prefix is a time-of-day sample, not a sample. This keeps the whole
    span, matching the repo's bounded-sample rule elsewhere.
    """
    if n >= len(items) or n <= 0:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).astype(int)
    return [items[i] for i in idx]


def _label_census(rows: list) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r.raw_label] = out.get(r.raw_label, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _collect_day(
    stem: str, archive: Path, stride: int, limit: int | None
) -> tuple[list, list, dict]:
    """Stream one archive once: real features per row, then keep the vectors.

    The featurizer is fresh per archive — the two archives are captures weeks
    apart, and carrying window state across the join would fabricate events.
    Within an archive the release's own clock orders the replay, so a 2-second
    window is a real 2 seconds and `event_rate` is a real burst rate.

    Only the 6-float vectors survive this function. The rows themselves are
    dropped at the end of the loop, so peak memory is the bounded sample rather
    than the archive.
    """
    rows, times = load_cicddos2019_timed(
        archive, sample_per_file=None, stride=stride, with_features=False,
    )
    if not rows:
        return [], [], {"rows_loaded": 0}

    flood: list[np.ndarray] = []
    benign: list[np.ndarray] = []
    benign_sources: list[str] = []   # parallel to `benign`; benign is rare, so free
    flood_hosts: set[str] = set()
    benign_hosts: set[str] = set()

    featurizer = FlowFeaturizer()
    for row, ts in zip(rows, times, strict=True):
        # Every row drives the featurizer, whatever its label: the window a
        # benign flow lands in is built from the traffic that actually shares
        # it, and skip-featurizing the other class would thin the stream.
        feats = featurizer.features_for(flow_row_to_event(row, ts=ts))
        vector = np.array([feats[name] for name in FEATURE_NAMES])
        if row.category == "dos":
            flood.append(vector)
            flood_hosts.add(row.source_ip)
        elif row.category == "normal":
            benign.append(vector)
            benign_sources.append(row.source_ip)
            benign_hosts.add(row.source_ip)

    # A host that floods also carries a little benign traffic. On this release
    # that is exactly one host per day — the victim (`192.168.50.1` on 01-12,
    # `192.168.50.4` on 03-11), which appears as a source on both attack and
    # benign rows. Its benign rows land in the same 2-second windows as its own
    # flood, so keeping them would ask the Bouncer to call a window benign that
    # is full of flood. They are well under 1% of the benign class, so dropping
    # them costs nothing and makes "no benign window contains flood traffic"
    # true rather than nearly true. Both the drop and the overlap it removes are
    # counted here and reported in the metrics.
    hosts_before_exclusion = set(benign_sources)
    contaminated = sum(1 for src in benign_sources if src in flood_hosts)
    if contaminated:
        keep = [i for i, src in enumerate(benign_sources) if src not in flood_hosts]
        benign = [benign[i] for i in keep]
        benign_sources = [benign_sources[i] for i in keep]
    benign_hosts = set(benign_sources)

    survey = {
        "rows_loaded": len(rows),
        "flood_rows": len(flood),
        "benign_rows": len(benign),
        "benign_fraction": round(len(benign) / max(len(rows), 1), 6),
        "flood_source_hosts": len(flood_hosts),
        "benign_source_hosts": len(benign_hosts),
        "shared_source_hosts": len(flood_hosts & benign_hosts),
        "benign_rows_dropped_on_flood_hosts": contaminated,
        "shared_source_hosts_before_exclusion": len(flood_hosts & hosts_before_exclusion),
        "raw_labels": _label_census(rows),
        "capture_start": str(min(times)) if times else None,
        "capture_end": str(max(times)) if times else None,
    }
    del rows, times, featurizer

    if limit:
        flood = _even_subsample(flood, limit)
        benign = _even_subsample(benign, limit)
    return flood, benign, survey


def collect_bouncer(stride: int, limit: int | None) -> tuple[dict, dict, dict]:
    """Per-day flood and benign feature-vector pools, plus the day survey."""
    flood: dict[str, list] = {}
    benign: dict[str, list] = {}
    survey: dict[str, dict] = {}

    for name, stem in zip(DAY_ARCHIVES, DAY_STEMS, strict=True):
        archive = _archive_path(name)
        if not archive.exists():
            continue
        day_flood, day_benign, day_survey = _collect_day(stem, archive, stride, limit)
        survey[stem] = day_survey
        if day_flood:
            flood[stem] = day_flood
        if day_benign:
            benign[stem] = day_benign
        print(f"  {stem}: {day_survey.get('rows_loaded', 0):,} rows -> "
              f"{len(day_flood):,} flood / {len(day_benign):,} benign "
              f"({day_survey.get('benign_fraction', 0):.3%} benign, "
              f"{day_survey.get('benign_rows_dropped_on_flood_hosts', 0)} "
              f"benign rows dropped on flood hosts)")

    return flood, benign, survey


def _split_by_day(pools: dict, seed: int, take: str = "train") -> dict[str, list]:
    """Split each day's pool 67/33 independently, preserving the day key."""
    out: dict[str, list] = {}
    for stem, vectors in pools.items():
        train, test = _stratified_split(vectors, TRAIN_RATIO, seed)
        chosen = train if take == "train" else test
        if chosen:
            out[stem] = chosen
    return out


def _stack(pools: dict) -> np.ndarray:
    """Concatenate a per-day pool of feature vectors, or an empty matrix."""
    parts = [np.vstack(v) for v in pools.values() if v]
    if not parts:
        return np.empty((0, len(FEATURE_NAMES)))
    return np.vstack(parts)


def _score(model: BouncerModel, X: np.ndarray, y: np.ndarray) -> dict:
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


def _fit_bouncer(flood: dict, benign: dict, seed: int) -> tuple[BouncerModel, dict]:
    """Fit on the given per-day pools and score on a stratified 67/33 split.

    The split is taken per day rather than over the pooled rows. Splitting the
    pool would let one day's rows land mostly in train and another's mostly in
    test, converting a day difference into a test-set difference; splitting each
    day separately keeps every day's contribution proportional on both sides.
    """
    flood_train = _split_by_day(flood, seed)
    flood_test = _split_by_day(flood, seed, take="test")
    benign_train = _split_by_day(benign, seed + 1)
    benign_test = _split_by_day(benign, seed + 1, take="test")

    Xf_train, Xb_train = _stack(flood_train), _stack(benign_train)
    Xf_test, Xb_test = _stack(flood_test), _stack(benign_test)
    X_tr = np.vstack([Xf_train, Xb_train])
    y_tr = np.concatenate([np.ones(len(Xf_train), int), np.zeros(len(Xb_train), int)])
    X_te = np.vstack([Xf_test, Xb_test])
    y_te = np.concatenate([np.ones(len(Xf_test), int), np.zeros(len(Xb_test), int)])
    if not y_tr.any() or not (y_tr == 0).any():
        raise ValueError("both a flood and a benign training pool are required")

    model = BouncerModel()
    started = time.perf_counter()
    model.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - started

    metrics = _score(model, X_te, y_te)
    metrics.update({
        "n_train": int(len(y_tr)),
        "n_train_flood": int(y_tr.sum()),
        "n_train_benign": int((y_tr == 0).sum()),
        "train_seconds": round(elapsed, 3),
    })
    return model, metrics


def train_bouncer(
    stride: int, limit: int | None, seed: int, skip_cross_day: bool
) -> tuple[BouncerModel | None, dict]:
    flood, benign, survey = collect_bouncer(stride, limit)

    # A day that came back without enough of a class is EXCLUDED and reported,
    # not fatal. The two capture days do not carry the same families (01-12 is
    # the DrDoS set, 03-11 the bare-named set), so one day can legitimately be
    # thin in benign while the other is not — aborting the whole experiment over
    # that would throw away a usable run. What is fatal is having no day at all
    # that can carry both classes.
    day_counts = {
        stem: {
            "flood": len(flood.get(stem, [])),
            "benign": len(benign.get(stem, [])),
        }
        for stem in DAY_STEMS
    }
    usable = [
        stem for stem in DAY_STEMS
        if day_counts[stem]["flood"] >= MIN_ROWS_PER_CLASS_PER_DAY
        and day_counts[stem]["benign"] >= MIN_ROWS_PER_CLASS_PER_DAY
    ]
    excluded = {stem: day_counts[stem] for stem in DAY_STEMS if stem not in usable}

    if not usable:
        return None, {
            "skipped": "no day carries both classes at the required depth",
            "min_rows_per_class_per_day": MIN_ROWS_PER_CLASS_PER_DAY,
            "day_counts": day_counts,
            "day_survey": survey,
        }
    if excluded:
        print(f"  excluded (too thin to train on): {excluded}")
    flood = {stem: flood[stem] for stem in usable}
    benign = {stem: benign[stem] for stem in usable}

    model, metrics = _fit_bouncer(flood, benign, seed)
    print(f"  Bouncer: Acc={metrics['accuracy']:.4f} Prec={metrics['precision']:.4f} "
          f"Rec={metrics['recall']:.4f} F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")

    # The evidence behind that score: class-conditional medians. A model that
    # separates on traffic shape and one that separates on a day fingerprint can
    # both score 0.99; these medians and the cross-day block tell them apart, so
    # they ship with the number rather than as an afterthought.
    Xf, Xb = _stack(flood), _stack(benign)
    metrics["feature_medians"] = {
        name: {
            "flood": round(float(np.median(Xf[:, j])), 4),
            "benign": round(float(np.median(Xb[:, j])), 4),
        }
        for j, name in enumerate(FEATURE_NAMES)
    }

    cross: list[dict] = []
    if not skip_cross_day:
        for test_stem in DAY_STEMS:
            train_stems = [s for s in DAY_STEMS if s != test_stem]
            # An empty training side would fit on nothing; a test day missing a
            # class cannot be scored. Both are skipped and visible in the survey
            # rather than silently producing a number.
            if not train_stems:
                continue
            if not all(s in flood and s in benign for s in train_stems):
                continue
            if test_stem not in flood or test_stem not in benign:
                print(f"  cross-day -> {test_stem}: skipped (day lacks a class)")
                continue
            # Fit on the training day(s) only, then score the day the model has
            # never seen. Every row of a training day is training data and every
            # row of the test day is unseen, so this block bypasses the 67/33
            # split inside _fit_bouncer.
            X_tr = np.vstack([_stack({s: flood[s] for s in train_stems}),
                              _stack({s: benign[s] for s in train_stems})])
            y_tr = np.concatenate([
                np.ones(sum(len(flood[s]) for s in train_stems), int),
                np.zeros(sum(len(benign[s]) for s in train_stems), int),
            ])
            unseen = BouncerModel().fit(X_tr, y_tr)
            X_te = np.vstack([_stack({test_stem: flood[test_stem]}),
                              _stack({test_stem: benign[test_stem]})])
            y_te = np.concatenate([np.ones(len(flood[test_stem]), int),
                                   np.zeros(len(benign[test_stem]), int)])
            row = {
                "trained_on": train_stems,
                "tested_on": test_stem,
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
            }
            row.update(_score(unseen, X_te, y_te))
            cross.append(row)
            print(f"  cross-day {'+'.join(train_stems)} -> {test_stem}: "
                  f"F1={row['f1']:.4f} AUC={row['auc']:.4f}")

    metrics.update({
        "component": "bouncer",
        "training_origin": "cicddos2019",
        "dataset_source": DATASET_SOURCE,
        "dataset_url": DATASET_URL,
        "model_type": "XGBoost + Temperature Scaling (trained from scratch)",
        "task": "binary volumetric DDoS flood vs benign classification",
        "synthetic_hosts": False,
        "seed": seed,
        "train_ratio": TRAIN_RATIO,
        "stride": stride,
        "features": FEATURE_NAMES,
        "cross_day": cross,
        "day_survey": survey,
        "day_counts": day_counts,
        "days_used": usable,
        "days_excluded_as_too_thin": excluded,
        "min_rows_per_class_per_day": MIN_ROWS_PER_CLASS_PER_DAY,
        "weight_artifacts": [
            str(BOUNCER_DIR / "bouncer.json"),
            str(BOUNCER_DIR / "calibration.json"),
        ],
    })
    return model, metrics


# --- weight verification ---------------------------------------------------

def verify_weights(bouncer_trained: bool) -> dict:
    results: dict[str, bool] = {}
    if bouncer_trained:
        try:
            model = BouncerModel.load(BOUNCER_DIR)
            verdict = model.predict_verdict(
                {name: 10.0 for name in FEATURE_NAMES}, window_id="verify-cicddos2019-bouncer"
            )
            ok = 0.0 <= float(verdict.confidence) <= 1.0
            print(f"  CIC-DDoS2019 Bouncer load: {'PASSED' if ok else 'FAILED'} "
                  f"(verdict={verdict.label}, conf={verdict.confidence:.3f})")
            results["cicddos2019_bouncer_loadable"] = ok
        except Exception as exc:  # noqa: BLE001 - reported to the metrics, not raised
            print(f"  CIC-DDoS2019 Bouncer load FAILED: {exc}")
            results["cicddos2019_bouncer_loadable"] = False
    else:
        print("  Bouncer not trained this run; skipping load check")
        results["cicddos2019_bouncer_loadable"] = False
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=40,
                        help="keep every Nth row of EVERY file (global, so the "
                             "flood and benign pools are decimated identically)")
    parser.add_argument("--bouncer-limit", type=int, default=60_000,
                        help="max vectors per class per day for the Bouncer pools")
    parser.add_argument("--skip-cross-day", action="store_true",
                        help="skip the Bouncer's leave-one-day-out block")
    args = parser.parse_args()

    print("=" * 70)
    print("CIC-DDoS2019 INDEPENDENT TRAINING & EVALUATION")
    print("=" * 70)

    missing = [a for a in DAY_ARCHIVES if not _archive_path(a).exists()]
    if missing:
        print(f"Dataset archives not found in {DATA_DIR}: {', '.join(missing)}")
        print("Run: python scripts/download_cicddos2019.py --register \\")
        print("         --first-name ... --last-name ... --email ... \\")
        print("         --institution ... --job-title ... --country ...")
        return 1

    started = time.perf_counter()

    print(f"\n--- Training Bouncer (XGBoost + TemperatureScaler, stride {args.stride}) ---")
    bouncer_model, bouncer_metrics = train_bouncer(
        args.stride, args.bouncer_limit, SEED, args.skip_cross_day
    )
    if bouncer_model is None:
        print(f"\nBouncer not trained: {bouncer_metrics.get('skipped')}")
        print("No metrics written — a score computed from too few rows is not a measurement.")
        return 1
    BOUNCER_DIR.mkdir(parents=True, exist_ok=True)
    bouncer_model.save(BOUNCER_DIR)

    print("\n--- Label quality (separate feature-carrying load) ---")
    quality_rows: list = []
    for name in DAY_ARCHIVES:
        rows, _ = load_cicddos2019_timed(
            _archive_path(name), sample_per_file=None, stride=QUALITY_STRIDE,
            with_features=True,
        )
        quality_rows.extend(rows)
        del rows
    label_quality = label_quality_report(quality_rows)
    host_overlap = host_overlap_report(quality_rows)
    del quality_rows
    if label_quality.get("rows"):
        print(f"  {label_quality['rows']:,} rows at stride {QUALITY_STRIDE}, "
              f"{label_quality['ambiguous_row_fraction']:.2%} sit on an ambiguous "
              f"feature vector, deterministic ceiling "
              f"{label_quality['deterministic_ceiling']:.4f}")
    print(f"  host overlap: {host_overlap['attack_hosts']} attack hosts, "
          f"{host_overlap['benign_hosts']} benign hosts, "
          f"{host_overlap['shared_hosts']} shared")

    print("\n--- Verifying Weight Loadability & Live Inference ---")
    weight_verification = verify_weights(bouncer_model is not None)

    b_f1 = bouncer_metrics.get("f1")
    verdict = "BOUNCER_REPORTABLE" if (b_f1 is not None and b_f1 >= 0.85) else "BOUNCER_BELOW_SLO"

    full_metrics = {
        "experiment": "cicddos2019",
        "status": "COMPLETE",
        "verdict": verdict,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "dataset": {
            "source": DATASET_SOURCE,
            "url": DATASET_URL,
            "day_archives": {s: _archive_path(a).name
                             for a, s in zip(DAY_ARCHIVES, DAY_STEMS, strict=True)},
            "stride": args.stride,
            "synthetic_hosts": False,
            "synthetic_hosts_note": (
                "Source IP, Source Port, Destination IP, Destination Port and a "
                "wall-clock microsecond Timestamp are all real in this release, "
                "so nothing is reconstructed. The Bouncer's 2-second window is "
                "keyed by the real source_ip and a 2-second window is a real 2 "
                "seconds."
            ),
            "label_mapping": {
                "BENIGN": "normal (Label.BENIGN)",
                "DrDoS_* / bare family names": (
                    "dos (Label.FLOOD) — LDAP, MSSQL, NetBIOS, Portmap, SNMP, "
                    "SSDP, DNS, NTP, TFTP reflection/amplification"
                ),
                "UDP / UDPLag / Syn / WebDDoS": "dos (Label.FLOOD) — direct floods",
                "_dropped": (
                    "any label outside that vocabulary is dropped and counted in "
                    "day_survey[<day>].raw_labels, never assumed to be a flood"
                ),
            },
            "detective_skipped_reason": (
                "Every labelled attack family in CIC-DDoS2019 is volumetric. "
                "There is no port-scan or lateral-movement class, so the "
                "Detective has no attack class to train on. The mirror image of "
                "Experiment D, which was Detective-only because a tunnel capture "
                "has no flood."
            ),
        },
        "bouncer": bouncer_metrics,
        "label_quality": label_quality,
        "host_overlap": host_overlap,
        "weight_verification": weight_verification,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(full_metrics, indent=2, default=str))

    print("\n" + "=" * 70)
    print("CIC-DDoS2019 EXPERIMENT COMPLETE")
    print(f"Verdict: {verdict}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Bouncer weights:  {BOUNCER_DIR}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
