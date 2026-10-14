#!/usr/bin/env python3
"""Diagnostic: why did the frozen Bouncer catch every UNSW-NB15 scan window?

Hypothesis under test: UNSW-NB15's pre-split CSVs carry no addresses, so
twin/unsw_nb15.py reconstructs them, and for the `probe` category it pins
`session` to 0 — collapsing every scan row onto a handful of synthetic sources.
The Bouncer's `event_rate` is per-source over a 2-second window, so a handful of
sources carrying thousands of rows reads as an extreme-rate flood. If that is
what happened, the substitution's success on UNSW-NB15 is a property of the
reconstruction, not of the fast features understanding a scan.

CIC-IDS2017 ships real addresses, so the same measurement there is the control.

Usage: python scratch/explain_substitution.py
"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from services.bouncer.features import FEATURE_NAMES, FlowFeaturizer
from services.telemetry_exporter.converters import flow_row_to_event
from twin.unsw_nb15 import load_unsw_nb15

TRAIN_RATIO, SEED, ROW_SPACING_MS = 0.67, 42, 20
T0 = datetime(2025, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
DATA = REPO_ROOT / "data" / "external" / "unsw_nb15"


def _split(items, ratio, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items)).tolist()
    n = int(len(idx) * ratio)
    tr = set(idx[:n])
    return ([items[i] for i in range(len(items)) if i in tr],
            [items[i] for i in range(len(items)) if i not in tr])


def profile(rows, label, n_sources_note):
    """Replay a class stream and report the Bouncer's feature distribution."""
    featurizer = FlowFeaturizer()
    rates, dests, ports = [], [], []
    for i, row in enumerate(rows):
        ev = flow_row_to_event(row, ts=T0 + timedelta(milliseconds=i * ROW_SPACING_MS))
        f = featurizer.features_for(ev)
        rates.append(f["event_rate"])
        dests.append(f["unique_dest_count"])
        ports.append(f["dest_port_entropy"])
    srcs = {r.source_ip for r in rows}
    print(f"\n  {label}: {len(rows):,} rows")
    print(f"    distinct source_ip      : {len(srcs):,}  "
          f"({len(rows) / max(len(srcs), 1):,.0f} rows per source)")
    print(f"    top source share        : "
          f"{Counter(r.source_ip for r in rows).most_common(1)[0][1] / len(rows):.1%}")
    print(f"    event_rate   median     : {np.median(rates):.2f}   p95: {np.percentile(rates, 95):.2f}")
    print(f"    unique_dest  median     : {np.median(dests):.2f}")
    print(f"    port_entropy median     : {np.median(ports):.2f}")
    print(f"    {n_sources_note}")


def main() -> int:
    rows = load_unsw_nb15(DATA)
    normal = [r for r in rows if r.category == "normal"][:10_000]
    probe = [r for r in rows if r.category == "probe"][:10_000]
    _, normal_test = _split(normal, TRAIN_RATIO, SEED)
    _, probe_test = _split(probe, TRAIN_RATIO, SEED + 1)

    print("=" * 72)
    print("UNSW-NB15 — the Detective's test stream, as the Bouncer sees it")
    print("=" * 72)
    print("  (hosts are RECONSTRUCTED: the pre-split CSVs carry no addresses)")
    profile(normal_test, "benign test", "control: normal rows are NOT collapsed")
    profile(probe_test, "scan test  ", "the scan class IS collapsed onto few sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
