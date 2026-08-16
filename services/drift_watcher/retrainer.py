"""features.txt component 9: "continuously measures live accuracy using
known attacks the Twin keeps injecting, retrains automatically on drift,
and promotes a new model only if it beats the current one on held-out
probe traffic — now with a third free training signal... every Decoy
interaction." This module is that loop's decision logic — given current
counts and drift scores, produce the spec.md §3.8 DriftReport and decide
whether a retrain is warranted. Actually running training is
services/bouncer/train.py or services/detective/train.py; promotion is
model_registry.py's job once a candidate is judged better.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from libs.constants import DriftAction, DriftMetric
from libs.observability import ACTIVE_GAUGE, observe
from libs.schemas import DriftReport, FeatureDrift, LiveAccuracy
from services.drift_watcher.metrics import PSI_DRIFT_THRESHOLD, per_feature_drift


def should_promote(current_metrics: dict | None, candidate_metrics: dict, key: str = "f1") -> bool:
    """spec.md: "promotes a new model only if it beats the current one."
    No current model at all is treated as an automatic win for the
    candidate (bootstrapping the very first version).
    """
    if current_metrics is None:
        return True
    return candidate_metrics.get(key, 0.0) > current_metrics.get(key, 0.0)


class DriftWatcher:
    def __init__(self, baseline_features: dict[str, np.ndarray]) -> None:
        self._baseline = baseline_features
        self._corrections_since_last_cycle = 0
        self._decoy_sessions_since_last_cycle = 0

    def record_correction(self) -> None:
        self._corrections_since_last_cycle += 1

    def record_decoy_session(self) -> None:
        self._decoy_sessions_since_last_cycle += 1

    def run_cycle(
        self, live_features: dict[str, np.ndarray], live_precision: float, live_recall: float,
        baseline_precision: float, window_start: datetime, window_end: datetime,
    ) -> DriftReport:
        with observe("drift_watcher", "run_cycle"):
            scores = per_feature_drift(self._baseline, live_features, metric="PSI")
            threshold_breached = any(score > PSI_DRIFT_THRESHOLD for score in scores.values())

            report = DriftReport(
                window=f"{window_start.isoformat()}/{window_end.isoformat()}",
                feature_drift=FeatureDrift(
                    metric=DriftMetric.PSI, per_feature_scores=scores,
                    threshold_breached=threshold_breached,
                ),
                live_accuracy=LiveAccuracy(
                    precision=live_precision, recall=live_recall,
                    vs_baseline_delta=live_precision - baseline_precision,
                ),
                corrections_since_last_cycle=self._corrections_since_last_cycle,
                decoy_sessions_since_last_cycle=self._decoy_sessions_since_last_cycle,
                action=DriftAction.RETRAIN_TRIGGERED if threshold_breached else DriftAction.NONE,
            )

            self._corrections_since_last_cycle = 0
            self._decoy_sessions_since_last_cycle = 0
            ACTIVE_GAUGE.labels(component="drift_watcher", kind="last_psi_max").set(
                max(scores.values()) if scores else 0.0
            )
            return report
