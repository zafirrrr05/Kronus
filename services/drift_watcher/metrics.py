"""features.txt component 9: "PSI / KS-test... continuously measures live
accuracy using known attacks the Twin keeps injecting." Two independent
drift signals, matching spec.md §3.8's FeatureDrift.metric: PSI|KS.

PSI is hand-implemented (a short, standard formula — not worth a
dependency). KS-test uses scipy.stats.ks_2samp directly (scipy is already
a dependency for calibration — libs/calibration.py — so this is the
"already installed" rung of the ladder, not a new one).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# spec.md doesn't fix a numeric drift threshold; 0.2 is the standard,
# widely-cited PSI convention (< 0.1 no shift, 0.1-0.2 moderate, > 0.2
# significant) and 0.05 is a conventional KS-test significance level.
PSI_DRIFT_THRESHOLD = 0.2
KS_PVALUE_THRESHOLD = 0.05


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Standard PSI: bin the expected (baseline/training) distribution
    into `bins` quantile buckets, then compare the actual (live)
    distribution's share in each bucket against the expected share.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    quantile_edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantile_edges[0], quantile_edges[-1] = -np.inf, np.inf
    quantile_edges = np.unique(quantile_edges)
    if len(quantile_edges) < 2:
        return 0.0  # expected has no spread (e.g. a constant feature) — nothing to compare

    expected_counts, _ = np.histogram(expected, bins=quantile_edges)
    actual_counts, _ = np.histogram(actual, bins=quantile_edges)

    expected_pct = np.maximum(expected_counts / len(expected), 1e-6)
    actual_pct = np.maximum(actual_counts / len(actual), 1e-6)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def ks_test(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Returns (statistic, p_value). A small p-value rejects the null
    hypothesis that both samples are drawn from the same distribution —
    i.e. evidence of drift.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0, 1.0
    result = stats.ks_2samp(expected, actual)
    return float(result.statistic), float(result.pvalue)


def per_feature_drift(
    expected: dict[str, np.ndarray], actual: dict[str, np.ndarray], metric: str = "PSI",
) -> dict[str, float]:
    """metric: "PSI" or "KS" — matches spec.md §3.8's FeatureDrift.metric.
    For KS, the reported score is the KS statistic (not the p-value) so
    higher-is-more-drifted holds for both metrics uniformly.
    """
    scores = {}
    for feature_name, expected_values in expected.items():
        actual_values = actual.get(feature_name, np.array([]))
        if metric == "PSI":
            scores[feature_name] = population_stability_index(expected_values, actual_values)
        else:
            statistic, _ = ks_test(expected_values, actual_values)
            scores[feature_name] = statistic
    return scores
