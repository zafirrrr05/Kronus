import numpy as np

from services.drift_watcher.metrics import (
    KS_PVALUE_THRESHOLD,
    PSI_DRIFT_THRESHOLD,
    ks_test,
    per_feature_drift,
    population_stability_index,
)


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, size=5000)
    same = rng.normal(0, 1, size=5000)
    psi = population_stability_index(baseline, same)
    assert psi < 0.05


def test_psi_exceeds_threshold_for_a_meaningfully_shifted_distribution():
    rng = np.random.default_rng(1)
    baseline = rng.normal(0, 1, size=5000)
    shifted = rng.normal(2.5, 1, size=5000)  # a large, meaningful mean shift
    psi = population_stability_index(baseline, shifted)
    assert psi > PSI_DRIFT_THRESHOLD


def test_psi_handles_empty_input_gracefully():
    assert population_stability_index(np.array([]), np.array([1, 2, 3])) == 0.0
    assert population_stability_index(np.array([1, 2, 3]), np.array([])) == 0.0


def test_psi_handles_constant_baseline_without_crashing():
    baseline = np.full(100, 5.0)
    actual = np.full(100, 5.0)
    assert population_stability_index(baseline, actual) == 0.0


def test_ks_test_fails_to_reject_for_identical_distributions():
    # KS is a hypothesis test with a real ~5% false-positive rate even for
    # truly identical distributions (p-values from a true null are
    # uniform on [0,1]) — seed=2 happened to land at p=0.045, just under
    # the boundary, making that seed a flaky choice. seed=0 gives p=0.90,
    # comfortably clear, and demonstrates the same property reliably.
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, size=2000)
    same = rng.normal(0, 1, size=2000)
    _, p_value = ks_test(baseline, same)
    assert p_value > KS_PVALUE_THRESHOLD


def test_ks_test_rejects_for_a_clearly_different_distribution():
    rng = np.random.default_rng(3)
    baseline = rng.normal(0, 1, size=2000)
    shifted = rng.normal(3, 1, size=2000)
    statistic, p_value = ks_test(baseline, shifted)
    assert p_value < KS_PVALUE_THRESHOLD
    assert statistic > 0.5


def test_per_feature_drift_reports_one_score_per_feature():
    rng = np.random.default_rng(4)
    expected = {"a": rng.normal(0, 1, 1000), "b": rng.normal(0, 1, 1000)}
    actual = {"a": rng.normal(0, 1, 1000), "b": rng.normal(5, 1, 1000)}  # b drifted, a didn't
    scores = per_feature_drift(expected, actual, metric="PSI")
    assert scores["a"] < PSI_DRIFT_THRESHOLD
    assert scores["b"] > PSI_DRIFT_THRESHOLD
