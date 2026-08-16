import numpy as np
import pytest

from libs.constants import AttributionMethod, Label, Tier
from services.bouncer.model import BouncerModel


def _toy_dataset(n=400, seed=0):
    # 6 features matching FEATURE_NAMES order; flood rows get an obvious
    # high-rate, low-entropy signature, benign rows the opposite — enough
    # separation for XGBoost to learn a near-perfect boundary, which is
    # the point: this test is about verdict *construction*, not accuracy.
    rng = np.random.default_rng(seed)
    half = n // 2
    flood = np.column_stack([
        rng.uniform(80, 120, half), rng.uniform(8000, 12000, half),
        rng.uniform(0, 0.2, half), rng.uniform(1, 2, half),
        rng.uniform(1, 5, half), rng.uniform(0.9, 1.0, half),
    ])
    benign = np.column_stack([
        rng.uniform(0, 5, half), rng.uniform(50, 500, half),
        rng.uniform(1, 3, half), rng.uniform(3, 10, half),
        rng.uniform(20, 200, half), rng.uniform(0.0, 0.3, half),
    ])
    X = np.vstack([flood, benign])
    y = np.array([1] * half + [0] * half)
    return X, y


@pytest.fixture(scope="module")
def trained_model():
    X, y = _toy_dataset()
    return BouncerModel().fit(X, y)


def test_fitted_model_separates_flood_from_benign(trained_model):
    X, y = _toy_dataset(n=100, seed=1)
    proba = trained_model.predict_proba(X)
    pred = (proba >= 0.5).astype(int)
    accuracy = (pred == y).mean()
    assert accuracy > 0.9


def test_predict_verdict_on_clear_flood_signature(trained_model):
    features = {
        "event_rate": 100.0, "byte_rate": 10000.0, "dest_port_entropy": 0.0,
        "unique_dest_count": 1.0, "avg_duration_ms": 2.0, "same_dest_ratio": 1.0,
    }
    verdict = trained_model.predict_verdict(features, window_id="w-1")
    assert verdict.tier == Tier.BOUNCER
    assert verdict.label in (Label.FLOOD, Label.UNCERTAIN)  # depends on calibrated confidence
    assert verdict.evidence.attribution_method == AttributionMethod.RATE_THRESHOLD
    assert verdict.evidence.node_ids == []  # bouncer never carries graph evidence


def test_predict_verdict_on_clear_benign_signature(trained_model):
    features = {
        "event_rate": 2.0, "byte_rate": 200.0, "dest_port_entropy": 2.0,
        "unique_dest_count": 5.0, "avg_duration_ms": 100.0, "same_dest_ratio": 0.1,
    }
    verdict = trained_model.predict_verdict(features, window_id="w-2")
    assert verdict.label == Label.BENIGN


def test_save_and_load_round_trip_produces_identical_predictions(trained_model, tmp_path):
    X, _ = _toy_dataset(n=20, seed=2)
    before = trained_model.predict_proba(X)

    trained_model.save(tmp_path)
    reloaded = BouncerModel.load(tmp_path)
    after = reloaded.predict_proba(X)

    assert np.allclose(before, after, atol=1e-6)
