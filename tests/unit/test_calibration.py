import numpy as np

from libs.calibration import TemperatureScaler, _neg_log_likelihood, _sigmoid


def test_temperature_scaling_softens_an_overconfident_binary_model():
    rng = np.random.default_rng(0)
    n = 2000
    true_prob = rng.uniform(0.05, 0.95, size=n)
    labels = rng.binomial(1, true_prob)
    # an overconfident model: true logit stretched by 4x
    true_logit = np.log(true_prob / (1 - true_prob))
    overconfident_logits = true_logit * 4.0

    scaler = TemperatureScaler().fit(overconfident_logits, labels)
    assert scaler.temperature > 1.5  # should learn to soften, not sharpen

    calibrated = scaler.calibrate(overconfident_logits)
    raw = _sigmoid(overconfident_logits)
    assert _neg_log_likelihood(calibrated, labels) < _neg_log_likelihood(raw, labels)


def test_temperature_scaling_is_near_identity_for_an_already_calibrated_model():
    rng = np.random.default_rng(1)
    n = 2000
    true_prob = rng.uniform(0.05, 0.95, size=n)
    labels = rng.binomial(1, true_prob)
    true_logit = np.log(true_prob / (1 - true_prob))

    scaler = TemperatureScaler().fit(true_logit, labels)
    assert 0.7 < scaler.temperature < 1.4  # roughly 1.0, some sampling noise expected


def test_multiclass_calibration_produces_a_valid_probability_simplex():
    rng = np.random.default_rng(2)
    n, k = 500, 4
    logits = rng.normal(size=(n, k)) * 3.0
    labels = rng.integers(0, k, size=n)

    scaler = TemperatureScaler().fit(logits, labels)
    probs = scaler.calibrate(logits)
    assert probs.shape == (n, k)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert (probs >= 0).all()
