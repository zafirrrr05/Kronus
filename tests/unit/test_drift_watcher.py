from datetime import datetime, timezone

import numpy as np
import pytest

from libs.constants import DriftAction
from services.bouncer.model import BouncerModel
from services.drift_watcher.model_registry import ModelRegistry
from services.drift_watcher.retrainer import DriftWatcher, should_promote

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


# --- should_promote ----------------------------------------------------------

def test_no_current_model_always_promotes():
    assert should_promote(None, {"f1": 0.1}) is True


def test_better_candidate_is_promoted():
    assert should_promote({"f1": 0.7}, {"f1": 0.8}) is True


def test_worse_candidate_is_not_promoted():
    assert should_promote({"f1": 0.8}, {"f1": 0.7}) is False


def test_equal_candidate_is_not_promoted():
    # "beats" — strictly better, not tied
    assert should_promote({"f1": 0.8}, {"f1": 0.8}) is False


# --- ModelRegistry, with a real trained Bouncer artifact ---------------------

@pytest.fixture
def trained_bouncer_dir(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 6))
    y = rng.integers(0, 2, size=100)
    model = BouncerModel().fit(X, y)
    artifact_dir = tmp_path / "trained_bouncer"
    model.save(artifact_dir)
    return artifact_dir


def test_register_copies_a_real_artifact_and_records_metrics(trained_bouncer_dir, tmp_path):
    registry = ModelRegistry(root=tmp_path / "registry")
    version = registry.register("bouncer", trained_bouncer_dir, metrics={"f1": 0.75})
    assert (tmp_path / "registry" / "bouncer" / version.version_id / "bouncer.json").exists()
    assert version.metrics == {"f1": 0.75}


def test_no_current_version_before_any_promotion(trained_bouncer_dir, tmp_path):
    registry = ModelRegistry(root=tmp_path / "registry")
    registry.register("bouncer", trained_bouncer_dir, metrics={"f1": 0.75})
    assert registry.get_current("bouncer") is None


def test_promote_makes_a_version_current(trained_bouncer_dir, tmp_path):
    registry = ModelRegistry(root=tmp_path / "registry")
    version = registry.register("bouncer", trained_bouncer_dir, metrics={"f1": 0.75})
    registry.promote("bouncer", version.version_id)
    current = registry.get_current("bouncer")
    assert current.version_id == version.version_id


def test_promoted_artifact_is_loadable_and_predicts(trained_bouncer_dir, tmp_path):
    registry = ModelRegistry(root=tmp_path / "registry")
    version = registry.register("bouncer", trained_bouncer_dir, metrics={"f1": 0.75})
    registry.promote("bouncer", version.version_id)

    artifact_dir = registry.current_artifact_dir("bouncer")
    reloaded = BouncerModel.load(artifact_dir)
    rng = np.random.default_rng(1)
    proba = reloaded.predict_proba(rng.normal(size=(5, 6)))
    assert proba.shape == (5,)


def test_promoting_a_nonexistent_version_raises(tmp_path):
    registry = ModelRegistry(root=tmp_path / "registry")
    with pytest.raises(FileNotFoundError):
        registry.promote("bouncer", "does-not-exist")


# --- DriftWatcher --------------------------------------------------------

def test_stable_features_produce_no_retrain():
    rng = np.random.default_rng(5)
    baseline = {"f1": rng.normal(0, 1, 1000)}
    watcher = DriftWatcher(baseline_features=baseline)
    live = {"f1": rng.normal(0, 1, 1000)}
    report = watcher.run_cycle(live, live_precision=0.9, live_recall=0.85,
                                baseline_precision=0.9, window_start=T0, window_end=T0)
    assert report.action == DriftAction.NONE
    assert report.feature_drift.threshold_breached is False


def test_shifted_features_trigger_retrain():
    rng = np.random.default_rng(6)
    baseline = {"f1": rng.normal(0, 1, 1000)}
    watcher = DriftWatcher(baseline_features=baseline)
    live = {"f1": rng.normal(3, 1, 1000)}  # meaningful shift
    report = watcher.run_cycle(live, live_precision=0.6, live_recall=0.5,
                                baseline_precision=0.9, window_start=T0, window_end=T0)
    assert report.action == DriftAction.RETRAIN_TRIGGERED
    assert report.live_accuracy.vs_baseline_delta == pytest.approx(-0.3)


def test_correction_and_decoy_counts_reset_after_each_cycle():
    rng = np.random.default_rng(7)
    baseline = {"f1": rng.normal(0, 1, 500)}
    watcher = DriftWatcher(baseline_features=baseline)
    watcher.record_correction()
    watcher.record_correction()
    watcher.record_decoy_session()

    live = {"f1": rng.normal(0, 1, 500)}
    report = watcher.run_cycle(live, 0.9, 0.85, 0.9, T0, T0)
    assert report.corrections_since_last_cycle == 2
    assert report.decoy_sessions_since_last_cycle == 1

    second_report = watcher.run_cycle(live, 0.9, 0.85, 0.9, T0, T0)
    assert second_report.corrections_since_last_cycle == 0
    assert second_report.decoy_sessions_since_last_cycle == 0
