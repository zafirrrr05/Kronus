"""Integration-test fixtures. Models here are trained fast and small on
purpose — deterministic toy data, not the full NSL-KDD sweep
(services/*/train.py owns that; see the demo, which uses the real
models/registry/ artifacts). What's under test here is the *wiring*
between components, not model accuracy — that's already covered by each
component's own unit tests against real data.
"""

from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio

from libs.constants import Label
from libs.event_bus import InMemoryEventBus
from pipeline.kronus_system import KronusSystem
from services.bouncer.model import BouncerModel
from services.detective.model import DetectiveModel
from services.drift_watcher.retrainer import DriftWatcher
from services.llm_explainer.explainer import LLMExplainer
from services.logbook.sqlite_store import SQLiteLogbookStore
from services.response_engine.engine import ResponseEngine
from services.response_engine.opa_client import PolicyClient
from tests.unit.test_detective_model import _benign_snapshot, _scan_snapshot


def _fast_bouncer() -> BouncerModel:
    rng = np.random.default_rng(0)
    half = 60
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
    return BouncerModel().fit(X, y)


def _fast_detective() -> DetectiveModel:
    model = DetectiveModel(rng=np.random.default_rng(11))
    batch = [(_scan_snapshot(), Label.PORT_SCAN), (_benign_snapshot(), Label.BENIGN)]
    for _ in range(150):
        model.train_batch(batch, learning_rate=0.05)
    return model


@pytest.fixture(scope="session")
def _trained_bouncer() -> BouncerModel:
    return _fast_bouncer()


@pytest.fixture(scope="session")
def _trained_detective() -> DetectiveModel:
    return _fast_detective()


@pytest_asyncio.fixture
async def kronus_system(_trained_bouncer, _trained_detective):
    # Model training (expensive, ~30s for the Detective's 150 batches) is
    # session-scoped since prediction is stateless given fixed params —
    # nothing about ingest/decide mutates the models. Everything else
    # (bus, logbook, response engine's correlator/breaker/blocks, drift
    # watcher's counters) is fresh per test, since those genuinely
    # accumulate state a shared fixture would leak between tests.
    bus = InMemoryEventBus()
    logbook = SQLiteLogbookStore(":memory:")
    system = KronusSystem(
        bus=bus,
        bouncer=_trained_bouncer,
        detective=_trained_detective,
        response_engine=ResponseEngine(PolicyClient()),
        logbook=logbook,
        explainer=LLMExplainer(api_key=None),  # offline — see explainer.py's docstring
        drift_watcher=DriftWatcher(baseline_features={"f": np.random.default_rng(0).normal(size=100)}),
    )
    await system.start()
    yield system
    await system.stop()
    logbook.close()
