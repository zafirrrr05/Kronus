"""Tests for ML Experiments (Repo Data & API Data).

Covers:
- Dataset loading
- Training verification
- Weight saving and loading
- Experiment separation (independence)
- Metric generation
- API data collection
- API inference
- API correctness
- Train/test separation and leakage prevention
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from libs.constants import Label, Protocol, Sensor
from libs.schemas import GraphEdge, GraphNode, GraphSnapshot, TelemetryEvent
from services.bouncer.model import BouncerModel
from services.detective.model import DetectiveModel
from services.graph_builder.builder import WindowedGraphBuilder

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_WEIGHTS_BOUNCER = REPO_ROOT / "models" / "experiments" / "repo_data" / "bouncer"
REPO_WEIGHTS_DETECTIVE = REPO_ROOT / "models" / "experiments" / "repo_data" / "detective"
API_WEIGHTS_BOUNCER = REPO_ROOT / "models" / "experiments" / "api_data" / "bouncer"
API_WEIGHTS_DETECTIVE = REPO_ROOT / "models" / "experiments" / "api_data" / "detective"

REPO_METRICS_PATH = REPO_ROOT / "results" / "experiments" / "repo_data_metrics.json"
API_METRICS_PATH = REPO_ROOT / "results" / "experiments" / "api_data_metrics.json"

API_TRAIN_JSONL = REPO_ROOT / "data" / "api" / "train_events.jsonl"
API_TEST_JSONL = REPO_ROOT / "data" / "api" / "test_events.jsonl"


def test_dataset_loading_api():
    """Verify API-collected JSONL dataset exists and loads properly."""
    assert API_TRAIN_JSONL.exists(), "API train dataset missing"
    assert API_TEST_JSONL.exists(), "API test dataset missing"

    train_count = 0
    with open(API_TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            assert "event" in record
            assert "ground_truth_label" in record
            assert "event_id" in record["event"]
            train_count += 1
    assert train_count > 0, "API train dataset is empty"


def test_train_test_separation_and_leakage():
    """Ensure zero overlap between train and test sets in API experiment."""
    train_ids = set()
    with open(API_TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            train_ids.add(rec["event"]["event_id"])

    test_ids = set()
    with open(API_TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            test_ids.add(rec["event"]["event_id"])

    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Data leakage detected! {len(overlap)} samples overlap between train and test."


def test_weight_saving_and_loading_repo():
    """Verify repo-trained weights load properly and perform inference."""
    assert (REPO_WEIGHTS_BOUNCER / "bouncer.json").exists()
    bouncer = BouncerModel.load(REPO_WEIGHTS_BOUNCER)
    feats = {
        "event_rate": 50.0,
        "byte_rate": 10000.0,
        "dest_port_entropy": 0.5,
        "unique_dest_count": 2,
        "avg_duration_ms": 15.0,
        "same_dest_ratio": 0.9,
    }
    v = bouncer.predict_verdict(feats, window_id="test-repo-b")
    assert v is not None
    assert 0.0 <= v.confidence <= 1.0

    assert (REPO_WEIGHTS_DETECTIVE / "detective.npz").exists()
    detective = DetectiveModel.load(REPO_WEIGHTS_DETECTIVE)
    now = datetime.now(timezone.utc)
    snap = GraphSnapshot(
        window_id="test-repo-d",
        window_start=now,
        window_end=now,
        nodes=[
            GraphNode(node_id="10.0.0.1", degree_in=0.0, degree_out=1.0, bytes_total=100.0, unique_ports_contacted=1),
            GraphNode(node_id="10.0.0.2", degree_in=1.0, degree_out=0.0, bytes_total=100.0, unique_ports_contacted=0),
        ],
        edges=[
            GraphEdge(src="10.0.0.1", dst="10.0.0.2", bytes=100.0, flow_count=1, port_entropy=0.0, duration_mean_ms=10.0),
        ],
    )
    v_det = detective.predict_verdict(snap, window_id="test-repo-d")
    assert v_det is not None
    assert 0.0 <= v_det.confidence <= 1.0


def test_weight_saving_and_loading_api():
    """Verify API-trained weights load properly and perform inference."""
    assert (API_WEIGHTS_BOUNCER / "bouncer.json").exists()
    bouncer = BouncerModel.load(API_WEIGHTS_BOUNCER)
    feats = {
        "event_rate": 20.0,
        "byte_rate": 5000.0,
        "dest_port_entropy": 0.2,
        "unique_dest_count": 1,
        "avg_duration_ms": 10.0,
        "same_dest_ratio": 1.0,
    }
    v = bouncer.predict_verdict(feats, window_id="test-api-b")
    assert v is not None
    assert 0.0 <= v.confidence <= 1.0

    assert (API_WEIGHTS_DETECTIVE / "detective.npz").exists()
    detective = DetectiveModel.load(API_WEIGHTS_DETECTIVE)
    now = datetime.now(timezone.utc)
    snap = GraphSnapshot(
        window_id="test-api-d",
        window_start=now,
        window_end=now,
        nodes=[
            GraphNode(node_id="10.0.1.5", degree_in=0.0, degree_out=1.0, bytes_total=100.0, unique_ports_contacted=1),
            GraphNode(node_id="10.0.2.10", degree_in=1.0, degree_out=0.0, bytes_total=100.0, unique_ports_contacted=0),
        ],
        edges=[
            GraphEdge(src="10.0.1.5", dst="10.0.2.10", bytes=100.0, flow_count=1, port_entropy=0.0, duration_mean_ms=10.0),
        ],
    )
    v_det = detective.predict_verdict(snap, window_id="test-api-d")
    assert v_det is not None
    assert 0.0 <= v_det.confidence <= 1.0


def test_experiment_independence():
    """Ensure repo-trained weights and API-trained weights are completely separate."""
    # Ensure separate directories
    assert REPO_WEIGHTS_BOUNCER != API_WEIGHTS_BOUNCER
    assert REPO_WEIGHTS_DETECTIVE != API_WEIGHTS_DETECTIVE

    # Ensure weights are not identical file copies (independent training)
    repo_b_bytes = (REPO_WEIGHTS_BOUNCER / "bouncer.json").read_bytes()
    api_b_bytes = (API_WEIGHTS_BOUNCER / "bouncer.json").read_bytes()
    assert repo_b_bytes != api_b_bytes, "Repo and API Bouncer weights must not be identical!"

    repo_d_bytes = (REPO_WEIGHTS_DETECTIVE / "detective.npz").read_bytes()
    api_d_bytes = (API_WEIGHTS_DETECTIVE / "detective.npz").read_bytes()
    assert repo_d_bytes != api_d_bytes, "Repo and API Detective weights must not be identical!"


def test_metric_generation():
    """Verify both metric files exist and contain complete, non-empty metric sections."""
    assert REPO_METRICS_PATH.exists()
    with open(REPO_METRICS_PATH, "r", encoding="utf-8") as f:
        repo_m = json.load(f)
    assert repo_m["experiment"] == "repo_data"
    assert "bouncer" in repo_m and "f1" in repo_m["bouncer"]
    assert "detective" in repo_m and "f1" in repo_m["detective"]
    assert "historical_comparison" in repo_m
    assert repo_m["weight_verification"]["bouncer_loadable"] is True
    assert repo_m["weight_verification"]["detective_loadable"] is True

    assert API_METRICS_PATH.exists()
    with open(API_METRICS_PATH, "r", encoding="utf-8") as f:
        api_m = json.load(f)
    assert api_m["experiment"] == "api_data"
    assert "api_performance" in api_m
    assert api_m["api_performance"]["total_requests"] > 0
    assert "live_api_correctness" in api_m
    assert "f1" in api_m["live_api_correctness"]
    assert "offline_api_models" in api_m
    assert "bouncer" in api_m["offline_api_models"]
    assert "detective" in api_m["offline_api_models"]
    assert api_m["weight_verification"]["api_bouncer_loadable"] is True
    assert api_m["weight_verification"]["api_detective_loadable"] is True


@pytest.mark.asyncio
async def test_api_inference_and_correctness():
    """Test actual API inference and verdict generation via FastAPI client."""
    import api.main as api_main
    from libs.event_bus import InMemoryEventBus
    from pipeline.kronus_system import KronusSystem
    from services.drift_watcher.retrainer import DriftWatcher
    from services.llm_explainer.explainer import LLMExplainer
    from services.logbook.sqlite_store import SQLiteLogbookStore
    from services.response_engine.engine import ResponseEngine
    from services.response_engine.opa_client import PolicyClient
    from tests.integration.conftest import _fast_bouncer, _fast_detective

    bus = InMemoryEventBus()
    system = KronusSystem(
        bus=bus,
        bouncer=_fast_bouncer(),
        detective=_fast_detective(),
        response_engine=ResponseEngine(PolicyClient()),
        logbook=SQLiteLogbookStore(":memory:"),
        explainer=LLMExplainer(api_key=None),
        drift_watcher=DriftWatcher(baseline_features={"f": np.random.default_rng(0).normal(size=10)}),
    )
    await system.start()
    api_main.bind_system(system)

    try:
        client = TestClient(api_main.app)
        ev_payload = {
            "source_ip": "192.0.2.45",
            "dest_ip": "10.0.0.1",
            "dest_port": 80,
            "protocol": "tcp",
            "bytes": 500,
            "duration_ms": 10,
            "sensor": "zeek",
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        resp = client.post("/events", json=ev_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ingested"
        assert "bouncer_verdict" in data
        assert "detective_verdict" in data
        assert "decisions" in data
    finally:
        await system.stop()
        api_main._system = None
