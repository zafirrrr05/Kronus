"""Tests for KRONUS ML Experiments.

EXPERIMENT A — REPO DATA (NSL-KDD):
  - Dataset loading
  - Training execution and weight saving
  - Weight loading and inference
  - Historical metric comparison

EXPERIMENT B — CIC-IDS2017:
  - Dataset loader unit tests (offline, no real data required)
  - CIC label mapping correctness
  - CIC feature mapping correctness
  - Graceful not-run handling when data is absent
  - Weight loading and inference (when experiment was run)
  - Experiment independence (CIC weights != repo weights when both exist)

API INTEGRATION:
  - KRONUS API endpoint smoke test (real POST /events)
  - API returns correct response schema
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Repo-data (NSL-KDD) experiment artifacts
REPO_WEIGHTS_BOUNCER   = REPO_ROOT / "models" / "experiments" / "repo_data" / "bouncer"
REPO_WEIGHTS_DETECTIVE = REPO_ROOT / "models" / "experiments" / "repo_data" / "detective"
REPO_METRICS_PATH      = REPO_ROOT / "results" / "experiments" / "repo_data_metrics.json"

# CIC-IDS2017 experiment artifacts (may not exist if data was not downloaded)
CIC_WEIGHTS_BOUNCER    = REPO_ROOT / "models" / "experiments" / "cic_ids2017" / "bouncer"
CIC_WEIGHTS_DETECTIVE  = REPO_ROOT / "models" / "experiments" / "cic_ids2017" / "detective"
CIC_METRICS_PATH       = REPO_ROOT / "results" / "experiments" / "cic_ids2017_metrics.json"
CIC_DATA_DIR           = REPO_ROOT / "data" / "external" / "cicids2017"

# Removed fake artifact paths (must NOT exist)
FAKE_API_TRAIN         = REPO_ROOT / "data" / "api" / "train_events.jsonl"
FAKE_API_TEST          = REPO_ROOT / "data" / "api" / "test_events.jsonl"
FAKE_API_METRICS       = REPO_ROOT / "results" / "experiments" / "api_data_metrics.json"
FAKE_API_WEIGHTS       = REPO_ROOT / "models" / "experiments" / "api_data"


# ─────────────────────────────────────────────────────────────────────────────
# REPO DATA (NSL-KDD) EXPERIMENT TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_repo_weights_bouncer_loadable():
    """Repo-trained Bouncer weights load from disk and produce a valid verdict."""
    assert (REPO_WEIGHTS_BOUNCER / "bouncer.json").exists(), \
        "Run scripts/run_repo_experiment.py first"
    bouncer = BouncerModel.load(REPO_WEIGHTS_BOUNCER)
    feats = {
        "event_rate": 50.0, "byte_rate": 10000.0, "dest_port_entropy": 0.5,
        "unique_dest_count": 2, "avg_duration_ms": 15.0, "same_dest_ratio": 0.9,
    }
    v = bouncer.predict_verdict(feats, window_id="test-repo-b")
    assert v is not None
    assert 0.0 <= v.confidence <= 1.0


def test_repo_weights_detective_loadable():
    """Repo-trained Detective weights load from disk and produce a valid verdict."""
    assert (REPO_WEIGHTS_DETECTIVE / "detective.npz").exists(), \
        "Run scripts/run_repo_experiment.py first"
    detective = DetectiveModel.load(REPO_WEIGHTS_DETECTIVE)
    now = datetime.now(timezone.utc)
    snap = GraphSnapshot(
        window_id="test-repo-d",
        window_start=now, window_end=now,
        nodes=[
            GraphNode(node_id="10.0.0.1", degree_in=0.0, degree_out=1.0,
                      bytes_total=100.0, unique_ports_contacted=1),
            GraphNode(node_id="10.0.0.2", degree_in=1.0, degree_out=0.0,
                      bytes_total=100.0, unique_ports_contacted=0),
        ],
        edges=[
            GraphEdge(src="10.0.0.1", dst="10.0.0.2",
                      bytes=100.0, flow_count=1, port_entropy=0.0, duration_mean_ms=10.0),
        ],
    )
    v = detective.predict_verdict(snap, window_id="test-repo-d")
    assert v is not None
    assert 0.0 <= v.confidence <= 1.0


def test_repo_metrics_complete():
    """Repo-data metric file exists and contains required fields."""
    assert REPO_METRICS_PATH.exists(), "Run scripts/run_repo_experiment.py first"
    with open(REPO_METRICS_PATH) as f:
        m = json.load(f)
    assert m["experiment"] == "repo_data"
    assert "bouncer" in m and "f1" in m["bouncer"]
    assert "detective" in m and "f1" in m["detective"]
    assert "historical_comparison" in m
    assert m["weight_verification"]["bouncer_loadable"] is True
    assert m["weight_verification"]["detective_loadable"] is True


def test_repo_metrics_match_historical():
    """Repo-data metrics are within rounding tolerance of historical claims."""
    with open(REPO_METRICS_PATH) as f:
        m = json.load(f)
    hist_b = m["historical_comparison"]["bouncer"]
    meas_b = hist_b["measured"]
    hist_vals = hist_b["historical"]
    # Allow ±0.001 for floating-point/seed variation
    assert abs(meas_b["accuracy"]  - hist_vals["accuracy"])  <= 0.001
    assert abs(meas_b["precision"] - hist_vals["precision"]) <= 0.001
    assert abs(meas_b["recall"]    - hist_vals["recall"])    <= 0.001
    assert abs(meas_b["f1"]        - hist_vals["f1"])        <= 0.001


def test_repo_nsl_kdd_dataset_present():
    """NSL-KDD training dataset is present."""
    assert (REPO_ROOT / "data" / "real" / "KDDTrain+.txt").exists()
    assert (REPO_ROOT / "data" / "real" / "KDDTest+.txt").exists()


# ─────────────────────────────────────────────────────────────────────────────
# CIC-IDS2017 EXPERIMENT TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_cic_metrics_file_exists():
    """CIC-IDS2017 metrics file exists (either NOT_RUN or COMPLETE)."""
    assert CIC_METRICS_PATH.exists(), (
        "Run scripts/run_cic_experiment.py first "
        "(it creates the file even when CIC data is absent)"
    )


def test_cic_metrics_not_fake():
    """CIC metrics file honestly reports NOT_RUN when data is absent, not fabricated."""
    with open(CIC_METRICS_PATH) as f:
        m = json.load(f)
    assert m["experiment"] == "cic_ids2017"
    assert m["status"] in ("NOT_RUN", "COMPLETE"), \
        f"Unexpected status: {m['status']}"
    if m["status"] == "NOT_RUN":
        assert "reason" in m
        assert "url" in m
        # Must NOT contain any fabricated metrics
        assert "bouncer" not in m or "f1" not in m.get("bouncer", {})


def test_cic_label_mapping():
    """CIC label mapping covers all known CIC-IDS2017 attack categories."""
    from twin.cicids2017 import CIC_ATTACK_CATEGORY
    # Must have BENIGN -> normal mapping
    assert "benign" in CIC_ATTACK_CATEGORY
    assert CIC_ATTACK_CATEGORY["benign"] == "normal"
    # Must have DoS mappings
    for dos_label in ("dos hulk", "dos goldeneye", "dos slowloris",
                      "dos slowhttptest", "ddos"):
        assert dos_label in CIC_ATTACK_CATEGORY, f"Missing DoS label: {dos_label}"
        assert CIC_ATTACK_CATEGORY[dos_label] == "dos"
    # Must have PortScan mapping
    assert "portscan" in CIC_ATTACK_CATEGORY
    assert CIC_ATTACK_CATEGORY["portscan"] == "probe"


def test_cic_loader_handles_missing_data_gracefully():
    """CIC loader raises FileNotFoundError or returns empty list for missing path."""
    from twin.cicids2017 import load_cicids2017
    fake_path = REPO_ROOT / "data" / "external" / "cicids2017_nonexistent_test_path"
    try:
        result = load_cicids2017(fake_path)
        assert result == [], f"Expected empty list for missing path, got {len(result)} rows"
    except (FileNotFoundError, OSError):
        pass  # Also acceptable


def test_cic_weights_loadable_if_present():
    """If CIC experiment was run, weights load and produce valid verdicts."""
    if not (CIC_WEIGHTS_BOUNCER / "bouncer.json").exists():
        pytest.skip("CIC experiment not run — CIC-IDS2017 data not present")
    bouncer = BouncerModel.load(CIC_WEIGHTS_BOUNCER)
    feats = {n: 10.0 for n in [
        "event_rate", "byte_rate", "dest_port_entropy",
        "unique_dest_count", "avg_duration_ms", "same_dest_ratio",
    ]}
    v = bouncer.predict_verdict(feats, window_id="test-cic-b")
    assert v is not None
    assert 0.0 <= v.confidence <= 1.0


def test_cic_detective_loadable_if_present():
    """If CIC experiment was run, Detective weights load and infer."""
    if not (CIC_WEIGHTS_DETECTIVE / "detective.npz").exists():
        pytest.skip("CIC experiment not run — CIC-IDS2017 data not present")
    detective = DetectiveModel.load(CIC_WEIGHTS_DETECTIVE)
    now = datetime.now(timezone.utc)
    snap = GraphSnapshot(
        window_id="test-cic-d",
        window_start=now, window_end=now,
        nodes=[
            GraphNode(node_id="10.0.1.1", degree_in=0.0, degree_out=2.0,
                      bytes_total=200.0, unique_ports_contacted=2),
            GraphNode(node_id="10.0.1.2", degree_in=2.0, degree_out=0.0,
                      bytes_total=200.0, unique_ports_contacted=0),
        ],
        edges=[
            GraphEdge(src="10.0.1.1", dst="10.0.1.2",
                      bytes=200.0, flow_count=2, port_entropy=0.5, duration_mean_ms=20.0),
        ],
    )
    v = detective.predict_verdict(snap, window_id="test-cic-d")
    assert v is not None
    assert 0.0 <= v.confidence <= 1.0


def test_experiment_independence_when_both_present():
    """When both experiments were run, CIC weights must differ from repo weights."""
    if not (CIC_WEIGHTS_BOUNCER / "bouncer.json").exists():
        pytest.skip("CIC experiment not run — skipping independence check")
    repo_b = (REPO_WEIGHTS_BOUNCER / "bouncer.json").read_bytes()
    cic_b  = (CIC_WEIGHTS_BOUNCER  / "bouncer.json").read_bytes()
    assert repo_b != cic_b, "Repo and CIC Bouncer weights must not be identical!"

    repo_d = (REPO_WEIGHTS_DETECTIVE / "detective.npz").read_bytes()
    cic_d  = (CIC_WEIGHTS_DETECTIVE  / "detective.npz").read_bytes()
    assert repo_d != cic_d, "Repo and CIC Detective weights must not be identical!"


# ─────────────────────────────────────────────────────────────────────────────
# CLEANUP VERIFICATION — fake synthetic API artifacts must be GONE
# ─────────────────────────────────────────────────────────────────────────────

def test_fake_api_data_removed():
    """Synthetic API JSONL datasets must no longer exist."""
    assert not FAKE_API_TRAIN.exists(), \
        f"Fake API train data still present: {FAKE_API_TRAIN}"
    assert not FAKE_API_TEST.exists(), \
        f"Fake API test data still present: {FAKE_API_TEST}"


def test_fake_api_metrics_removed():
    """Synthetic API metrics JSON must no longer exist."""
    assert not FAKE_API_METRICS.exists(), \
        f"Fake API metrics still present: {FAKE_API_METRICS}"


def test_fake_api_weights_removed():
    """Synthetic API weight directory must no longer exist."""
    assert not FAKE_API_WEIGHTS.exists(), \
        f"Fake API weights directory still present: {FAKE_API_WEIGHTS}"


# ─────────────────────────────────────────────────────────────────────────────
# KRONUS API INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kronus_api_post_events():
    """KRONUS POST /events accepts a TelemetryEvent and returns correct schema."""
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
        drift_watcher=DriftWatcher(
            baseline_features={"f": np.random.default_rng(0).normal(size=10)}
        ),
    )
    await system.start()
    api_main.bind_system(system)

    try:
        client = TestClient(api_main.app)
        payload = {
            "source_ip": "192.0.2.10",
            "dest_ip": "10.0.0.1",
            "dest_port": 80,
            "protocol": "tcp",
            "bytes": 500,
            "duration_ms": 10,
            "sensor": "zeek",
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        resp = client.post("/events", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ingested"
        assert "bouncer_verdict" in data
        assert "detective_verdict" in data
        assert "decisions" in data
    finally:
        await system.stop()
        api_main._system = None
