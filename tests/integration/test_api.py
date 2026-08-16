import numpy as np
import pytest_asyncio
from fastapi.testclient import TestClient

from libs.event_bus import InMemoryEventBus
from pipeline.kronus_system import KronusSystem
from services.drift_watcher.retrainer import DriftWatcher
from services.llm_explainer.explainer import LLMExplainer
from services.logbook.sqlite_store import SQLiteLogbookStore
from services.response_engine.engine import ResponseEngine
from services.response_engine.opa_client import PolicyClient
from tests.integration.conftest import _fast_bouncer, _fast_detective


@pytest_asyncio.fixture
async def api_client():
    import api.main as api_main

    bus = InMemoryEventBus()
    system = KronusSystem(
        bus=bus, bouncer=_fast_bouncer(), detective=_fast_detective(),
        response_engine=ResponseEngine(PolicyClient()), logbook=SQLiteLogbookStore(":memory:"),
        explainer=LLMExplainer(api_key=None),
        drift_watcher=DriftWatcher(baseline_features={"f": np.random.default_rng(0).normal(size=10)}),
    )
    await system.start()
    api_main.bind_system(system)

    from datetime import datetime, timedelta, timezone

    from libs.constants import Protocol, Sensor
    from libs.schemas import TelemetryEvent

    # Real wall-clock time, not a fixed historical replay timestamp: the
    # /blocks endpoint correctly checks TTL expiry against
    # datetime.now(timezone.utc) (that's the right behavior for a real
    # client asking "is this blocked right now"), so exercising it here
    # needs the same clock the API itself uses — a fixed t0 like the
    # flow-case tests use for deterministic replay would already read as
    # "expired" by real time, which is a different bug entirely from
    # what this fixture is meant to set up.
    now = datetime.now(timezone.utc)
    for i in range(200):
        ev = TelemetryEvent(
            source_ip="203.0.113.9", dest_ip="10.0.0.5", dest_port=80, protocol=Protocol.TCP,
            bytes=100, duration_ms=2, sensor=Sensor.ZEEK, ts=now + timedelta(milliseconds=i * 5),
        )
        await system.ingest_event(ev, now=now)

    yield TestClient(api_main.app), system
    await system.stop()
    api_main._system = None


def test_health_reports_ready(api_client):
    client, _ = api_client
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "system_ready": True}


def test_verdicts_endpoint_returns_real_logged_detections(api_client):
    client, _ = api_client
    response = client.get("/verdicts")
    assert response.status_code == 200
    body = response.json()
    assert len(body) > 0
    assert "tier" in body[0]


def test_decisions_endpoint_returns_real_policy_decisions(api_client):
    client, _ = api_client
    response = client.get("/decisions")
    assert response.status_code == 200
    assert any(d["action"] == "block" for d in response.json())


def test_logbook_verify_reports_intact_chain(api_client):
    client, _ = api_client
    response = client.get("/logbook/verify")
    assert response.status_code == 200
    assert response.json() == {"intact": True, "broken_at_index": None}


def test_block_status_reflects_a_real_block(api_client):
    client, _ = api_client
    response = client.get("/blocks/203.0.113.9")
    assert response.status_code == 200
    assert response.json()["blocked"] is True


def test_correction_reverses_a_real_block_via_the_api(api_client):
    client, _ = api_client
    response = client.post(
        "/corrections",
        json={"target_ip": "203.0.113.9", "is_false_positive": True, "operator": "zafir"},
    )
    assert response.status_code == 200
    assert client.get("/blocks/203.0.113.9").json()["blocked"] is False


def test_correction_for_unknown_ip_returns_404(api_client):
    client, _ = api_client
    response = client.post(
        "/corrections",
        json={"target_ip": "9.9.9.9", "is_false_positive": True, "operator": "zafir"},
    )
    assert response.status_code == 404


def test_health_without_bound_system_reports_not_ready():
    import api.main as api_main

    api_main._system = None
    client = TestClient(api_main.app)
    response = client.get("/health")
    assert response.json() == {"status": "ok", "system_ready": False}
