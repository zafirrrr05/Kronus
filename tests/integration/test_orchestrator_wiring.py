from datetime import datetime, timedelta, timezone

import pytest

from libs.constants import Protocol, Sensor
from libs.schemas import TelemetryEvent
from services.decoy.ssh_honeypot import DecoySession

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _flood_events(n=200, source_ip="203.0.113.9", dest_ip="10.0.0.5"):
    # n=200 at 5ms spacing keeps every event inside FlowFeaturizer's 2s
    # window and pushes event_rate/byte_rate into the range
    # tests/integration/conftest.py's toy Bouncer was trained on (~100/s,
    # ~10000 bytes/s) — verified directly against the real featurizer
    # before picking this number, not guessed. n=60 (tried first) computed
    # event_rate=30, well outside the trained range, and never produced a
    # confident block.
    return [
        TelemetryEvent(
            source_ip=source_ip, dest_ip=dest_ip, dest_port=80, protocol=Protocol.TCP,
            bytes=100, duration_ms=2, sensor=Sensor.ZEEK,
            ts=T0 + timedelta(milliseconds=i * 5),
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_ingest_event_produces_a_bouncer_verdict_and_logs_it(kronus_system):
    outcome = await kronus_system.ingest_event(_flood_events(1)[0], now=T0)
    assert outcome.bouncer_verdict is not None
    entries = await kronus_system._logbook.read_all()
    assert any(e.entry_type == "detection" for e in entries)


@pytest.mark.asyncio
async def test_flood_sequence_eventually_produces_a_block_decision(kronus_system):
    outcomes = [await kronus_system.ingest_event(ev, now=T0) for ev in _flood_events()]
    actions = [d.action for o in outcomes for d in o.decisions]
    assert "block" in actions


@pytest.mark.asyncio
async def test_the_job_worker_registration_fix_actually_prevents_dropped_jobs(kronus_system):
    # This is the property services/pipeline/kronus_system.py's start()
    # exists for: without registering the job-queue consumer group before
    # any event is ingested, the first enqueued explanation job would be
    # silently dropped (InMemoryEventBus only delivers to groups that
    # already exist — see tests/unit/test_event_bus.py). kronus_system
    # fixture already calls start() before yielding; this test confirms
    # that actually pays off, not just that it runs without error.
    outcomes = [await kronus_system.ingest_event(ev, now=T0) for ev in _flood_events()]
    assert any(o.explanation_enqueued for o in outcomes)

    processed = await kronus_system.drain_explanation_jobs()
    assert processed >= 1

    entries = await kronus_system._logbook.read_all()
    explanations = [e for e in entries if e.entry_type == "explanation"]
    assert len(explanations) == processed


@pytest.mark.asyncio
async def test_multiple_enqueued_jobs_are_all_drained_not_just_the_first(kronus_system):
    # Exercises the defensive multi-yield in drain_explanation_jobs
    # against a real burst of several non-benign verdicts in one batch,
    # not just a single event.
    events = _flood_events() + _flood_events(200, source_ip="198.51.100.4", dest_ip="10.0.0.6")
    outcomes = [await kronus_system.ingest_event(ev, now=T0) for ev in events]
    enqueued_count = sum(1 for o in outcomes if o.explanation_enqueued)
    assert enqueued_count >= 2

    processed = await kronus_system.drain_explanation_jobs()
    assert processed == enqueued_count


@pytest.mark.asyncio
async def test_decoy_session_produces_a_block_decision_directly(kronus_system):
    session = DecoySession(
        source_ip="198.51.100.9", dest_port=22, credentials_tried=["root:toor"],
        commands_typed=["whoami"],
    )
    session.ended_at = session.started_at
    outcome = await kronus_system.handle_decoy_session(session, now=T0)
    assert outcome.decisions[0].action == "block"


@pytest.mark.asyncio
async def test_correction_reverses_a_block_and_reaches_drift_watcher(kronus_system):
    outcomes = [await kronus_system.ingest_event(ev, now=T0) for ev in _flood_events()]
    blocked_ip = next(
        ev.source_ip for ev, o in zip(_flood_events(), outcomes, strict=True)
        if any(d.action == "block" for d in o.decisions)
    )
    assert kronus_system._response_engine.is_blocked(blocked_ip, T0)

    await kronus_system.record_correction(blocked_ip, is_false_positive=True, operator="zafir")

    assert not kronus_system._response_engine.is_blocked(blocked_ip, T0)
    entries = await kronus_system._logbook.read_all()
    assert any(e.entry_type == "correction" for e in entries)
    assert kronus_system._drift_watcher._corrections_since_last_cycle == 1


@pytest.mark.asyncio
async def test_correcting_an_unknown_ip_raises(_trained_bouncer, _trained_detective):
    import numpy as np

    from libs.event_bus import InMemoryEventBus
    from pipeline.kronus_system import KronusSystem
    from services.drift_watcher.retrainer import DriftWatcher
    from services.llm_explainer.explainer import LLMExplainer
    from services.logbook.sqlite_store import SQLiteLogbookStore
    from services.response_engine.engine import ResponseEngine
    from services.response_engine.opa_client import PolicyClient

    bus = InMemoryEventBus()
    system = KronusSystem(
        bus=bus, bouncer=_trained_bouncer, detective=_trained_detective,
        response_engine=ResponseEngine(PolicyClient()), logbook=SQLiteLogbookStore(":memory:"),
        explainer=LLMExplainer(api_key=None),
        drift_watcher=DriftWatcher(baseline_features={"f": np.random.default_rng(0).normal(size=10)}),
    )
    await system.start()
    with pytest.raises(ValueError):
        await system.record_correction("9.9.9.9", is_false_positive=True, operator="zafir")
    await system.stop()
