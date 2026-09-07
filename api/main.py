"""features.txt architecture diagram: "REMEMBER — always: Live Map +
hash-chained Logbook." The Live Map is a live view over Response Engine +
Logbook state, not its own component — this is that view, exposed over
HTTP/WebSocket for a frontend (see docs/frontend_guide.md).

Kept intentionally thin: every route reads from the Logbook or the
Response Engine's own state; none of them contain detection or policy
logic of their own. A KronusSystem instance is constructed once at
startup (see main()) and shared across requests.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from libs.observability import configure_tracing, start_metrics_server
from libs.schemas import TelemetryEvent
from pipeline.kronus_system import KronusSystem

app = FastAPI(title="KRONUS API", version="1.0.0")

_system: KronusSystem | None = None
_live_subscribers: set[WebSocket] = set()


def get_system() -> KronusSystem:
    if _system is None:
        raise HTTPException(status_code=503, detail="system not initialized")
    return _system


def bind_system(system: KronusSystem) -> None:
    """Called once at process startup by whatever entry point constructs
    the real KronusSystem (see demo/run_demo.py for the reference wiring;
    a production deployment's entry point does the same thing).
    """
    global _system
    _system = system


class CorrectionRequest(BaseModel):
    target_ip: str
    is_false_positive: bool
    operator: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "system_ready": _system is not None}


@app.get("/verdicts")
async def recent_verdicts(limit: int = 50) -> list[dict]:
    system = get_system()
    entries = await system._logbook.read_all()
    detections = [e for e in entries if e.entry_type == "detection"]
    return [e.payload for e in detections[-limit:]]


@app.get("/decisions")
async def recent_decisions(limit: int = 50) -> list[dict]:
    system = get_system()
    entries = await system._logbook.read_all()
    decisions = [e for e in entries if e.entry_type == "policy_decision"]
    return [e.payload for e in decisions[-limit:]]


@app.get("/logbook")
async def logbook_page(limit: int = 100) -> list[dict]:
    """The Live Map's underlying feed — every entry type, oldest first,
    the same audit trail spec.md §3.5 defines."""
    system = get_system()
    entries = await system._logbook.read_all()
    return [e.model_dump(mode="json") for e in entries[-limit:]]


@app.get("/logbook/verify")
async def verify_logbook() -> dict:
    """NFR-13's "nightly verify," exposed on demand — walks the whole
    chain and confirms no entry was altered after the fact."""
    system = get_system()
    ok, broken_at = await system._logbook.verify_chain()
    return {"intact": ok, "broken_at_index": broken_at}


@app.post("/corrections")
async def submit_correction(correction: CorrectionRequest) -> dict:
    """Case 5's human-in-the-loop path."""
    system = get_system()
    try:
        await system.record_correction(
            correction.target_ip, correction.is_false_positive, correction.operator
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "recorded"}


@app.get("/blocks/{target_ip}")
async def block_status(target_ip: str) -> dict:
    system = get_system()
    now = datetime.now(timezone.utc)
    return {"target_ip": target_ip, "blocked": system._response_engine.is_blocked(target_ip, now)}


@app.post("/events")
async def ingest_event_endpoint(event: TelemetryEvent) -> dict:
    """Ingests a single telemetry event through the KRONUS detection and policy pipeline."""
    system = get_system()
    now = datetime.now(timezone.utc)
    outcome = await system.ingest_event(event, now=now)
    return {
        "status": "ingested",
        "event_id": event.event_id,
        "bouncer_verdict": outcome.bouncer_verdict.model_dump(mode="json") if outcome.bouncer_verdict else None,
        "detective_verdict": outcome.detective_verdict.model_dump(mode="json") if outcome.detective_verdict else None,
        "decisions": [d.model_dump(mode="json") for d in outcome.decisions],
    }


@app.post("/events/batch")
async def ingest_events_batch_endpoint(events: list[TelemetryEvent]) -> dict:
    """Ingests a batch of telemetry events through the KRONUS pipeline."""
    system = get_system()
    now = datetime.now(timezone.utc)
    outcomes = []
    for ev in events:
        outcomes.append(await system.ingest_event(ev, now=now))
    return {
        "status": "ingested",
        "count": len(events),
        "bouncer_verdicts": sum(1 for o in outcomes if o.bouncer_verdict),
        "detective_verdicts": sum(1 for o in outcomes if o.detective_verdict),
        "decisions": sum(len(o.decisions) for o in outcomes),
    }


@app.websocket("/live")
async def live_feed(websocket: WebSocket) -> None:
    """Streams new Logbook entries as they're appended — the Live Map's
    real-time channel. Polling-based (short interval), not a true
    push subscription, since LogbookStore's interface (base.py) doesn't
    define a change-notification hook — see docs/frontend_guide.md for
    the extension point if a production deployment wants push instead.
    """
    await websocket.accept()
    _live_subscribers.add(websocket)
    system = get_system()
    last_seen = 0
    try:
        import asyncio

        while True:
            entries = await system._logbook.read_all()
            if len(entries) > last_seen:
                for entry in entries[last_seen:]:
                    await websocket.send_json(entry.model_dump(mode="json"))
                last_seen = len(entries)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        _live_subscribers.discard(websocket)


def start_observability() -> None:
    configure_tracing()
    start_metrics_server()
