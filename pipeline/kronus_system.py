"""The composition root. Every one of the 27 edges verified against
features.txt's connectivity map, spec.md, and the drawio sketch is either
a direct call or a bus hop below — this module is what makes "the system"
an actual running thing rather than 11 separately-tested components.

Not a live daemon (that's a documented extension point — see
docs/instructions.md — the pieces are async-ready: WindowedGraphBuilder,
FlowFeaturizer, and EventBus.subscribe() all support it). This is the
synchronous ingest path the demo and the integration flow-case tests use:
deterministic, real data in, real verdicts/decisions/log entries out.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from libs.constants import AuditEntryType, Label, PolicyAction
from libs.event_bus import EventBus
from libs.observability import observe
from libs.schemas import DetectionVerdict, ExplanationJob, PolicyDecision, TelemetryEvent
from services.bouncer.features import FlowFeaturizer
from services.bouncer.model import BouncerModel
from services.decoy.pipeline import DecoyPipeline
from services.decoy.ssh_honeypot import DecoySession
from services.detective.model import DetectiveModel
from services.drift_watcher.retrainer import DriftWatcher
from services.graph_builder.builder import WindowedGraphBuilder
from services.llm_explainer.explainer import LLMExplainer
from services.llm_explainer.job_queue import consume, enqueue
from services.logbook.base import LogbookStore
from services.response_engine.engine import PolicyRejectedError, ResponseEngine
from services.telemetry_exporter.exporter import TelemetryExporter


class Outcome:
    """One event's full trip through the system — what the demo and
    integration tests inspect to assert on."""

    def __init__(self) -> None:
        self.bouncer_verdict: DetectionVerdict | None = None
        self.detective_verdict: DetectionVerdict | None = None
        self.decisions: list[PolicyDecision] = []
        self.explanation_enqueued = False
        self.rejected: DetectionVerdict | None = None


class KronusSystem:
    def __init__(
        self, bus: EventBus, bouncer: BouncerModel, detective: DetectiveModel,
        response_engine: ResponseEngine, logbook: LogbookStore, explainer: LLMExplainer,
        drift_watcher: DriftWatcher,
    ) -> None:
        self._bus = bus
        self._exporter = TelemetryExporter(bus)
        self._featurizer = FlowFeaturizer()
        self._graph_builder = WindowedGraphBuilder()
        self._bouncer = bouncer
        self._detective = detective
        self._response_engine = response_engine
        self._logbook = logbook
        self._explainer = explainer
        self._drift_watcher = drift_watcher
        # DecoyPipeline requires an on_verdict callback synchronously (a
        # plain call, not awaited) inside handle_session — but the actual
        # decide()+log() handling is async (_handle_verdict). The two
        # can't be the same function, so the callback here is a no-op;
        # handle_decoy_session calls _handle_verdict explicitly right
        # after handle_session() returns, using that same verdict object.
        self._decoy_pipeline = DecoyPipeline(bus, on_verdict=lambda verdict: None)
        self._last_decision_id_by_ip: dict[str, str] = {}
        self._received_jobs: list[ExplanationJob] = []
        self._job_worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Registers the job-queue consumer group before anything can be
        enqueued to it. Must be called once before ingest_event/
        handle_decoy_session — otherwise the first explanation job
        published would be silently dropped (InMemoryEventBus delivers
        only to consumer groups that already exist; see
        tests/unit/test_event_bus.py's publish-before-subscribe test).
        A background task is what actually keeps the group "listening";
        asyncio.sleep(0) yields control once so that task reaches its
        first await (queue.get(), inside consume()/subscribe()) — which
        is where registration happens, synchronously, before the
        suspend point — confirmed by test in
        tests/integration/test_orchestrator_wiring.py, not assumed.
        """
        self._job_worker_task = asyncio.create_task(self._job_worker_loop())
        await asyncio.sleep(0)

    async def _job_worker_loop(self) -> None:
        async for job in consume(self._bus):
            self._received_jobs.append(job)

    async def stop(self) -> None:
        if self._job_worker_task is not None:
            self._job_worker_task.cancel()
            try:
                await self._job_worker_task
            except asyncio.CancelledError:
                pass

    # --- fast lane + deep lane: real network/replay traffic -----------------

    async def ingest_event(self, event: TelemetryEvent, now: datetime) -> Outcome:
        """features.txt edges 6+7 (Ingestion -> Graph Builder, Ingestion ->
        Bouncer): one event feeds both lanes independently. Edge 9
        (Graph Builder -> Detective) fires only when a window closes.
        """
        with observe("orchestrator", "ingest_event", source_ip=event.source_ip):
            await self._exporter.publish(event)  # edge: onto the Event Stream
            outcome = Outcome()

            # fast lane
            bouncer_features = self._featurizer.features_for(event)
            window_id = f"bouncer-{event.event_id}"
            bouncer_verdict = self._bouncer.predict_verdict(bouncer_features, window_id=window_id)
            outcome.bouncer_verdict = bouncer_verdict
            await self._handle_verdict(
                bouncer_verdict, target_ip=event.source_ip, now=now, outcome=outcome
            )

            # deep lane — only produces a verdict when a window closes
            snapshot = self._graph_builder.ingest(event)
            if snapshot is not None:
                detective_verdict = self._detective.predict_verdict(
                    snapshot, window_id=snapshot.window_id
                )
                outcome.detective_verdict = detective_verdict
                target_ip = (
                    detective_verdict.evidence.node_ids[0]
                    if detective_verdict.evidence.node_ids
                    else event.source_ip
                )
                await self._handle_verdict(
                    detective_verdict, target_ip=target_ip, now=now, outcome=outcome
                )

            return outcome

    def flush_graph_window(self, now: datetime) -> DetectionVerdict | None:
        """Forces the current graph window closed (end of a replay batch)
        rather than losing a partial window's evidence — see
        WindowedGraphBuilder.flush.
        """
        snapshot = self._graph_builder.flush(end_time=now)
        if snapshot is None:
            return None
        return self._detective.predict_verdict(snapshot, window_id=snapshot.window_id)

    # --- decoy: bypass path, edges 4 (sensor) and 5 (direct to Response Engine) --

    async def handle_decoy_session(self, session: DecoySession, now: datetime) -> Outcome:
        outcome = Outcome()
        verdict = await self._decoy_pipeline.handle_session(session)
        # The on_verdict callback passed to DecoyPipeline is a no-op (see
        # __init__) — decision/logging happens here instead, using this
        # same verdict object, since _handle_verdict is async and
        # DecoyPipeline's callback contract is synchronous.
        await self._handle_verdict(verdict, target_ip=session.source_ip, now=now, outcome=outcome)
        self._drift_watcher.record_decoy_session()  # edge 8: decoy signal reaches Drift Watcher
        return outcome

    # --- shared decision + logging path, used by every tier -----------------

    # --- shared decision + logging path, used by every tier -----------------
    async def _handle_verdict(
        self, verdict: DetectionVerdict, target_ip: str, now: datetime, outcome: Outcome
    ) -> None:
        await self._logbook.append(
            AuditEntryType.DETECTION,
            {"verdict_id": verdict.verdict_id, "tier": verdict.tier, "label": verdict.label,
             "confidence": verdict.confidence, "window_id": verdict.window_id},
        )

        try:
            decision = self._response_engine.decide(verdict, target_ip=target_ip, now=now)
        except PolicyRejectedError:
            outcome.rejected = verdict
            await self._logbook.append(
                AuditEntryType.POLICY_DECISION,
                {"verdict_id": verdict.verdict_id, "rejected": True, "reason": "malformed_input"},
            )
            return

        outcome.decisions.append(decision)
        self._last_decision_id_by_ip[target_ip] = decision.decision_id
        await self._logbook.append(
            AuditEntryType.POLICY_DECISION,
            {"decision_id": decision.decision_id, "verdict_id": decision.verdict_id,
             "action": decision.action, "reason_codes": decision.reason_codes,
             "target_ip": target_ip},
        )

        if decision.action in (PolicyAction.BLOCK, PolicyAction.THROTTLED):
            await self._logbook.append(
                AuditEntryType.ENFORCEMENT,
                {"decision_id": decision.decision_id, "target_ip": target_ip,
                 "action": decision.action, "ttl_seconds": decision.ttl_seconds},
            )

        # cold path: explain anything that wasn't just "nothing happened"
        if verdict.label != Label.BENIGN:
            await enqueue(self._bus, ExplanationJob(verdict_id=verdict.verdict_id))
            outcome.explanation_enqueued = True

    # --- human-in-the-loop correction, Case 5 --------------------------------

    async def record_correction(
        self, target_ip: str, is_false_positive: bool, operator: str
    ) -> None:
        from libs.constants import CorrectionVerdict
        from libs.schemas import Correction

        if is_false_positive:
            self._response_engine.reverse_block(target_ip)
        decision_id = self._last_decision_id_by_ip.get(target_ip)
        if decision_id is None:
            raise ValueError(f"no known decision for target_ip={target_ip!r} to correct")

        verdict_was = (
            CorrectionVerdict.FALSE_POSITIVE if is_false_positive
            else CorrectionVerdict.TRUE_POSITIVE
        )
        correction = Correction(decision_id=decision_id, operator=operator, verdict_was=verdict_was)
        await self._logbook.append(
            AuditEntryType.CORRECTION, correction.model_dump(mode="json"),
            actor=f"operator:{operator}",
        )
        self._drift_watcher.record_correction()  # edge 16: Logbook corrections -> Drift Watcher

    # --- run explanation workers (cold path, drains the job queue) ----------

    async def drain_explanation_jobs(self) -> int:
        """Processes every job the background worker task (started by
        start()) has collected so far (edge 15: Response Engine -> LLM
        Explainer via Job Queue; edge 17: LLM Explainer -> Logbook).
        """
        # Give the background worker task a few scheduling turns to drain
        # anything already sitting in the bus queue before reading what
        # it's collected — defensive rather than assumed, since asyncio
        # doesn't guarantee a background task drains a multi-item queue
        # within a single await point in the caller.
        for _ in range(3):
            await asyncio.sleep(0)

        entries = await self._logbook.read_all()
        detections = {
            e.payload["verdict_id"]: e.payload
            for e in entries
            if e.entry_type == AuditEntryType.DETECTION
        }

        to_process, self._received_jobs = self._received_jobs, []
        processed = 0
        for job in to_process:
            payload = detections.get(job.verdict_id)
            if payload is None:
                continue
            verdict = _verdict_from_payload(payload)
            result = self._explainer.explain(verdict, job_id=job.job_id)
            await self._logbook.append(
                AuditEntryType.EXPLANATION,
                {"job_id": result.job_id, "verdict_id": result.verdict_id,
                 "narrative": result.narrative, "attck_technique_id": result.attck_technique_id},
            )
            processed += 1
        return processed


def _verdict_from_payload(payload: dict) -> DetectionVerdict:
    from libs.constants import AttributionMethod
    from libs.schemas import VerdictEvidence

    return DetectionVerdict(
        window_id=payload["window_id"], tier=payload["tier"], label=payload["label"],
        confidence=payload["confidence"],
        evidence=VerdictEvidence(attribution_method=AttributionMethod.RATE_THRESHOLD),
    )
