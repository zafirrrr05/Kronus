"""features.txt component 11: "Two roles: as a sensor, parallel to the
Telemetry Exporters, feeding the Event Stream. As a detector, a direct
deterministic path straight to the Response Engine, skipping the Graph
Builder, Bouncer, and Detective entirely."

This module is that fan-out: one finished DecoySession becomes both
artifacts. No scoring happens here — confidence is fixed at 1.0 by
construction (enforced again by DetectionVerdict's own validator in
libs/schemas.py), matching "any interaction is a true positive by
construction" (features.txt Part 0).
"""

from __future__ import annotations

from libs.constants import AttributionMethod, Label, Tier
from libs.event_bus import EventBus
from libs.observability import observe
from libs.schemas import DetectionVerdict, VerdictEvidence
from services.decoy.ssh_honeypot import DecoySession
from services.telemetry_exporter.converters import decoy_session_to_event
from services.telemetry_exporter.exporter import TELEMETRY_TOPIC


def decoy_verdict(session: DecoySession, window_id: str) -> DetectionVerdict:
    return DetectionVerdict(
        window_id=window_id,
        tier=Tier.DECOY,
        label=Label.DECOY_INTERACTION,
        confidence=1.0,
        evidence=VerdictEvidence(attribution_method=AttributionMethod.HONEYPOT_INTERACTION),
    )


class DecoyPipeline:
    """Wires a running honeypot's completed sessions to both destinations.
    `on_verdict` is the direct, deterministic path to the Response Engine
    (features.txt: "skipping the Graph Builder, Bouncer, and Detective
    entirely") — a plain callback, not routed through the Event Stream,
    since it is not a lane and does not compete with either analytical
    tier for the bus.
    """

    def __init__(self, bus: EventBus, on_verdict) -> None:
        self._bus = bus
        self._on_verdict = on_verdict

    async def handle_session(self, session: DecoySession) -> DetectionVerdict:
        with observe("decoy", "handle_session", source_ip=session.source_ip):
            event = decoy_session_to_event(
                source_ip=session.source_ip,
                dest_ip="internal",  # the decoy host itself; no real dest to name
                dest_port=session.dest_port,
                decoy_service=session.decoy_service,
                credentials_tried=session.credentials_tried,
                commands_typed=session.commands_typed,
                session_duration_ms=session.duration_ms,
            )
            await self._bus.publish(TELEMETRY_TOPIC, key=session.source_ip, value=event.model_dump_json().encode())

            verdict = decoy_verdict(session, window_id=event.event_id)
            self._on_verdict(verdict)
            return verdict
