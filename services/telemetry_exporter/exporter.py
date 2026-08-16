"""The running half of the Ingestion Pipeline: takes converted events and
actually puts them on the Event Stream, one topic, independently readable
by every downstream consumer group (see libs/event_bus.py).
"""

from __future__ import annotations

from collections.abc import Iterable

from libs.event_bus import EventBus
from libs.observability import ACTIVE_GAUGE, observe
from libs.schemas import TelemetryEvent

TELEMETRY_TOPIC = "telemetry-events"


class TelemetryExporter:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._published = 0

    async def publish(self, event: TelemetryEvent) -> None:
        with observe("telemetry_exporter", "publish", sensor=event.sensor):
            await self._bus.publish(
                TELEMETRY_TOPIC, key=event.source_ip, value=event.model_dump_json().encode()
            )
            self._published += 1
            ACTIVE_GAUGE.labels(component="telemetry_exporter", kind="published_total").set(
                self._published
            )

    async def publish_many(self, events: Iterable[TelemetryEvent]) -> int:
        count = 0
        for event in events:
            await self.publish(event)
            count += 1
        return count
