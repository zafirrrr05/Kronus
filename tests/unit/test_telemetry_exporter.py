import asyncio
import json

import pytest

from libs.constants import DataOrigin, Label, Protocol, Sensor
from libs.event_bus import InMemoryEventBus
from services.telemetry_exporter.converters import decoy_session_to_event, flow_row_to_event
from services.telemetry_exporter.exporter import TELEMETRY_TOPIC, TelemetryExporter
from twin.nsl_kdd import NSLKDDRow


def _sample_row() -> NSLKDDRow:
    return NSLKDDRow(
        source_ip="10.10.1.2", dest_ip="10.20.3.4", source_port=None, dest_port=80,
        protocol=Protocol.TCP, total_bytes=1500, duration_ms=12000, raw_label="normal",
        category="normal", kronus_label=Label.BENIGN, difficulty=20, features={},
        origin=DataOrigin.REAL,
    )


def test_flow_row_to_event_preserves_core_fields():
    event = flow_row_to_event(_sample_row())
    assert event.source_ip == "10.10.1.2"
    assert event.dest_ip == "10.20.3.4"
    assert event.dest_port == 80
    assert event.protocol == Protocol.TCP
    assert event.bytes == 1500
    assert event.sensor == Sensor.ZEEK


def test_flow_row_to_event_clamps_negative_values():
    row = _sample_row()
    negative = NSLKDDRow(**{**row.__dict__, "total_bytes": -5, "duration_ms": -1})
    event = flow_row_to_event(negative)
    assert event.bytes == 0
    assert event.duration_ms == 0


def test_flow_row_to_event_honors_explicit_timestamp():
    from datetime import datetime, timezone

    fixed_ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    event = flow_row_to_event(_sample_row(), ts=fixed_ts)
    assert event.ts == fixed_ts


def test_decoy_session_to_event_is_honeypot_tagged():
    event = decoy_session_to_event(
        source_ip="198.51.100.9", dest_ip="10.30.0.254", dest_port=22,
        decoy_service="ssh", credentials_tried=["root:toor", "admin:admin"],
        commands_typed=["whoami", "ls -la"], session_duration_ms=4200,
    )
    assert event.sensor == Sensor.HONEYPOT
    assert event.honeypot_detail.decoy_service == "ssh"
    assert event.honeypot_detail.commands_typed == ["whoami", "ls -la"]


@pytest.mark.asyncio
async def test_exporter_publish_reaches_independent_consumer_groups():
    bus = InMemoryEventBus()
    exporter = TelemetryExporter(bus)
    bouncer_seen, graph_builder_seen = [], []

    async def bouncer_reader():
        async for raw in bus.subscribe(TELEMETRY_TOPIC, "bouncer-consumers"):
            bouncer_seen.append(json.loads(raw))
            return

    async def graph_builder_reader():
        async for raw in bus.subscribe(TELEMETRY_TOPIC, "graph-builder-consumers"):
            graph_builder_seen.append(json.loads(raw))
            return

    t1 = asyncio.create_task(bouncer_reader())
    t2 = asyncio.create_task(graph_builder_reader())
    await asyncio.sleep(0.01)

    await exporter.publish(flow_row_to_event(_sample_row()))
    await asyncio.wait_for(asyncio.gather(t1, t2), timeout=1)

    assert bouncer_seen[0]["source_ip"] == "10.10.1.2"
    assert graph_builder_seen[0]["source_ip"] == "10.10.1.2"
    assert exporter._published == 1
