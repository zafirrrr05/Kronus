from datetime import datetime, timedelta, timezone

from libs.constants import Protocol, Sensor
from libs.schemas import TelemetryEvent
from services.graph_builder.builder import WindowedGraphBuilder, _port_entropy, build_snapshot

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _event(src, dst, port, bytes_=100, duration_ms=10, ts=T0) -> TelemetryEvent:
    return TelemetryEvent(
        source_ip=src, dest_ip=dst, dest_port=port, protocol=Protocol.TCP,
        bytes=bytes_, duration_ms=duration_ms, sensor=Sensor.ZEEK, ts=ts,
    )


def test_port_entropy_of_single_repeated_port_is_zero():
    assert _port_entropy([80, 80, 80]) == 0.0


def test_port_entropy_of_uniform_distinct_ports_is_maximal():
    # 4 equally-likely distinct ports -> entropy == log2(4) == 2.0
    assert _port_entropy([1, 2, 3, 4]) == 2.0


def test_snapshot_builds_one_node_per_distinct_host():
    events = [_event("10.0.0.1", "10.0.0.2", 80), _event("10.0.0.2", "10.0.0.3", 443)]
    snap = build_snapshot(events, window_start=T0, window_end=T0 + timedelta(seconds=2))
    assert {n.node_id for n in snap.nodes} == {"10.0.0.1", "10.0.0.2", "10.0.0.3"}


def test_snapshot_aggregates_repeated_edges_into_one_with_correct_flow_count():
    events = [_event("10.0.0.1", "10.0.0.2", 80) for _ in range(5)]
    snap = build_snapshot(events, window_start=T0, window_end=T0 + timedelta(seconds=2))
    assert len(snap.edges) == 1
    assert snap.edges[0].flow_count == 5
    assert snap.edges[0].bytes == 500.0


def test_port_scan_shape_is_visible_as_high_fan_out_and_port_entropy():
    # one attacker, six distinct destination hosts, distinct ports each —
    # exactly the fan-out shape a fixed-threshold rate check would miss
    # (features.txt Case 2) but a graph snapshot makes structural.
    attacker = "203.0.113.9"
    events = [_event(attacker, f"10.0.0.{i}", 1000 + i) for i in range(6)]
    snap = build_snapshot(events, window_start=T0, window_end=T0 + timedelta(seconds=2))
    attacker_node = next(n for n in snap.nodes if n.node_id == attacker)
    assert attacker_node.degree_out == 6
    assert attacker_node.unique_ports_contacted == 6
    # a benign node in the same window, single conversation, no fan-out
    victim_node = next(n for n in snap.nodes if n.node_id == "10.0.0.0")
    assert victim_node.degree_in == 1


def test_windowed_builder_emits_snapshot_when_window_elapses():
    builder = WindowedGraphBuilder(window_seconds=2.0)
    assert builder.ingest(_event("a", "b", 80, ts=T0)) is None
    assert builder.ingest(_event("a", "c", 80, ts=T0 + timedelta(seconds=1))) is None
    snap = builder.ingest(_event("a", "d", 80, ts=T0 + timedelta(seconds=2, milliseconds=1)))
    assert snap is not None
    assert len(snap.edges) == 2  # a->b, a->c — the event that closed the window starts the next one


def test_windowed_builder_flush_closes_a_partial_window():
    builder = WindowedGraphBuilder(window_seconds=2.0)
    builder.ingest(_event("a", "b", 80, ts=T0))
    snap = builder.flush(end_time=T0 + timedelta(milliseconds=500))
    assert snap is not None
    assert len(snap.edges) == 1
    assert builder.flush(end_time=T0) is None  # nothing left to flush
