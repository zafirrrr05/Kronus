from datetime import datetime, timedelta, timezone

from libs.constants import Protocol, Sensor
from libs.schemas import TelemetryEvent
from services.bouncer.features import FlowFeaturizer

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _event(src, dst, port, ts, bytes_=100, duration_ms=5) -> TelemetryEvent:
    return TelemetryEvent(
        source_ip=src, dest_ip=dst, dest_port=port, protocol=Protocol.TCP,
        bytes=bytes_, duration_ms=duration_ms, sensor=Sensor.ZEEK, ts=ts,
    )


def test_flood_signature_has_high_rate_and_low_port_entropy():
    featurizer = FlowFeaturizer(window_seconds=2.0)
    feats = {}
    for i in range(50):
        ev = _event("203.0.113.1", "10.0.0.5", 80, ts=T0 + timedelta(milliseconds=i * 10))
        feats = featurizer.features_for(ev)
    assert feats["event_rate"] > 20  # 50 events in ~0.5s window slice
    assert feats["dest_port_entropy"] == 0.0  # always port 80
    assert feats["same_dest_ratio"] == 1.0  # always the same victim


def test_scan_signature_has_high_port_entropy_and_low_same_dest_ratio():
    featurizer = FlowFeaturizer(window_seconds=2.0)
    feats = {}
    for i in range(20):
        ev = _event("203.0.113.2", f"10.0.0.{i}", 1000 + i, ts=T0 + timedelta(milliseconds=i * 10))
        feats = featurizer.features_for(ev)
    assert feats["dest_port_entropy"] > 3.0  # 20 distinct ports, high entropy
    assert feats["unique_dest_count"] == 20
    assert feats["same_dest_ratio"] < 0.1  # never repeats a destination


def test_window_evicts_events_older_than_window_seconds():
    featurizer = FlowFeaturizer(window_seconds=2.0)
    featurizer.features_for(_event("a", "b", 80, ts=T0))
    # jump forward well past the window — the first event should no longer count
    feats = featurizer.features_for(_event("a", "c", 443, ts=T0 + timedelta(seconds=5)))
    assert feats["unique_dest_count"] == 1  # only "c", "b" was evicted


def test_different_sources_have_independent_windows():
    featurizer = FlowFeaturizer(window_seconds=2.0)
    for i in range(10):
        featurizer.features_for(_event("attacker", f"10.0.0.{i}", 1000 + i, ts=T0))
    feats_other = featurizer.features_for(_event("normal-user", "10.0.0.99", 80, ts=T0))
    assert feats_other["unique_dest_count"] == 1  # unaffected by "attacker"'s history
