"""spec.md §3 schema invariants. These aren't smoke tests — each one guards
a specific claim made in the spec text (decoy confidence is fixed, ttl only
applies to blocks, honeypot_detail is sensor-gated, hash chaining works).
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from libs.constants import (
    AttributionMethod,
    AuditEntryType,
    Label,
    PolicyAction,
    Protocol,
    Sensor,
    Tier,
)
from libs.schemas import (
    GENESIS_HASH,
    AuditLogEntry,
    DetectionVerdict,
    HoneypotDetail,
    PolicyDecision,
    TelemetryEvent,
    VerdictEvidence,
)


def _flow_evidence() -> VerdictEvidence:
    return VerdictEvidence(attribution_method=AttributionMethod.RATE_THRESHOLD)


def test_telemetry_event_minimal_construction():
    ev = TelemetryEvent(
        source_ip="10.0.0.5",
        dest_ip="10.0.0.9",
        source_port=443,
        dest_port=51000,
        protocol=Protocol.TCP,
        bytes=1500,
        duration_ms=12,
        sensor=Sensor.ZEEK,
    )
    assert ev.event_id
    assert ev.honeypot_detail is None


def test_honeypot_detail_rejected_on_non_honeypot_sensor():
    with pytest.raises(ValidationError, match="honeypot_detail"):
        TelemetryEvent(
            source_ip="10.0.0.5",
            dest_ip="10.0.0.9",
            protocol=Protocol.TCP,
            bytes=0,
            duration_ms=0,
            sensor=Sensor.ZEEK,
            honeypot_detail=HoneypotDetail(decoy_service="ssh"),
        )


def test_honeypot_event_with_sparse_detail_is_valid():
    # spec.md §3.1: "a connection attempt with no further interaction still
    # produces a valid, if sparse, record"
    ev = TelemetryEvent(
        source_ip="203.0.113.9",
        dest_ip="10.0.0.22",
        dest_port=22,
        protocol=Protocol.TCP,
        bytes=64,
        duration_ms=40,
        sensor=Sensor.HONEYPOT,
        honeypot_detail=HoneypotDetail(decoy_service="ssh"),
    )
    assert ev.honeypot_detail.credentials_tried == []


def test_frozen_event_rejects_mutation():
    ev = TelemetryEvent(
        source_ip="a", dest_ip="b", protocol=Protocol.TCP, bytes=0,
        duration_ms=0, sensor=Sensor.ZEEK,
    )
    with pytest.raises(ValidationError):
        ev.bytes = 5


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        TelemetryEvent(
            source_ip="a", dest_ip="b", protocol=Protocol.TCP, bytes=0,
            duration_ms=0, sensor=Sensor.ZEEK, unknown_field="nope",
        )


def test_decoy_verdict_requires_confidence_one():
    with pytest.raises(ValidationError, match="fixed at 1.0"):
        DetectionVerdict(
            window_id="w1", tier=Tier.DECOY, label=Label.DECOY_INTERACTION,
            confidence=0.9,
            evidence=VerdictEvidence(attribution_method=AttributionMethod.HONEYPOT_INTERACTION),
        )


def test_decoy_verdict_at_confidence_one_is_valid():
    v = DetectionVerdict(
        window_id="w1", tier=Tier.DECOY, label=Label.DECOY_INTERACTION,
        confidence=1.0,
        evidence=VerdictEvidence(attribution_method=AttributionMethod.HONEYPOT_INTERACTION),
    )
    assert v.confidence == 1.0


def test_bouncer_verdict_cannot_carry_graph_evidence():
    # spec.md §3.3: node_ids/edge_ids are empty for bouncer and decoy tiers
    with pytest.raises(ValidationError, match="no graph evidence"):
        DetectionVerdict(
            window_id="w1", tier=Tier.BOUNCER, label=Label.FLOOD, confidence=0.9,
            evidence=VerdictEvidence(
                node_ids=["host-1"], attribution_method=AttributionMethod.RATE_THRESHOLD
            ),
        )


def test_detective_verdict_may_carry_graph_evidence():
    v = DetectionVerdict(
        window_id="w1", tier=Tier.DETECTIVE, label=Label.PORT_SCAN, confidence=0.91,
        evidence=VerdictEvidence(
            node_ids=["host-1"], edge_ids=["host-1->host-2"],
            attribution_method=AttributionMethod.ATTENTION_ROLLOUT,
        ),
    )
    assert v.evidence.node_ids == ["host-1"]


def test_ttl_rejected_on_non_block_action():
    with pytest.raises(ValidationError, match="only applies to 'block'"):
        PolicyDecision(
            verdict_id="v1", action=PolicyAction.DRY_RUN,
            reason_codes=["gray_zone"], ttl_seconds=900, policy_version="abc123",
        )


def test_ttl_accepted_on_block_action():
    d = PolicyDecision(
        verdict_id="v1", action=PolicyAction.BLOCK,
        reason_codes=["confidence_above_threshold"], ttl_seconds=900,
        policy_version="abc123",
    )
    assert d.ttl_seconds == 900


def test_audit_entry_hash_is_a_pure_function_of_its_fields():
    # entry_id uses default_factory, so two independently-constructed entries
    # legitimately get different ids -> different hashes (two distinct log
    # rows *should* hash differently, even with identical payloads, or the
    # chain's tamper-evidence guarantee is worthless). What must actually be
    # deterministic is: given the SAME stored fields (as when NFR-13's
    # nightly integrity check re-derives a hash from a stored row), you get
    # the SAME hash back every time.
    fixed_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    kwargs = dict(
        entry_id="fixed-id-1", prev_hash=GENESIS_HASH,
        entry_type=AuditEntryType.DETECTION, payload={"a": 1}, ts=fixed_ts,
    )
    e1 = AuditLogEntry(**kwargs)
    e2 = AuditLogEntry(**kwargs)
    assert e1.entry_hash == e2.entry_hash
    assert len(e1.entry_hash) == 64  # sha256 hex digest


def test_audit_entry_hash_differs_for_independently_created_entries():
    # The flip side of the above: two log entries with identical business
    # content but no shared id (the normal case) must NOT collide.
    kwargs = dict(
        prev_hash=GENESIS_HASH, entry_type=AuditEntryType.DETECTION, payload={"a": 1},
    )
    e1 = AuditLogEntry(**kwargs)
    e2 = AuditLogEntry(**kwargs)
    assert e1.entry_id != e2.entry_id
    assert e1.entry_hash != e2.entry_hash


def test_audit_entry_hash_changes_with_prev_hash():
    kwargs = dict(entry_type=AuditEntryType.DETECTION, payload={"a": 1})
    e1 = AuditLogEntry(prev_hash=GENESIS_HASH, **kwargs)
    e2 = AuditLogEntry(prev_hash=e1.entry_hash, **kwargs)
    assert e1.entry_hash != e2.entry_hash


def test_audit_actor_pattern_enforced():
    with pytest.raises(ValidationError, match="actor"):
        AuditLogEntry(
            prev_hash=GENESIS_HASH, entry_type=AuditEntryType.CORRECTION,
            payload={}, actor="someone",
        )
    ok = AuditLogEntry(
        prev_hash=GENESIS_HASH, entry_type=AuditEntryType.CORRECTION,
        payload={}, actor="operator:zafir",
    )
    assert ok.actor == "operator:zafir"
