from datetime import datetime, timedelta, timezone

import pytest

from libs.constants import AttributionMethod, Label, PolicyAction, Tier
from libs.schemas import DetectionVerdict, VerdictEvidence
from services.response_engine.circuit_breaker import CircuitBreaker
from services.response_engine.correlator import WindowCorrelator
from services.response_engine.engine import PolicyRejectedError, ResponseEngine
from services.response_engine.opa_client import PolicyClient

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _bouncer_verdict(window_id="w1", confidence=0.9, label=Label.FLOOD) -> DetectionVerdict:
    return DetectionVerdict(
        window_id=window_id, tier=Tier.BOUNCER, label=label, confidence=confidence,
        evidence=VerdictEvidence(attribution_method=AttributionMethod.RATE_THRESHOLD),
    )


def _decoy_verdict(window_id="w-decoy") -> DetectionVerdict:
    return DetectionVerdict(
        window_id=window_id, tier=Tier.DECOY, label=Label.DECOY_INTERACTION, confidence=1.0,
        evidence=VerdictEvidence(attribution_method=AttributionMethod.HONEYPOT_INTERACTION),
    )


@pytest.fixture(scope="module")
def policy_client() -> PolicyClient:
    return PolicyClient()


# --- CircuitBreaker ------------------------------------------------------

def test_circuit_breaker_trips_at_the_configured_cap():
    breaker = CircuitBreaker(max_actions_per_minute=3)
    for _ in range(3):
        breaker.record_action(T0)
    assert breaker.is_tripped(T0) is True


def test_circuit_breaker_evicts_actions_older_than_one_minute():
    breaker = CircuitBreaker(max_actions_per_minute=3)
    for _ in range(3):
        breaker.record_action(T0)
    later = T0 + timedelta(seconds=61)
    assert breaker.is_tripped(later) is False


# --- WindowCorrelator ------------------------------------------------------

def test_correlator_only_records_acting_decisions():
    correlator = WindowCorrelator()
    correlator.record("w1", PolicyAction.DRY_RUN)
    assert correlator.prior_action_for("w1") is None  # dry_run doesn't consume the slot
    correlator.record("w1", PolicyAction.BLOCK)
    assert correlator.prior_action_for("w1") == PolicyAction.BLOCK


def test_correlator_keeps_first_acting_action_not_the_latest():
    correlator = WindowCorrelator()
    correlator.record("w1", PolicyAction.BLOCK)
    correlator.record("w1", PolicyAction.THROTTLED)
    assert correlator.prior_action_for("w1") == PolicyAction.BLOCK


# --- ResponseEngine, against the real opa binary --------------------------

def test_high_confidence_flood_blocks(policy_client):
    engine = ResponseEngine(policy_client)
    decision = engine.decide(_bouncer_verdict(confidence=0.95), target_ip="203.0.113.9", now=T0)
    assert decision.action == PolicyAction.BLOCK
    assert engine.is_blocked("203.0.113.9", T0)


def test_allowlisted_source_never_blocks(policy_client):
    engine = ResponseEngine(policy_client, allowlist={"10.0.0.5"})
    decision = engine.decide(_bouncer_verdict(confidence=1.0), target_ip="10.0.0.5", now=T0)
    assert decision.action == PolicyAction.ALLOWLIST_EXEMPT
    assert not engine.is_blocked("10.0.0.5", T0)


def test_second_verdict_on_same_window_does_not_reblock(policy_client):
    engine = ResponseEngine(policy_client)
    first = engine.decide(_bouncer_verdict(window_id="shared", confidence=0.9), target_ip="1.2.3.4", now=T0)
    assert first.action == PolicyAction.BLOCK

    second = engine.decide(
        _bouncer_verdict(window_id="shared", confidence=0.99), target_ip="5.6.7.8", now=T0
    )
    assert second.action == PolicyAction.DRY_RUN
    assert second.reason_codes == ["window_already_resolved"]


def test_decoy_verdict_flows_through_the_identical_path_as_bouncer(policy_client):
    engine = ResponseEngine(policy_client)
    decision = engine.decide(_decoy_verdict(), target_ip="198.51.100.7", now=T0)
    assert decision.action == PolicyAction.BLOCK
    assert engine.is_blocked("198.51.100.7", T0)


def test_circuit_breaker_throttles_after_repeated_blocks(policy_client):
    engine = ResponseEngine(policy_client, circuit_breaker=CircuitBreaker(max_actions_per_minute=2))
    engine.decide(_bouncer_verdict(window_id="a", confidence=0.9), target_ip="1.1.1.1", now=T0)
    engine.decide(_bouncer_verdict(window_id="b", confidence=0.9), target_ip="2.2.2.2", now=T0)
    third = engine.decide(_bouncer_verdict(window_id="c", confidence=0.9), target_ip="3.3.3.3", now=T0)
    assert third.action == PolicyAction.THROTTLED


def test_block_expires_after_ttl(policy_client):
    engine = ResponseEngine(policy_client)
    decision = engine.decide(_bouncer_verdict(confidence=0.95), target_ip="4.4.4.4", now=T0)
    assert engine.is_blocked("4.4.4.4", T0)
    after_ttl = T0 + timedelta(seconds=decision.ttl_seconds + 1)
    assert not engine.is_blocked("4.4.4.4", after_ttl)


def test_reverse_block_lifts_it_immediately_without_waiting_for_ttl(policy_client):
    engine = ResponseEngine(policy_client)
    engine.decide(_bouncer_verdict(confidence=0.95), target_ip="5.5.5.5", now=T0)
    assert engine.is_blocked("5.5.5.5", T0)
    assert engine.reverse_block("5.5.5.5") is True
    assert not engine.is_blocked("5.5.5.5", T0)


def test_reversing_a_nonexistent_block_returns_false(policy_client):
    engine = ResponseEngine(policy_client)
    assert engine.reverse_block("9.9.9.9") is False


def test_gray_zone_verdict_never_blocks(policy_client):
    engine = ResponseEngine(policy_client)
    decision = engine.decide(_bouncer_verdict(confidence=0.6), target_ip="6.6.6.6", now=T0)
    assert decision.action == PolicyAction.DRY_RUN
    assert not engine.is_blocked("6.6.6.6", T0)


def test_engine_raises_on_a_policy_client_that_signals_undefined(policy_client, monkeypatch):
    # Fail-closed at the engine level: if the policy layer has no defined
    # decision (see PolicyClient.evaluate's None return), the engine must
    # not silently substitute a default action — it raises, and the
    # caller (the orchestrator, once built) is responsible for logging
    # the rejection per FR-11.
    engine = ResponseEngine(policy_client)
    monkeypatch.setattr(policy_client, "evaluate", lambda *a, **k: None)
    with pytest.raises(PolicyRejectedError):
        engine.decide(_bouncer_verdict(), target_ip="7.7.7.7", now=T0)
