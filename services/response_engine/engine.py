"""features.txt component 6: "The convergence point. Upstream: all three
verdict sources — Bouncer, Detective, Decoy." This is that convergence:
every verdict from every tier passes through the identical decide() path
(FR-13 — no special-case bypass for the Decoy), which consults the
correlator (FR-12), the circuit breaker (NFR-10), and OPA (FR-9), then
tracks the resulting block for its TTL (NFR-9) so it can be looked up or
reversed later (Case 5, the false-positive correction flow).

target_ip is not part of DetectionVerdict (spec.md §3.3 has no such
field — enforcement targeting is this engine's own operational concern,
not a cross-component wire contract) so it's supplied explicitly by
whichever caller produced the verdict: the Bouncer/Decoy already know the
flow's source_ip directly; the Detective's evidence.node_ids names the
suspect host(s) and the orchestrator picks the primary one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from libs.constants import PolicyAction
from libs.observability import ACTIVE_GAUGE, observe
from libs.schemas import DetectionVerdict, PolicyDecision
from services.response_engine.circuit_breaker import CircuitBreaker
from services.response_engine.correlator import WindowCorrelator
from services.response_engine.opa_client import PolicyClient


class PolicyRejectedError(Exception):
    """The policy had no defined decision for this input (opa's `decision`
    rule stayed undefined — malformed verdict). No automatic action is
    taken; the caller is responsible for recording the rejection (FR-11:
    every detection is logged, including ones the policy layer refused).
    """

    def __init__(self, verdict: DetectionVerdict) -> None:
        self.verdict = verdict
        super().__init__(f"policy has no defined decision for verdict {verdict.verdict_id}")


@dataclass
class ActiveBlock:
    target_ip: str
    decision_id: str
    expires_at: datetime
    reversed: bool = False


class ResponseEngine:
    def __init__(
        self, policy_client: PolicyClient, allowlist: set[str] | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._policy_client = policy_client
        self._correlator = WindowCorrelator()
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._allowlist = set(allowlist or ())
        self._active_blocks: dict[str, ActiveBlock] = {}  # target_ip -> block

    def decide(self, verdict: DetectionVerdict, target_ip: str, now: datetime) -> PolicyDecision:
        with observe("response_engine", "decide", tier=verdict.tier, label=verdict.label):
            on_allowlist = target_ip in self._allowlist
            breaker_tripped = self._circuit_breaker.is_tripped(now)
            prior_action = self._correlator.prior_action_for(verdict.window_id)

            decision = self._policy_client.evaluate(verdict, on_allowlist, breaker_tripped, prior_action)
            if decision is None:
                raise PolicyRejectedError(verdict)

            self._correlator.record(verdict.window_id, decision.action)

            if decision.action == PolicyAction.BLOCK:
                self._circuit_breaker.record_action(now)
                self._active_blocks[target_ip] = ActiveBlock(
                    target_ip=target_ip, decision_id=decision.decision_id,
                    expires_at=now + timedelta(seconds=decision.ttl_seconds or 0),
                )
                ACTIVE_GAUGE.labels(component="response_engine", kind="active_blocks").set(
                    len(self._active_blocks)
                )

            return decision

    def is_blocked(self, target_ip: str, now: datetime) -> bool:
        block = self._active_blocks.get(target_ip)
        if block is None or block.reversed:
            return False
        return now < block.expires_at

    def reverse_block(self, target_ip: str) -> bool:
        """Case 5: a human operator marks an auto-block a false positive.
        Reverses immediately, without waiting for the TTL."""
        block = self._active_blocks.get(target_ip)
        if block is None or block.reversed:
            return False
        block.reversed = True
        return True

    def allowlist_add(self, target_ip: str) -> None:
        self._allowlist.add(target_ip)
