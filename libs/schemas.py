"""spec.md §3 — component interfaces, verbatim.

"Every producer validates on write; every consumer validates on read. A
message that fails validation is rejected, logged, and never silently
coerced." These eight models are that validation boundary. No service
redefines its own copy — they import from here.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from libs.constants import (
    AttributionMethod,
    AuditEntryType,
    CorrectionVerdict,
    DriftAction,
    DriftMetric,
    Label,
    PolicyAction,
    Protocol,
    Sensor,
    Tier,
)


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class KronusModel(BaseModel):
    """Base for every wire schema: reject unknown fields (spec.md §3's
    'never silently coerced') and freeze once constructed — these are
    messages on a bus, not objects a consumer should be able to mutate
    out from under a producer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)


# ---------------------------------------------------------------------------
# 3.1 Telemetry Event (sensor -> Event Stream)
# ---------------------------------------------------------------------------


class HostEvent(KronusModel):
    """Present only for Falco/eBPF-sourced records; network-only sensors
    omit the parent field entirely rather than sending this with nulls.
    """

    process: str | None = None
    syscall: str | None = None


class HoneypotDetail(KronusModel):
    """Present only when sensor == honeypot. Every field is optional since
    a bare connection attempt with no further interaction is still a valid,
    if sparse, record (spec.md §3.1).
    """

    decoy_service: str | None = None
    credentials_tried: list[str] = Field(default_factory=list)
    commands_typed: list[str] = Field(default_factory=list)
    session_duration_ms: int | None = None


class TelemetryEvent(KronusModel):
    event_id: str = Field(default_factory=_uuid)
    ts: datetime = Field(default_factory=_now)
    source_ip: str
    dest_ip: str
    source_port: int | None = None
    dest_port: int | None = None
    protocol: Protocol
    bytes: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    sensor: Sensor
    host_event: HostEvent | None = None
    honeypot_detail: HoneypotDetail | None = None

    @model_validator(mode="after")
    def _honeypot_detail_matches_sensor(self) -> "TelemetryEvent":
        # honeypot_detail is meaningful only when sensor == honeypot (§3.1).
        # A honeypot-tagged event with no detail is still valid (a bare scan
        # touch); a non-honeypot event carrying honeypot_detail is not.
        if self.honeypot_detail is not None and self.sensor != Sensor.HONEYPOT:
            raise ValueError("honeypot_detail is only valid when sensor == 'honeypot'")
        return self


# ---------------------------------------------------------------------------
# 3.2 Graph Snapshot (Graph Builder -> Detective, deep lane only)
# ---------------------------------------------------------------------------


class GraphNode(KronusModel):
    node_id: str
    degree_in: float = 0.0
    degree_out: float = 0.0
    bytes_total: float = 0.0
    unique_ports_contacted: int = 0


class GraphEdge(KronusModel):
    src: str
    dst: str
    bytes: float = 0.0
    flow_count: int = 0
    port_entropy: float = 0.0
    duration_mean_ms: float = 0.0


class GraphSnapshot(KronusModel):
    snapshot_id: str = Field(default_factory=_uuid)
    window_id: str
    window_start: datetime
    window_end: datetime
    nodes: list[GraphNode]
    edges: list[GraphEdge]


# ---------------------------------------------------------------------------
# 3.3 Detection Verdict (Bouncer, Detective, or Decoy -> Response Engine)
# ---------------------------------------------------------------------------


class VerdictEvidence(KronusModel):
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    attribution_method: AttributionMethod


class DetectionVerdict(KronusModel):
    verdict_id: str = Field(default_factory=_uuid)
    window_id: str
    tier: Tier
    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: VerdictEvidence

    @model_validator(mode="after")
    def _decoy_confidence_is_fixed(self) -> "DetectionVerdict":
        # spec.md §3.3: "confidence is fixed at 1.0 by policy, not computed"
        # for the decoy tier — enforced here so no caller can accidentally
        # emit a "calibrated-looking" decoy verdict.
        if self.tier == Tier.DECOY and self.confidence != 1.0:
            raise ValueError("decoy verdicts must have confidence fixed at 1.0")
        if self.tier in (Tier.BOUNCER, Tier.DECOY) and (
            self.evidence.node_ids or self.evidence.edge_ids
        ):
            raise ValueError(f"{self.tier} verdicts carry no graph evidence (spec.md §3.3)")
        return self


# ---------------------------------------------------------------------------
# 3.4 Policy Decision (OPA output, Response Engine input)
# ---------------------------------------------------------------------------

ReasonCode = Literal[
    "confidence_above_threshold",
    "on_allowlist",
    "gray_zone",
    "circuit_breaker_exceeded",
    "below_gray_zone",
    "window_already_resolved",
]


class PolicyDecision(KronusModel):
    decision_id: str = Field(default_factory=_uuid)
    verdict_id: str
    action: PolicyAction
    reason_codes: list[ReasonCode]
    ttl_seconds: int | None = None
    policy_version: str

    @model_validator(mode="after")
    def _ttl_only_for_block(self) -> "PolicyDecision":
        # Pydantic v2 gotcha, caught for real during KRONUS's own build:
        # a @field_validator on ttl_seconds does NOT run when the caller
        # omits the field (it never touches the default) unless the field
        # sets validate_default=True. A model_validator(mode="after") runs
        # once every field is set, default or not, which is what a
        # cross-field invariant like this needs.
        if self.action != PolicyAction.BLOCK and self.ttl_seconds is not None:
            raise ValueError("ttl_seconds only applies to 'block' decisions")
        return self


# ---------------------------------------------------------------------------
# 3.5 Audit Log Entry (Logbook, append-only, hash-chained)
# ---------------------------------------------------------------------------

GENESIS_HASH = "0" * 64  # fixed seed for the first entry in any chain


class AuditLogEntry(KronusModel):
    entry_id: str = Field(default_factory=_uuid)
    prev_hash: str
    entry_hash: str = ""  # computed below, never supplied by the caller
    ts: datetime = Field(default_factory=_now)
    entry_type: AuditEntryType
    payload: dict[str, Any]
    actor: str = "system"

    @model_validator(mode="after")
    def _compute_hash_and_validate_actor(self) -> "AuditLogEntry":
        if self.actor != "system" and not self.actor.startswith("operator:"):
            raise ValueError("actor must be 'system' or 'operator:<id>'")
        if not self.entry_hash:
            canonical = json.dumps(
                {
                    "entry_id": self.entry_id,
                    "prev_hash": self.prev_hash,
                    "ts": self.ts.isoformat(),
                    "entry_type": self.entry_type,
                    "payload": self.payload,
                    "actor": self.actor,
                },
                sort_keys=True,
                default=str,
            )
            object.__setattr__(
                self, "entry_hash", hashlib.sha256(canonical.encode()).hexdigest()
            )
        return self


# ---------------------------------------------------------------------------
# 3.6 Explanation Job / Result (Job Queue <-> LLM Explainer)
# ---------------------------------------------------------------------------


class ExplanationJob(KronusModel):
    job_id: str = Field(default_factory=_uuid)
    verdict_id: str
    attempt: int = Field(default=1, ge=1)
    enqueued_at: datetime = Field(default_factory=_now)


class ExplanationResult(KronusModel):
    job_id: str
    verdict_id: str
    narrative: str
    attck_technique_id: str
    retrieved_sources: list[str]
    generated_at: datetime = Field(default_factory=_now)
    latency_ms: int = Field(ge=0)
    token_cost_usd: float = Field(ge=0.0)


# ---------------------------------------------------------------------------
# 3.7 Correction (human review -> Drift Watcher training set)
# ---------------------------------------------------------------------------


class Correction(KronusModel):
    correction_id: str = Field(default_factory=_uuid)
    decision_id: str
    operator: str
    verdict_was: CorrectionVerdict
    note: str | None = None
    ts: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# 3.8 Drift Report (Drift Watcher -> Auto-Retrainer trigger)
# ---------------------------------------------------------------------------


class FeatureDrift(KronusModel):
    metric: DriftMetric
    per_feature_scores: dict[str, float]
    threshold_breached: bool


class LiveAccuracy(KronusModel):
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    vs_baseline_delta: float


class DriftReport(KronusModel):
    report_id: str = Field(default_factory=_uuid)
    window: str  # "ISO-8601/ISO-8601" per spec.md §3.8
    feature_drift: FeatureDrift
    live_accuracy: LiveAccuracy
    corrections_since_last_cycle: int = Field(ge=0)
    decoy_sessions_since_last_cycle: int = Field(ge=0)
    action: DriftAction
