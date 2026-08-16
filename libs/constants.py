"""Fixed vocabularies shared by every service.

These enums mirror spec.md §3's schema definitions exactly. They exist here,
once, because `tier` and `label` values are compared across bouncer,
detective, decoy, response_engine, and drift_watcher — defining them in one
place is what makes "the same verdict shape, uniformly policed" (FR-12,
FR-13) actually enforceable in code, not just asserted in docs.
"""

from __future__ import annotations

from enum import Enum


class Protocol(str, Enum):
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    OTHER = "other"


class Sensor(str, Enum):
    """spec.md §3.1 — which sensor produced a TelemetryEvent."""

    ZEEK = "zeek"
    SOFTFLOWD = "softflowd"
    FALCO = "falco"
    HONEYPOT = "honeypot"


class Tier(str, Enum):
    """spec.md §3.3 — the three verdict sources the Response Engine accepts."""

    BOUNCER = "bouncer"
    DETECTIVE = "detective"
    DECOY = "decoy"


class Label(str, Enum):
    """spec.md §3.3 — the fixed verdict label set. Do not add values here;
    a label the Response Engine's policy table (spec.md §6) doesn't know
    about would fail closed (see policy/response.rego), by design.
    """

    FLOOD = "flood"
    PORT_SCAN = "port_scan"
    LATERAL_MOVEMENT = "lateral_movement"
    BENIGN = "benign"
    UNCERTAIN = "uncertain"
    DECOY_INTERACTION = "decoy_interaction"


class AttributionMethod(str, Enum):
    GNN_EXPLAINER = "gnn_explainer"
    ATTENTION_ROLLOUT = "attention_rollout"
    RATE_THRESHOLD = "rate_threshold"
    HONEYPOT_INTERACTION = "honeypot_interaction"


class PolicyAction(str, Enum):
    """spec.md FR-6 — the four response modes."""

    BLOCK = "block"
    DRY_RUN = "dry_run"
    ALLOWLIST_EXEMPT = "allowlist_exempt"
    THROTTLED = "throttled"


class AuditEntryType(str, Enum):
    DETECTION = "detection"
    EXPLANATION = "explanation"
    POLICY_DECISION = "policy_decision"
    ENFORCEMENT = "enforcement"
    CORRECTION = "correction"


class CorrectionVerdict(str, Enum):
    FALSE_POSITIVE = "false_positive"
    TRUE_POSITIVE = "true_positive"
    FALSE_NEGATIVE = "false_negative"


class DriftMetric(str, Enum):
    PSI = "PSI"
    KS = "KS"


class DriftAction(str, Enum):
    NONE = "none"
    RETRAIN_TRIGGERED = "retrain_triggered"


class DataOrigin(str, Enum):
    """Not part of the wire schema (spec.md §3 is exact and fixed) — this is
    an orchestration-layer tag so callers (twin/, telemetry_exporter/,
    drift_watcher/) can enforce "demo and tests run on real data only;
    synthetic is for the Drift Watcher's improvement loop" without touching
    TelemetryEvent's shape.
    """

    REAL = "real"
    SYNTHETIC = "synthetic"
    DECOY_CAPTURED = "decoy_captured"


# spec.md §2.2 — starting SLO targets, used as policy/model defaults.
# Re-recorded as *measured* values once the load harness / eval suite runs;
# see docs produced at the end of the build for the measured numbers.
BLOCK_CONFIDENCE_THRESHOLD = 0.85
GRAY_ZONE_LOW_THRESHOLD = 0.50
AUTO_BLOCK_TTL_SECONDS = 15 * 60  # NFR-9: <=15 min TTL
CIRCUIT_BREAKER_MAX_ACTIONS_PER_MIN = 20  # NFR-10
SNAPSHOT_WINDOW_SECONDS = 2.0  # spec.md §4 default tumbling window
