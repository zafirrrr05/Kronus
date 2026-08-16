"""features.txt component 2 (Ingestion Pipeline): "sensors that turn raw
traffic and host activity into small structured events." This module is
that translation step — pure functions, no I/O, easy to test in isolation
from the event bus they eventually get published onto (see exporter.py).

Deliberately origin-agnostic: whether a row came from twin.nsl_kdd (real)
or twin.synthetic (synthetic), the conversion logic is identical — that's
what spec.md §5.4's "no feature may derive from a twin-specific artifact"
requires structurally, not just by policy. Which loader a caller used is
the caller's concern (demo/tests use twin.nsl_kdd only; drift_watcher's
improvement loop may use either) — TelemetryEvent's wire shape carries no
origin tag by design (libs/schemas is spec.md §3, exact, no extra fields).
"""

from __future__ import annotations

from libs.constants import Protocol, Sensor
from libs.schemas import HoneypotDetail, TelemetryEvent
from twin.nsl_kdd import NSLKDDRow


def flow_row_to_event(row: NSLKDDRow, ts=None) -> TelemetryEvent:
    """Real (twin.nsl_kdd) or synthetic (twin.synthetic) flow rows share
    the NSLKDDRow shape (see twin/synthetic.py's docstring for why), so one
    function covers both.

    `ts` defaults to "now" (TelemetryEvent's own default) for a genuinely
    live sensor. Replaying a historical file needs explicit, increasing
    timestamps instead — every row converted in a tight loop would
    otherwise land in the same instant, making the Graph Builder's ~2s
    windowing meaningless (see services/graph_builder/windowing and
    services/detective/train.py, which passes ts explicitly for exactly
    this reason).
    """
    kwargs = dict(
        source_ip=row.source_ip,
        dest_ip=row.dest_ip,
        source_port=row.source_port,
        dest_port=row.dest_port,
        protocol=row.protocol,
        bytes=max(row.total_bytes, 0),
        duration_ms=max(row.duration_ms, 0),
        sensor=Sensor.ZEEK,
    )
    if ts is not None:
        kwargs["ts"] = ts
    return TelemetryEvent(**kwargs)


def decoy_session_to_event(
    source_ip: str,
    dest_ip: str,
    dest_port: int,
    decoy_service: str,
    credentials_tried: list[str],
    commands_typed: list[str],
    session_duration_ms: int,
) -> TelemetryEvent:
    """spec.md §3.1: honeypot_detail is populated only for sensor=='honeypot'
    events — enforced again here at construction (belt-and-suspenders with
    the schema's own model_validator).
    """
    return TelemetryEvent(
        source_ip=source_ip,
        dest_ip=dest_ip,
        dest_port=dest_port,
        protocol=Protocol.TCP,
        bytes=0,
        duration_ms=session_duration_ms,
        sensor=Sensor.HONEYPOT,
        honeypot_detail=HoneypotDetail(
            decoy_service=decoy_service,
            credentials_tried=credentials_tried,
            commands_typed=commands_typed,
            session_duration_ms=session_duration_ms,
        ),
    )
