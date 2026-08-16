"""The Observability Stack's code-side half (features.txt connectivity map,
component 10): "not a stop on the line — a hub, ten spokes." Every one of
the other ten components calls `observe()` (or the lower-level primitives
directly) around its real work. That call *is* the edge to this hub — there
is no separate "observability service" to stand up locally; Prometheus/
Grafana (the dashboards) are configured infrastructure that scrapes what
this module exposes (see infra/grafana and infra/docker/prometheus.yml).

Design note: one metric family per *shape* of measurement, parameterized by
a `component` label, rather than one Counter/Histogram per component. Real
Prometheus practice — and it's what makes "every component reports here"
a small, reusable surface instead of ten near-identical metric blocks.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

REGISTRY = CollectorRegistry()

EVENTS_TOTAL = Counter(
    "kronus_events_total",
    "Units of work processed, by component and outcome.",
    ["component", "outcome"],
    registry=REGISTRY,
)

OPERATION_LATENCY_SECONDS = Histogram(
    "kronus_operation_latency_seconds",
    "Wall-clock latency of a named operation within a component.",
    ["component", "operation"],
    registry=REGISTRY,
)

ACTIVE_GAUGE = Gauge(
    "kronus_active",
    "Current level of a named quantity (active blocks, consumer lag, etc).",
    ["component", "kind"],
    registry=REGISTRY,
)

# spec.md §10 "cost-per-detection" panel: LLM Explainer is the only
# component that spends real money per call.
TOKEN_COST_USD_TOTAL = Counter(
    "kronus_token_cost_usd_total",
    "Cumulative LLM Explainer spend in USD.",
    ["component"],
    registry=REGISTRY,
)

_tracer_provider: TracerProvider | None = None


def configure_tracing(exporter: SpanExporter | None = None) -> None:
    """Call once at process start. Defaults to a console exporter — good
    enough for a laptop demo; swap in an OTLP exporter for a real cluster
    by passing one in (see infra/helm values for the collector endpoint).
    """
    global _tracer_provider
    resource = Resource.create({SERVICE_NAME: "kronus"})
    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(BatchSpanProcessor(exporter or ConsoleSpanExporter()))
    trace.set_tracer_provider(_tracer_provider)


def get_tracer(component: str) -> trace.Tracer:
    return trace.get_tracer(component)


@contextmanager
def observe(component: str, operation: str, **span_attributes: str) -> Iterator[trace.Span]:
    """One call wraps a unit of work with: a span (trace ID propagates
    automatically via OTel context, satisfying spec.md §8's "single trace
    ID from ingestion through enforcement"), a latency histogram
    observation, and an outcome counter. This is the single call site every
    service uses — it's what makes 10/10 components reaching Observability
    checkable by grep, not just asserted.
    """
    tracer = get_tracer(component)
    start = time.perf_counter()
    outcome = "ok"
    with tracer.start_as_current_span(operation, attributes=span_attributes) as span:
        try:
            yield span
        except Exception:
            outcome = "error"
            raise
        finally:
            OPERATION_LATENCY_SECONDS.labels(component=component, operation=operation).observe(
                time.perf_counter() - start
            )
            EVENTS_TOTAL.labels(component=component, outcome=outcome).inc()


def start_metrics_server(port: int = 9100) -> None:
    """Exposes /metrics for Prometheus to scrape (infra/docker/prometheus.yml
    already points at this port for each service in docker-compose)."""
    start_http_server(port, registry=REGISTRY)
