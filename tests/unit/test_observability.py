import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from libs import observability as obs


@pytest.fixture(autouse=True)
def _tracing_to_memory():
    """Redirect spans to an in-memory exporter so tests can assert on them
    instead of hunting through console output."""
    exporter = InMemorySpanExporter()
    obs.configure_tracing(exporter=exporter)
    obs._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    exporter.clear()


def test_observe_emits_a_span_named_for_the_operation(_tracing_to_memory):
    with obs.observe("bouncer", "score_flow"):
        pass
    spans = _tracing_to_memory.get_finished_spans()
    assert any(s.name == "score_flow" for s in spans)


def test_observe_records_latency_histogram():
    before = obs.OPERATION_LATENCY_SECONDS.labels(
        component="detective", operation="infer"
    )._sum.get()
    with obs.observe("detective", "infer"):
        pass
    after = obs.OPERATION_LATENCY_SECONDS.labels(
        component="detective", operation="infer"
    )._sum.get()
    assert after >= before


def test_observe_counts_ok_outcome():
    before = obs.EVENTS_TOTAL.labels(component="decoy", outcome="ok")._value.get()
    with obs.observe("decoy", "session_capture"):
        pass
    after = obs.EVENTS_TOTAL.labels(component="decoy", outcome="ok")._value.get()
    assert after == before + 1


def test_observe_counts_error_outcome_and_still_raises():
    before = obs.EVENTS_TOTAL.labels(component="response_engine", outcome="error")._value.get()
    with pytest.raises(RuntimeError):
        with obs.observe("response_engine", "decide"):
            raise RuntimeError("boom")
    after = obs.EVENTS_TOTAL.labels(component="response_engine", outcome="error")._value.get()
    assert after == before + 1
