import asyncio

import pytest

from libs.constants import AttributionMethod, Label, Tier
from libs.event_bus import InMemoryEventBus
from libs.schemas import DetectionVerdict, ExplanationJob, VerdictEvidence
from services.llm_explainer.attck_kb import ATTCK_CORPUS, lookup_technique
from services.llm_explainer.explainer import LLMExplainer
from services.llm_explainer.job_queue import consume, enqueue


def _verdict(label=Label.PORT_SCAN, confidence=0.9, tier=Tier.DETECTIVE) -> DetectionVerdict:
    evidence = (
        VerdictEvidence(node_ids=["h1", "h2"], edge_ids=["h1->h2"], attribution_method=AttributionMethod.ATTENTION_ROLLOUT)
        if tier == Tier.DETECTIVE
        else VerdictEvidence(attribution_method=AttributionMethod.RATE_THRESHOLD)
    )
    return DetectionVerdict(window_id="w1", tier=tier, label=label, confidence=confidence, evidence=evidence)


# --- MITRE corpus -----------------------------------------------------------

def test_every_attack_label_has_a_technique():
    for label in (Label.FLOOD, Label.PORT_SCAN, Label.LATERAL_MOVEMENT, Label.DECOY_INTERACTION):
        assert lookup_technique(label) is not None


def test_benign_and_uncertain_have_no_technique():
    assert lookup_technique(Label.BENIGN) is None
    assert lookup_technique(Label.UNCERTAIN) is None


def test_technique_ids_are_well_formed():
    for technique in ATTCK_CORPUS.values():
        assert technique.technique_id.startswith("T")
        assert technique.technique_id[1:].isdigit()


# --- Explainer, offline mode (what demo/tests actually run) -----------------

def test_explainer_defaults_to_offline_without_api_key():
    explainer = LLMExplainer(api_key=None)
    assert explainer.is_online is False


def test_offline_explanation_costs_nothing():
    explainer = LLMExplainer(api_key=None)
    result = explainer.explain(_verdict(), job_id="j1")
    assert result.token_cost_usd == 0.0


def test_offline_explanation_attributes_the_correct_technique():
    explainer = LLMExplainer(api_key=None)
    result = explainer.explain(_verdict(label=Label.PORT_SCAN), job_id="j1")
    assert result.attck_technique_id == "T1046"
    assert "T1046" in result.retrieved_sources[0]


def test_offline_explanation_for_decoy_attributes_brute_force():
    explainer = LLMExplainer(api_key=None)
    verdict = _verdict(label=Label.DECOY_INTERACTION, confidence=1.0, tier=Tier.DECOY)
    result = explainer.explain(verdict, job_id="j1")
    assert result.attck_technique_id == "T1110"


def test_offline_explanation_narrative_mentions_the_window_and_label():
    explainer = LLMExplainer(api_key=None)
    result = explainer.explain(_verdict(), job_id="j1")
    assert "w1" in result.narrative
    assert "port_scan" in result.narrative


def test_uncertain_verdicts_get_a_narrative_with_no_technique():
    explainer = LLMExplainer(api_key=None)
    verdict = _verdict(label=Label.UNCERTAIN, confidence=0.6)
    result = explainer.explain(verdict, job_id="j1")
    assert result.attck_technique_id == "none"
    assert result.retrieved_sources == []


def test_explanation_result_is_schema_valid():
    explainer = LLMExplainer(api_key=None)
    result = explainer.explain(_verdict(), job_id="j1")
    assert result.latency_ms >= 0
    assert result.job_id == "j1"


# --- Job queue (reuses EventBus, not a new dependency) -----------------------

@pytest.mark.asyncio
async def test_job_queue_round_trips_through_the_event_bus():
    bus = InMemoryEventBus()
    received: list[ExplanationJob] = []

    async def worker():
        async for job in consume(bus):
            received.append(job)
            return

    task = asyncio.create_task(worker())
    await asyncio.sleep(0.01)

    await enqueue(bus, ExplanationJob(verdict_id="v-42"))
    await asyncio.wait_for(task, timeout=1)

    assert received[0].verdict_id == "v-42"
