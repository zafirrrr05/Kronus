"""features.txt component 8: "Takes a verdict and its evidence, looks up
the matching ATT&CK technique, and writes a plain-English report —
strictly after enforcement, never before." Cold path, async via the Job
Queue (see job_queue.py) — this module is the actual explanation logic;
the queue plumbing is separate.

Runs in two modes:
  - Real: calls the Anthropic API (services/response_engine has already
    enforced by the time this runs — the LLM is never on the block-or-not
    decision path, matching "strictly after enforcement").
  - Offline: no ANTHROPIC_API_KEY configured. Produces a deterministic,
    templated narrative from the same retrieved technique instead of
    failing — the demo and every flow-case test run end to end with zero
    API cost and zero external dependency at request time, and the real
    call path is exercised identically whenever a key is present (see
    docs/instructions.md for how to enable it).
"""

from __future__ import annotations

import time

from libs.config import SETTINGS
from libs.constants import Label
from libs.observability import TOKEN_COST_USD_TOTAL, observe
from libs.schemas import DetectionVerdict, ExplanationResult
from services.llm_explainer.attck_kb import AttckTechnique, lookup_technique

# Approximate, clearly-labeled: Anthropic's published per-token pricing
# changes over time and varies by model — this constant exists so
# token_cost_usd (spec.md §3.6) is populated from something, not to be a
# billing source of truth. See docs/instructions.md.
_APPROX_USD_PER_1K_INPUT_TOKENS = 0.003
_APPROX_USD_PER_1K_OUTPUT_TOKENS = 0.015

_SYSTEM_PROMPT = (
    "You are KRONUS's incident explainer. Given a detection verdict, its "
    "evidence, and the matching MITRE ATT&CK technique, write a short, "
    "factual, plain-English incident note for a security analyst. Two to "
    "four sentences. State what was observed, why it matched this "
    "technique, and what evidence supports it. Do not speculate beyond "
    "the evidence given, and do not recommend further action — a human "
    "reviewer decides that."
)


def _offline_narrative(verdict: DetectionVerdict, technique: AttckTechnique | None) -> str:
    if technique is None:
        return (
            f"Window {verdict.window_id}: the {verdict.tier} tier reported "
            f"'{verdict.label}' at {verdict.confidence:.0%} confidence. No "
            f"specific technique is attributed at this confidence level."
        )
    evidence_note = (
        f"Evidence: {len(verdict.evidence.node_ids)} host(s), "
        f"{len(verdict.evidence.edge_ids)} flow(s)."
        if verdict.evidence.node_ids or verdict.evidence.edge_ids
        else "Evidence: direct signal, no graph attribution required for this tier."
    )
    return (
        f"Window {verdict.window_id}: the {verdict.tier} tier flagged "
        f"'{verdict.label}' at {verdict.confidence:.0%} confidence, matching "
        f"MITRE ATT&CK {technique.technique_id} ({technique.name}, "
        f"{technique.tactic}). {technique.summary} {evidence_note}"
    )


def _build_prompt(verdict: DetectionVerdict, technique: AttckTechnique | None) -> str:
    technique_block = (
        f"ATT&CK technique: {technique.technique_id} ({technique.name}, tactic: "
        f"{technique.tactic}). {technique.summary}"
        if technique is not None
        else "No ATT&CK technique attributed at this confidence level."
    )
    return (
        f"Detection tier: {verdict.tier}\n"
        f"Label: {verdict.label}\n"
        f"Confidence: {verdict.confidence:.2f}\n"
        f"Evidence node count: {len(verdict.evidence.node_ids)}\n"
        f"Evidence edge count: {len(verdict.evidence.edge_ids)}\n"
        f"Attribution method: {verdict.evidence.attribution_method}\n"
        f"{technique_block}"
    )


class LLMExplainer:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key if api_key is not None else SETTINGS.anthropic_api_key
        self._model = model or SETTINGS.anthropic_model

    @property
    def is_online(self) -> bool:
        return bool(self._api_key)

    def explain(self, verdict: DetectionVerdict, job_id: str) -> ExplanationResult:
        with observe("llm_explainer", "explain", tier=verdict.tier, label=verdict.label):
            technique = lookup_technique(_as_label(verdict.label))
            start = time.perf_counter()

            if self.is_online:
                narrative, cost_usd = self._call_anthropic(verdict, technique)
            else:
                narrative = _offline_narrative(verdict, technique)
                cost_usd = 0.0

            latency_ms = int((time.perf_counter() - start) * 1000)
            TOKEN_COST_USD_TOTAL.labels(component="llm_explainer").inc(cost_usd)

            return ExplanationResult(
                job_id=job_id,
                verdict_id=verdict.verdict_id,
                narrative=narrative,
                attck_technique_id=technique.technique_id if technique else "none",
                retrieved_sources=[f"MITRE ATT&CK {technique.technique_id}"] if technique else [],
                latency_ms=latency_ms,
                token_cost_usd=cost_usd,
            )

    def _call_anthropic(self, verdict: DetectionVerdict, technique: AttckTechnique | None) -> tuple[str, float]:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=300,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_prompt(verdict, technique)}],
        )
        narrative = "".join(block.text for block in response.content if block.type == "text")
        cost_usd = (
            response.usage.input_tokens / 1000 * _APPROX_USD_PER_1K_INPUT_TOKENS
            + response.usage.output_tokens / 1000 * _APPROX_USD_PER_1K_OUTPUT_TOKENS
        )
        return narrative, cost_usd


def _as_label(label) -> Label:
    # verdict.label is a plain str at runtime (KronusModel's
    # use_enum_values=True) — normalize back to the enum for the corpus
    # lookup's dict key.
    return label if isinstance(label, Label) else Label(label)
