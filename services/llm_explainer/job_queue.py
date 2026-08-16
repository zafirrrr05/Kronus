"""features.txt architecture diagram: "EXPLAIN — async, Job Queue ->
LLM Explainer." The Job Queue is infrastructure, not a new technology
choice of its own — spec.md's own repo philosophy treats Event Stream as
shared, configured infra (see libs/event_bus.py), and a job queue is the
same kind of thing: an ordered, durable channel with one consumer group.
Reusing EventBus for it (a second topic, not a second broker) is the
"reuse in this codebase" rung of the dependency ladder actually firing —
a job queue and an event topic are the same primitive here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from libs.event_bus import EventBus
from libs.schemas import ExplanationJob

JOB_QUEUE_TOPIC = "explanation-jobs"
JOB_QUEUE_CONSUMER_GROUP = "llm-explainer-workers"


async def enqueue(bus: EventBus, job: ExplanationJob) -> None:
    await bus.publish(JOB_QUEUE_TOPIC, key=job.verdict_id, value=job.model_dump_json().encode())


async def consume(bus: EventBus) -> AsyncIterator[ExplanationJob]:
    async for raw in bus.subscribe(JOB_QUEUE_TOPIC, JOB_QUEUE_CONSUMER_GROUP):
        yield ExplanationJob.model_validate_json(raw)
