"""Standalone entrypoint for running the API as a real service (what
infra/docker-compose.yml's `api` service and the Helm chart's
response-engine Deployment both run). Builds a real KronusSystem using
whatever backends libs/config.py resolves from the environment — memory/
sqlite for local dev, redpanda/postgres when those env vars point at real
infrastructure (see docs/setup.md).

Run: python -m api.serve
"""

from __future__ import annotations

import asyncio

import numpy as np
import uvicorn

from libs.config import SETTINGS
from libs.event_bus import EventBus, InMemoryEventBus, RedpandaEventBus
from libs.observability import configure_tracing, start_metrics_server
from pipeline.kronus_system import KronusSystem
from services.bouncer.model import BouncerModel
from services.detective.model import DetectiveModel
from services.drift_watcher.retrainer import DriftWatcher
from services.llm_explainer.explainer import LLMExplainer
from services.logbook.base import LogbookStore
from services.logbook.postgres_store import PostgresLogbookStore
from services.logbook.sqlite_store import SQLiteLogbookStore
from services.response_engine.engine import ResponseEngine
from services.response_engine.opa_client import PolicyClient


def _build_event_bus() -> EventBus:
    if SETTINGS.event_bus_backend == "redpanda":
        return RedpandaEventBus(SETTINGS.redpanda_bootstrap_servers)
    return InMemoryEventBus()


def _build_logbook() -> LogbookStore:
    if SETTINGS.logbook_backend == "postgres":
        return PostgresLogbookStore(SETTINGS.postgres_dsn)
    return SQLiteLogbookStore(SETTINGS.sqlite_path)


async def _build_system() -> KronusSystem:
    system = KronusSystem(
        bus=_build_event_bus(),
        bouncer=BouncerModel.load("models/registry/bouncer"),
        detective=DetectiveModel.load("models/registry/detective"),
        response_engine=ResponseEngine(
            PolicyClient(
                binary_path=SETTINGS.opa_binary_path, policy_dir=SETTINGS.policy_bundle_dir
            )
        ),
        logbook=_build_logbook(),
        explainer=LLMExplainer(),
        drift_watcher=DriftWatcher(
            baseline_features={"f": np.random.default_rng(0).normal(size=200)}
        ),
    )
    await system.start()
    return system


def main() -> None:
    import api.main as api_module

    configure_tracing()
    start_metrics_server(SETTINGS.metrics_port)

    system = asyncio.run(_build_system())
    api_module.bind_system(system)

    uvicorn.run(api_module.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
