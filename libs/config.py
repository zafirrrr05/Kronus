"""One place every service reads runtime config from. Deliberately a plain
dataclass over os.environ rather than a settings framework — a dozen
env-backed fields don't earn a new dependency (ponytail rung 5: an
already-installed one, python-dotenv, covers loading the .env file; stdlib
covers reading it).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Loaded once at import time. Real secrets (ANTHROPIC_API_KEY, DB password)
# come from the environment in every deployment target that matters (k8s
# Secret, CI secret store); .env is a local-dev convenience only and is
# git-ignored — see SR-2 in spec.md and docs/setup.md.
load_dotenv(REPO_ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    return default if val is None else val.strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    # Ingestion Pipeline / Event Stream
    event_bus_backend: str = field(
        default_factory=lambda: os.environ.get("KRONUS_EVENT_BUS_BACKEND", "memory")
    )
    redpanda_bootstrap_servers: str = field(
        default_factory=lambda: os.environ.get("KRONUS_REDPANDA_BOOTSTRAP", "localhost:9092")
    )

    # Logbook
    logbook_backend: str = field(
        default_factory=lambda: os.environ.get("KRONUS_LOGBOOK_BACKEND", "sqlite")
    )
    postgres_dsn: str = field(
        default_factory=lambda: os.environ.get(
            "KRONUS_POSTGRES_DSN",
            "dbname=kronus user=kronus password=kronus_dev_local host=localhost",
        )
    )
    sqlite_path: str = field(
        default_factory=lambda: os.environ.get("KRONUS_SQLITE_PATH", "data/logbook.db")
    )

    # Response Engine
    opa_binary_path: str = field(
        default_factory=lambda: os.environ.get("KRONUS_OPA_BINARY", "opa")
    )
    policy_bundle_dir: str = field(
        default_factory=lambda: os.environ.get("KRONUS_POLICY_DIR", "policy")
    )
    dry_run_mode: bool = field(default_factory=lambda: _bool("KRONUS_DRY_RUN_MODE", False))

    # LLM Explainer
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.environ.get("KRONUS_ANTHROPIC_MODEL", "claude-sonnet-5")
    )

    # Observability
    metrics_port: int = field(
        default_factory=lambda: int(os.environ.get("KRONUS_METRICS_PORT", "9100"))
    )

    # Decoy
    decoy_ssh_port: int = field(
        default_factory=lambda: int(os.environ.get("KRONUS_DECOY_SSH_PORT", "2222"))
    )
    real_ssh_relocated_port: int = field(
        default_factory=lambda: int(os.environ.get("KRONUS_REAL_SSH_PORT", "2022"))
    )


SETTINGS = Settings()
