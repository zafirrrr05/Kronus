"""Production backend for the Logbook (spec.md: "table stakes for
anything aimed at a regulated industry like banking"). Uses psycopg2
(already a dependency; asyncpg would be a second Postgres driver for no
real gain here) wrapped with asyncio.to_thread so the public interface
stays honestly async without needing a second driver — this class is
called once per policy decision, not at a rate where the thread-pool
hop matters.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import psycopg2
import psycopg2.extras

from libs.schemas import GENESIS_HASH, AuditLogEntry
from services.logbook.base import LogbookStore

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class PostgresLogbookStore(LogbookStore):
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA_PATH.read_text())

    async def _get_latest_hash(self) -> str:
        return await asyncio.to_thread(self._get_latest_hash_sync)

    def _get_latest_hash_sync(self) -> str:
        with self._conn.cursor() as cur:
            cur.execute("SELECT entry_hash FROM logbook ORDER BY seq DESC LIMIT 1")
            row = cur.fetchone()
            return row[0] if row else GENESIS_HASH

    async def _persist(self, entry: AuditLogEntry) -> None:
        await asyncio.to_thread(self._persist_sync, entry)

    def _persist_sync(self, entry: AuditLogEntry) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO logbook (entry_id, prev_hash, entry_hash, ts, entry_type, payload, actor) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (entry.entry_id, entry.prev_hash, entry.entry_hash, entry.ts, entry.entry_type,
                 psycopg2.extras.Json(entry.payload), entry.actor),
            )

    async def read_all(self) -> list[AuditLogEntry]:
        return await asyncio.to_thread(self._read_all_sync)

    def _read_all_sync(self) -> list[AuditLogEntry]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT entry_id, prev_hash, entry_hash, ts, entry_type, payload, actor "
                "FROM logbook ORDER BY seq ASC"
            )
            rows = cur.fetchall()
        return [
            AuditLogEntry(
                entry_id=str(r[0]), prev_hash=r[1], entry_hash=r[2], ts=r[3],
                entry_type=r[4], payload=r[5], actor=r[6],
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()

    async def _tamper_for_testing(self, entry_id: str, new_payload: dict) -> None:
        """Test-only, mirrors SQLiteLogbookStore's — see that class's
        docstring for what this simulates."""
        await asyncio.to_thread(self._tamper_sync, entry_id, new_payload)

    def _tamper_sync(self, entry_id: str, new_payload: dict) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE logbook SET payload = %s WHERE entry_id = %s",
                (json.dumps(new_payload), entry_id),
            )
