"""Demo/test backend for the Logbook. stdlib sqlite3 only — no new
dependency for the path that needs to run without any infrastructure
standing up first (ponytail: stdlib beats an installed dependency when it
covers the need, and it does here).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from libs.schemas import GENESIS_HASH, AuditLogEntry
from services.logbook.base import LogbookStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS logbook (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,
    prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL,
    ts TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    actor TEXT NOT NULL
);
"""


class SQLiteLogbookStore(LogbookStore):
    def __init__(self, path: str | Path = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: sqlite3 connections are thread-bound by
        # default, which breaks the moment this store is used behind an
        # ASGI server that may service requests from a different thread
        # than the one that constructed it (caught for real via FastAPI's
        # TestClient, which runs the app in its own thread — see
        # tests/integration/test_api.py). Safe to relax here specifically
        # because WAL mode (below) already gives SQLite proper concurrent
        # access semantics; this store still isn't meant for production
        # multi-writer load — that's PostgresLogbookStore's job.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        # WAL + NORMAL synchronous: the standard SQLite tuning for
        # write-heavy workloads. Caught for real during the flow-case
        # integration tests: the default (fsync on every single commit)
        # made a ~2,400-row replay noticeably slow — nowhere near NFR-5's
        # 3,000-5,000 events/sec target. WAL still fsyncs on checkpoint,
        # not on every write, which is the right trade for a demo/dev
        # backend; production uses Postgres (postgres_store.py) regardless.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    async def _get_latest_hash(self) -> str:
        row = self._conn.execute(
            "SELECT entry_hash FROM logbook ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else GENESIS_HASH

    async def _persist(self, entry: AuditLogEntry) -> None:
        self._conn.execute(
            "INSERT INTO logbook (entry_id, prev_hash, entry_hash, ts, entry_type, payload, actor) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (entry.entry_id, entry.prev_hash, entry.entry_hash, entry.ts.isoformat(),
             entry.entry_type, json.dumps(entry.payload), entry.actor),
        )
        self._conn.commit()

    async def read_all(self) -> list[AuditLogEntry]:
        rows = self._conn.execute(
            "SELECT entry_id, prev_hash, entry_hash, ts, entry_type, payload, actor "
            "FROM logbook ORDER BY seq ASC"
        ).fetchall()
        return [
            AuditLogEntry(
                entry_id=r[0], prev_hash=r[1], entry_hash=r[2], ts=r[3],
                entry_type=r[4], payload=json.loads(r[5]), actor=r[6],
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()

    async def _tamper_for_testing(self, entry_id: str, new_payload: dict) -> None:
        """Test-only: mutates a stored row's payload without recomputing
        its hash, simulating exactly the kind of quiet edit NFR-13 exists
        to detect. Underscore-prefixed and documented as such — not part
        of the public store interface.
        """
        self._conn.execute(
            "UPDATE logbook SET payload = ? WHERE entry_id = ?",
            (json.dumps(new_payload), entry_id),
        )
        self._conn.commit()
