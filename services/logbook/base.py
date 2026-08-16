"""features.txt component 7: "a durable, permanent, tamper-evident record
of every detection, decision, and action — hash-chained so each entry
locks in the one before it." NFR-13: "100%, 0 broken hash links... nightly
verify."

The hash computation itself already lives in AuditLogEntry's own
model_validator (libs/schemas.py) — a store's job is orchestration
(fetch the current tip's hash, construct the next entry with it as
prev_hash, persist) and verification (walk the chain, recompute every
hash from stored fields, confirm nothing was altered). Both are backend-
agnostic, so they're implemented once here; SQLiteLogbookStore and
PostgresLogbookStore only implement the storage primitives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from libs.constants import AuditEntryType
from libs.observability import observe
from libs.schemas import GENESIS_HASH, AuditLogEntry


def verify_chain_integrity(entries: list[AuditLogEntry]) -> tuple[bool, int | None]:
    """Walks entries in order, recomputing each hash from its own stored
    fields and confirming: (a) it links to the previous entry's hash, and
    (b) the recomputed hash matches what was stored. Returns (ok, index)
    where index is the first broken entry's position, or None if the
    chain is intact.
    """
    prev_hash = GENESIS_HASH
    for i, entry in enumerate(entries):
        if entry.prev_hash != prev_hash:
            return False, i
        recomputed = AuditLogEntry(
            entry_id=entry.entry_id, prev_hash=entry.prev_hash, ts=entry.ts,
            entry_type=entry.entry_type, payload=entry.payload, actor=entry.actor,
        )
        if recomputed.entry_hash != entry.entry_hash:
            return False, i
        prev_hash = entry.entry_hash
    return True, None


class LogbookStore(ABC):
    @abstractmethod
    async def _get_latest_hash(self) -> str:
        """Returns GENESIS_HASH if the chain is empty."""

    @abstractmethod
    async def _persist(self, entry: AuditLogEntry) -> None: ...

    @abstractmethod
    async def read_all(self) -> list[AuditLogEntry]: ...

    async def append(
        self, entry_type: AuditEntryType, payload: dict, actor: str = "system"
    ) -> AuditLogEntry:
        with observe("logbook", "append", entry_type=entry_type):
            prev_hash = await self._get_latest_hash()
            entry = AuditLogEntry(prev_hash=prev_hash, entry_type=entry_type, payload=payload, actor=actor)
            await self._persist(entry)
            return entry

    async def verify_chain(self) -> tuple[bool, int | None]:
        with observe("logbook", "verify_chain"):
            entries = await self.read_all()
            return verify_chain_integrity(entries)
