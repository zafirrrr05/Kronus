import pytest

from libs.constants import AuditEntryType
from services.logbook.sqlite_store import SQLiteLogbookStore

try:
    from services.logbook.postgres_store import PostgresLogbookStore

    _pg_available = True
except ImportError:
    _pg_available = False

POSTGRES_DSN = "dbname=kronus user=kronus password=kronus_dev_local host=localhost"


def _make_sqlite():
    return SQLiteLogbookStore(":memory:")


def _make_postgres():
    store = PostgresLogbookStore(POSTGRES_DSN)
    store._conn.cursor().execute("TRUNCATE logbook")  # isolate each test run
    return store


def _backend_ids():
    ids = ["sqlite"]
    factories = [_make_sqlite]
    if _pg_available:
        try:
            _make_postgres().close()
            ids.append("postgres")
            factories.append(_make_postgres)
        except Exception:
            pass  # real postgres not reachable in this environment — sqlite still covers the contract
    return factories, ids


_FACTORIES, _IDS = _backend_ids()


@pytest.fixture(params=_FACTORIES, ids=_IDS)
def store(request):
    s = request.param()
    yield s
    s.close()


@pytest.mark.asyncio
async def test_first_entry_chains_to_genesis(store):
    from libs.schemas import GENESIS_HASH

    entry = await store.append(AuditEntryType.DETECTION, {"verdict_id": "v1"})
    assert entry.prev_hash == GENESIS_HASH
    assert len(entry.entry_hash) == 64


@pytest.mark.asyncio
async def test_second_entry_chains_to_first(store):
    e1 = await store.append(AuditEntryType.DETECTION, {"n": 1})
    e2 = await store.append(AuditEntryType.POLICY_DECISION, {"n": 2})
    assert e2.prev_hash == e1.entry_hash


@pytest.mark.asyncio
async def test_read_all_returns_entries_in_append_order(store):
    await store.append(AuditEntryType.DETECTION, {"n": 1})
    await store.append(AuditEntryType.DETECTION, {"n": 2})
    await store.append(AuditEntryType.DETECTION, {"n": 3})
    entries = await store.read_all()
    assert [e.payload["n"] for e in entries] == [1, 2, 3]


@pytest.mark.asyncio
async def test_verify_chain_is_true_for_an_untampered_chain(store):
    for i in range(5):
        await store.append(AuditEntryType.DETECTION, {"n": i})
    ok, broken_at = await store.verify_chain()
    assert ok is True
    assert broken_at is None


@pytest.mark.asyncio
async def test_verify_chain_detects_a_tampered_entry(store):
    # This is the actual NFR-13 claim ("0 broken hash links... nightly
    # verify") demonstrated, not just asserted: quietly edit a stored
    # row's payload without recomputing its hash, and confirm the chain
    # walk catches exactly where it broke.
    e0 = await store.append(AuditEntryType.DETECTION, {"n": 0})
    await store.append(AuditEntryType.DETECTION, {"n": 1})
    await store.append(AuditEntryType.DETECTION, {"n": 2})

    await store._tamper_for_testing(e0.entry_id, {"n": 999})

    ok, broken_at = await store.verify_chain()
    assert ok is False
    assert broken_at == 0


@pytest.mark.asyncio
async def test_actor_pattern_is_preserved_through_storage(store):
    await store.append(AuditEntryType.CORRECTION, {}, actor="operator:zafir")
    entries = await store.read_all()
    assert entries[0].actor == "operator:zafir"
