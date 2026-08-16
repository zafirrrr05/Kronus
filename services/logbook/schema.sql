-- features.txt component 7: "Postgres append-only + hash chain."
-- Applied once at deployment time (see docs/setup.md). The application
-- role should be granted INSERT and SELECT only — never UPDATE or
-- DELETE — so "append-only" is enforced by the database, not just by
-- convention in application code.

CREATE TABLE IF NOT EXISTS logbook (
    seq         BIGSERIAL PRIMARY KEY,
    entry_id    UUID NOT NULL UNIQUE,
    prev_hash   CHAR(64) NOT NULL,
    entry_hash  CHAR(64) NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    entry_type  TEXT NOT NULL,
    payload     JSONB NOT NULL,
    actor       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_logbook_ts ON logbook (ts);
CREATE INDEX IF NOT EXISTS idx_logbook_entry_type ON logbook (entry_type);
