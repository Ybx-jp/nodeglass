"""SQLite schema DDL and migrations (NOD-29).

NOD-29 spec (Linear):
- storage/migrations.py
- DDL for `workflows` table (id, name, dag_json, risk_profile_json, timestamps)
- DDL for `executions` table (id, workflow_id FK, dag_json, records_json,
  predicted_risk, actual_outcome, timestamps) with indexes
- Idempotent CREATE TABLE IF NOT EXISTS

AC:
- [x] Calling initialize_db(connection) twice doesn't error
- [x] Tables and indexes exist after initialization
- [x] Commented placeholder for future workflow_embeddings table

Behavioral constraints from description:
- Indexes on executions table
- Idempotent schema creation
"""

from __future__ import annotations

import sqlite3


def initialize_db(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes. Idempotent — safe to call multiple times."""
    conn.executescript(_SCHEMA_DDL)


_SCHEMA_DDL = """\
-- workflows: stores scored workflow DAGs
CREATE TABLE IF NOT EXISTS workflows (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    dag_json    TEXT NOT NULL,
    risk_profile_json TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_workflows_name ON workflows(name);
CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON workflows(created_at);

-- executions: observed workflow runs with outcomes
CREATE TABLE IF NOT EXISTS executions (
    id              TEXT PRIMARY KEY,
    workflow_id     TEXT NOT NULL REFERENCES workflows(id),
    dag_json        TEXT NOT NULL,
    records_json    TEXT NOT NULL,
    predicted_risk  REAL,
    actual_outcome  TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_executions_workflow_id ON executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_executions_created_at ON executions(created_at);

-- Placeholder for Layer 3: learned workflow embeddings.
-- This table will store vector representations of workflow DAGs
-- for similarity search and risk prediction from historical data.
--
-- CREATE TABLE IF NOT EXISTS workflow_embeddings (
--     workflow_id TEXT PRIMARY KEY REFERENCES workflows(id),
--     embedding   BLOB NOT NULL,
--     model_version TEXT NOT NULL,
--     created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
-- );
"""
