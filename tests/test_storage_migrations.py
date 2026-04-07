"""Tests for SQLite schema and migrations (NOD-29).

AC:
- [x] Calling initialize_db(connection) twice doesn't error
- [x] Tables and indexes exist after initialization
- [x] Commented placeholder for future workflow_embeddings table
"""

import sqlite3

import pytest

from workflow_eval.storage.migrations import initialize_db


@pytest.fixture()
def conn() -> sqlite3.Connection:
    """In-memory SQLite connection."""
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# AC: Calling initialize_db(connection) twice doesn't error
# ---------------------------------------------------------------------------


class TestIdempotent:
    def test_initialize_twice_no_error(self, conn: sqlite3.Connection) -> None:
        initialize_db(conn)
        initialize_db(conn)  # should not raise

    def test_initialize_three_times(self, conn: sqlite3.Connection) -> None:
        """Repeated calls remain idempotent."""
        for _ in range(3):
            initialize_db(conn)


# ---------------------------------------------------------------------------
# AC: Tables and indexes exist after initialization
# ---------------------------------------------------------------------------


class TestTablesExist:
    def test_workflows_table_exists(self, conn: sqlite3.Connection) -> None:
        initialize_db(conn)
        tables = _get_tables(conn)
        assert "workflows" in tables

    def test_executions_table_exists(self, conn: sqlite3.Connection) -> None:
        initialize_db(conn)
        tables = _get_tables(conn)
        assert "executions" in tables

    def test_workflows_columns(self, conn: sqlite3.Connection) -> None:
        initialize_db(conn)
        cols = _get_columns(conn, "workflows")
        expected = {"id", "name", "dag_json", "risk_profile_json", "created_at", "updated_at"}
        assert expected == {c["name"] for c in cols}

    def test_executions_columns(self, conn: sqlite3.Connection) -> None:
        initialize_db(conn)
        cols = _get_columns(conn, "executions")
        expected = {
            "id", "workflow_id", "dag_json", "records_json",
            "predicted_risk", "actual_outcome", "created_at", "updated_at",
        }
        assert expected == {c["name"] for c in cols}

    def test_executions_predicted_risk_nullable(self, conn: sqlite3.Connection) -> None:
        """predicted_risk and actual_outcome should allow NULL."""
        initialize_db(conn)
        cols = _get_columns(conn, "executions")
        col_map = {c["name"]: c for c in cols}
        assert col_map["predicted_risk"]["notnull"] == 0
        assert col_map["actual_outcome"]["notnull"] == 0


class TestIndexesExist:
    def test_workflows_indexes(self, conn: sqlite3.Connection) -> None:
        initialize_db(conn)
        indexes = _get_indexes(conn)
        assert "idx_workflows_name" in indexes
        assert "idx_workflows_created_at" in indexes

    def test_executions_indexes(self, conn: sqlite3.Connection) -> None:
        initialize_db(conn)
        indexes = _get_indexes(conn)
        assert "idx_executions_workflow_id" in indexes
        assert "idx_executions_created_at" in indexes


# ---------------------------------------------------------------------------
# AC: Commented placeholder for future workflow_embeddings table
# ---------------------------------------------------------------------------


class TestEmbeddingsPlaceholder:
    def test_placeholder_in_source(self) -> None:
        """The DDL contains a commented-out workflow_embeddings table."""
        from workflow_eval.storage.migrations import _SCHEMA_DDL
        assert "workflow_embeddings" in _SCHEMA_DDL
        # It should be commented out, not active
        tables_with_embeddings = [
            line for line in _SCHEMA_DDL.splitlines()
            if "workflow_embeddings" in line and not line.strip().startswith("--")
        ]
        assert len(tables_with_embeddings) == 0

    def test_embeddings_table_not_created(self, conn: sqlite3.Connection) -> None:
        """The placeholder should NOT create an actual table."""
        initialize_db(conn)
        tables = _get_tables(conn)
        assert "workflow_embeddings" not in tables


# ---------------------------------------------------------------------------
# Functional: basic insert/query to validate schema works
# ---------------------------------------------------------------------------


class TestSchemaFunctional:
    def test_insert_workflow(self, conn: sqlite3.Connection) -> None:
        initialize_db(conn)
        conn.execute(
            "INSERT INTO workflows (id, name, dag_json, risk_profile_json) VALUES (?, ?, ?, ?)",
            ("w1", "test-workflow", '{"nodes":[]}', '{"score":0.5}'),
        )
        row = conn.execute("SELECT * FROM workflows WHERE id = 'w1'").fetchone()
        assert row is not None

    def test_insert_execution_with_fk(self, conn: sqlite3.Connection) -> None:
        initialize_db(conn)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO workflows (id, name, dag_json, risk_profile_json) VALUES (?, ?, ?, ?)",
            ("w1", "test", '{}', '{}'),
        )
        conn.execute(
            "INSERT INTO executions (id, workflow_id, dag_json, records_json) VALUES (?, ?, ?, ?)",
            ("e1", "w1", '{}', '[]'),
        )
        row = conn.execute("SELECT * FROM executions WHERE id = 'e1'").fetchone()
        assert row is not None

    def test_timestamps_auto_populated(self, conn: sqlite3.Connection) -> None:
        initialize_db(conn)
        conn.execute(
            "INSERT INTO workflows (id, name, dag_json, risk_profile_json) VALUES (?, ?, ?, ?)",
            ("w1", "test", '{}', '{}'),
        )
        row = conn.execute("SELECT created_at, updated_at FROM workflows WHERE id = 'w1'").fetchone()
        assert row[0] is not None
        assert row[1] is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _get_columns(conn: sqlite3.Connection, table: str) -> list[dict]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [{"name": r[1], "type": r[2], "notnull": r[3], "pk": r[5]} for r in rows]


def _get_indexes(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    return {r[0] for r in rows}
