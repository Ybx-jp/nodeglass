"""WorkflowRepository protocol and SQLite implementation (NOD-30).

NOD-30 spec (Linear):
- storage/repository.py
- WorkflowRepository protocol + SQLiteWorkflowRepository(db_path)
- Methods: store_workflow(dag, risk_profile), get_workflow(id),
  store_execution(workflow_id, execution), get_execution(id),
  list_executions(workflow_id)

AC:
- [x] Store -> retrieve round-trip preserves all fields
- [x] Missing key raises KeyError
- [x] List returns correct subset
- [x] Works with :memory: and file-backed SQLite

Behavioral constraints from description:
- Protocol + concrete SQLite implementation
- store_workflow returns generated id
- Execution FK links to workflow via workflow_id parameter
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Protocol, runtime_checkable

from workflow_eval.storage.migrations import initialize_db
from workflow_eval.types import (
    ExecutionOutcome,
    ExecutionRecord,
    RiskProfile,
    WorkflowDAG,
    WorkflowExecution,
)


@runtime_checkable
class WorkflowRepository(Protocol):
    """Interface for workflow + execution persistence."""

    def store_workflow(self, dag: WorkflowDAG, risk_profile: RiskProfile) -> str: ...

    def get_workflow(self, workflow_id: str) -> tuple[WorkflowDAG, RiskProfile]: ...

    def store_execution(
        self, workflow_id: str, execution: WorkflowExecution,
    ) -> None: ...

    def get_execution(self, execution_id: str) -> WorkflowExecution: ...

    def list_executions(self, workflow_id: str) -> list[WorkflowExecution]: ...


class SQLiteWorkflowRepository:
    """SQLite-backed WorkflowRepository."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        initialize_db(self._conn)

    # -- workflows ------------------------------------------------------------

    def store_workflow(self, dag: WorkflowDAG, risk_profile: RiskProfile) -> str:
        """Persist a scored workflow DAG. Returns the generated id."""
        workflow_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO workflows (id, name, dag_json, risk_profile_json)"
            " VALUES (?, ?, ?, ?)",
            (workflow_id, dag.name, dag.model_dump_json(), risk_profile.model_dump_json()),
        )
        self._conn.commit()
        return workflow_id

    def get_workflow(self, workflow_id: str) -> tuple[WorkflowDAG, RiskProfile]:
        """Retrieve a workflow by id. Raises KeyError if not found."""
        row = self._conn.execute(
            "SELECT dag_json, risk_profile_json FROM workflows WHERE id = ?",
            (workflow_id,),
        ).fetchone()
        if row is None:
            raise KeyError(workflow_id)
        dag = WorkflowDAG.model_validate_json(row[0])
        risk_profile = RiskProfile.model_validate_json(row[1])
        return (dag, risk_profile)

    # -- executions ------------------------------------------------------------

    def store_execution(
        self, workflow_id: str, execution: WorkflowExecution,
    ) -> None:
        """Persist an execution trace linked to a workflow."""
        records_json = json.dumps([r.model_dump() for r in execution.records])
        self._conn.execute(
            "INSERT INTO executions"
            " (id, workflow_id, dag_json, records_json, predicted_risk, actual_outcome)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                execution.id,
                workflow_id,
                execution.dag.model_dump_json(),
                records_json,
                execution.predicted_risk,
                execution.actual_outcome.value if execution.actual_outcome else None,
            ),
        )
        self._conn.commit()

    def get_execution(self, execution_id: str) -> WorkflowExecution:
        """Retrieve an execution by id. Raises KeyError if not found."""
        row = self._conn.execute(
            "SELECT id, workflow_id, dag_json, records_json,"
            " predicted_risk, actual_outcome"
            " FROM executions WHERE id = ?",
            (execution_id,),
        ).fetchone()
        if row is None:
            raise KeyError(execution_id)
        return self._row_to_execution(row)

    def list_executions(self, workflow_id: str) -> list[WorkflowExecution]:
        """List all executions for a workflow, ordered by creation time."""
        rows = self._conn.execute(
            "SELECT id, workflow_id, dag_json, records_json,"
            " predicted_risk, actual_outcome"
            " FROM executions WHERE workflow_id = ?"
            " ORDER BY created_at",
            (workflow_id,),
        ).fetchall()
        return [self._row_to_execution(r) for r in rows]

    @staticmethod
    def _row_to_execution(row: tuple) -> WorkflowExecution:
        """Reconstruct a WorkflowExecution from a database row.

        Row columns: (id, workflow_id, dag_json, records_json,
                       predicted_risk, actual_outcome)
        """
        exec_id, _workflow_id, dag_json, records_json, predicted_risk, actual_outcome = row
        dag = WorkflowDAG.model_validate_json(dag_json)
        records = tuple(
            ExecutionRecord.model_validate(r) for r in json.loads(records_json)
        )
        return WorkflowExecution(
            id=exec_id,
            workflow_name=dag.name,
            dag=dag,
            records=records,
            predicted_risk=predicted_risk,
            actual_outcome=ExecutionOutcome(actual_outcome) if actual_outcome else None,
        )

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
