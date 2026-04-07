"""Unit tests for storage layer (NOD-31).

AC:
- [x] pytest tests/test_storage.py -v passes
- [x] Covers store/get/list for both workflows and executions
- [x] Tests missing key error cases

Complements test_storage_migrations.py (schema DDL) and
test_storage_repository.py (field-level round-trip fidelity).
This file focuses on CRUD behavior, multi-entity interactions,
and edge cases.
"""

import sqlite3

import pytest

from workflow_eval.storage.repository import SQLiteWorkflowRepository
from workflow_eval.types import (
    DAGEdge,
    DAGNode,
    EdgeType,
    ExecutionOutcome,
    ExecutionRecord,
    RiskLevel,
    RiskProfile,
    SubScore,
    WorkflowDAG,
    WorkflowExecution,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo() -> SQLiteWorkflowRepository:
    r = SQLiteWorkflowRepository(":memory:")
    yield r
    r.close()


def _dag(name: str = "wf") -> WorkflowDAG:
    return WorkflowDAG(
        name=name,
        nodes=(
            DAGNode(id="n1", operation="read_file"),
            DAGNode(id="n2", operation="invoke_api"),
        ),
        edges=(DAGEdge(source_id="n1", target_id="n2"),),
    )


def _risk(name: str = "wf", score: float = 0.5) -> RiskProfile:
    return RiskProfile(
        workflow_name=name,
        aggregate_score=score,
        risk_level=RiskLevel.MEDIUM,
        sub_scores=(SubScore(name="test", score=score, weight=1.0),),
        node_count=2,
        edge_count=1,
    )


def _execution(
    exec_id: str,
    name: str = "wf",
    outcome: ExecutionOutcome | None = ExecutionOutcome.SUCCESS,
) -> WorkflowExecution:
    return WorkflowExecution(
        id=exec_id,
        workflow_name=name,
        dag=_dag(name),
        records=(
            ExecutionRecord(
                node_id="n1", operation="read_file",
                outcome=ExecutionOutcome.SUCCESS,
            ),
        ),
        predicted_risk=0.5,
        actual_outcome=outcome,
    )


# ---------------------------------------------------------------------------
# AC: Covers store/get for workflows
# ---------------------------------------------------------------------------


class TestWorkflowCRUD:
    def test_store_returns_id(self, repo: SQLiteWorkflowRepository) -> None:
        wid = repo.store_workflow(_dag(), _risk())
        assert isinstance(wid, str)
        assert len(wid) > 0

    def test_store_generates_unique_ids(self, repo: SQLiteWorkflowRepository) -> None:
        ids = {repo.store_workflow(_dag(), _risk()) for _ in range(10)}
        assert len(ids) == 10

    def test_get_after_store(self, repo: SQLiteWorkflowRepository) -> None:
        dag, risk = _dag("alpha"), _risk("alpha", 0.3)
        wid = repo.store_workflow(dag, risk)
        got_dag, got_risk = repo.get_workflow(wid)
        assert got_dag.name == "alpha"
        assert got_risk.aggregate_score == pytest.approx(0.3)

    def test_multiple_workflows_independent(self, repo: SQLiteWorkflowRepository) -> None:
        wid1 = repo.store_workflow(_dag("first"), _risk("first", 0.2))
        wid2 = repo.store_workflow(_dag("second"), _risk("second", 0.8))
        dag1, risk1 = repo.get_workflow(wid1)
        dag2, risk2 = repo.get_workflow(wid2)
        assert dag1.name == "first"
        assert dag2.name == "second"
        assert risk1.aggregate_score == pytest.approx(0.2)
        assert risk2.aggregate_score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# AC: Covers store/get/list for executions
# ---------------------------------------------------------------------------


class TestExecutionCRUD:
    def test_store_and_get(self, repo: SQLiteWorkflowRepository) -> None:
        wid = repo.store_workflow(_dag(), _risk())
        repo.store_execution(wid, _execution("e1"))
        got = repo.get_execution("e1")
        assert got.id == "e1"
        assert got.actual_outcome == ExecutionOutcome.SUCCESS

    def test_multiple_executions_per_workflow(self, repo: SQLiteWorkflowRepository) -> None:
        wid = repo.store_workflow(_dag(), _risk())
        for i in range(5):
            repo.store_execution(wid, _execution(f"e{i}"))
        results = repo.list_executions(wid)
        assert len(results) == 5

    def test_list_filters_correctly(self, repo: SQLiteWorkflowRepository) -> None:
        wid_a = repo.store_workflow(_dag("a"), _risk("a"))
        wid_b = repo.store_workflow(_dag("b"), _risk("b"))
        repo.store_execution(wid_a, _execution("ea1", "a"))
        repo.store_execution(wid_a, _execution("ea2", "a"))
        repo.store_execution(wid_b, _execution("eb1", "b"))

        list_a = repo.list_executions(wid_a)
        list_b = repo.list_executions(wid_b)
        assert {e.id for e in list_a} == {"ea1", "ea2"}
        assert {e.id for e in list_b} == {"eb1"}

    def test_list_empty_when_no_executions(self, repo: SQLiteWorkflowRepository) -> None:
        wid = repo.store_workflow(_dag(), _risk())
        assert repo.list_executions(wid) == []

    def test_execution_with_skipped_outcome(self, repo: SQLiteWorkflowRepository) -> None:
        wid = repo.store_workflow(_dag(), _risk())
        repo.store_execution(wid, _execution("e-skip", outcome=ExecutionOutcome.SKIPPED))
        got = repo.get_execution("e-skip")
        assert got.actual_outcome == ExecutionOutcome.SKIPPED


# ---------------------------------------------------------------------------
# AC: Tests missing key error cases
# ---------------------------------------------------------------------------


class TestMissingKeys:
    def test_get_workflow_not_found(self, repo: SQLiteWorkflowRepository) -> None:
        with pytest.raises(KeyError):
            repo.get_workflow("does-not-exist")

    def test_get_execution_not_found(self, repo: SQLiteWorkflowRepository) -> None:
        with pytest.raises(KeyError):
            repo.get_execution("does-not-exist")

    def test_missing_key_contains_id(self, repo: SQLiteWorkflowRepository) -> None:
        """KeyError message includes the missing id for debugging."""
        with pytest.raises(KeyError, match="abc123"):
            repo.get_workflow("abc123")

    def test_fk_violation_on_bad_workflow_id(self, repo: SQLiteWorkflowRepository) -> None:
        """Storing an execution with a nonexistent workflow_id fails."""
        with pytest.raises(Exception):
            repo.store_execution("bad-wf-id", _execution("e1"))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_duplicate_execution_id_rejected(self, repo: SQLiteWorkflowRepository) -> None:
        wid = repo.store_workflow(_dag(), _risk())
        repo.store_execution(wid, _execution("dup"))
        with pytest.raises(Exception):
            repo.store_execution(wid, _execution("dup"))

    def test_execution_none_fields(self, repo: SQLiteWorkflowRepository) -> None:
        wid = repo.store_workflow(_dag(), _risk())
        repo.store_execution(wid, _execution("e-none", outcome=None))
        got = repo.get_execution("e-none")
        assert got.actual_outcome is None

    def test_close_then_reopen_memory(self) -> None:
        """After close, a new :memory: repo starts empty."""
        repo = SQLiteWorkflowRepository()
        wid = repo.store_workflow(_dag(), _risk())
        repo.close()
        repo2 = SQLiteWorkflowRepository()
        with pytest.raises(KeyError):
            repo2.get_workflow(wid)
        repo2.close()
