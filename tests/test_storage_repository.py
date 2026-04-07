"""Tests for WorkflowRepository (NOD-30).

AC:
- [x] Store -> retrieve round-trip preserves all fields
- [x] Missing key raises KeyError
- [x] List returns correct subset
- [x] Works with :memory: and file-backed SQLite
"""

import tempfile
from pathlib import Path

import pytest

from workflow_eval.storage.repository import (
    SQLiteWorkflowRepository,
    WorkflowRepository,
)
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


def _sample_dag(name: str = "test-wf") -> WorkflowDAG:
    return WorkflowDAG(
        name=name,
        nodes=(
            DAGNode(id="a", operation="read_file", params={"path": "/tmp/x"}),
            DAGNode(id="b", operation="invoke_api", metadata={"retries": 3}),
        ),
        edges=(
            DAGEdge(source_id="a", target_id="b", edge_type=EdgeType.DATA_FLOW),
        ),
        metadata={"version": "1.0"},
    )


def _sample_risk_profile() -> RiskProfile:
    return RiskProfile(
        workflow_name="test-wf",
        aggregate_score=0.42,
        risk_level=RiskLevel.MEDIUM,
        sub_scores=(
            SubScore(name="fan_out", score=0.3, weight=0.15, details={"max": 2}),
            SubScore(name="chain_depth", score=0.5, weight=0.20, flagged_nodes=("b",)),
        ),
        node_count=2,
        edge_count=1,
        critical_paths=(("a", "b"),),
        chokepoints=("b",),
    )


def _sample_execution(exec_id: str = "e1") -> WorkflowExecution:
    return WorkflowExecution(
        id=exec_id,
        workflow_name="test-wf",
        dag=_sample_dag(),
        records=(
            ExecutionRecord(
                node_id="a", operation="read_file",
                outcome=ExecutionOutcome.SUCCESS, duration_ms=12.5,
            ),
            ExecutionRecord(
                node_id="b", operation="invoke_api",
                outcome=ExecutionOutcome.FAILURE, error="timeout",
            ),
        ),
        predicted_risk=0.42,
        actual_outcome=ExecutionOutcome.FAILURE,
    )


# ---------------------------------------------------------------------------
# AC: Store -> retrieve round-trip preserves all fields
# ---------------------------------------------------------------------------


class TestWorkflowRoundTrip:
    def test_dag_round_trip(self, repo: SQLiteWorkflowRepository) -> None:
        dag = _sample_dag()
        risk = _sample_risk_profile()
        wid = repo.store_workflow(dag, risk)
        got_dag, got_risk = repo.get_workflow(wid)
        assert got_dag == dag

    def test_risk_profile_round_trip(self, repo: SQLiteWorkflowRepository) -> None:
        dag = _sample_dag()
        risk = _sample_risk_profile()
        wid = repo.store_workflow(dag, risk)
        got_dag, got_risk = repo.get_workflow(wid)
        assert got_risk == risk

    def test_sub_scores_preserved(self, repo: SQLiteWorkflowRepository) -> None:
        risk = _sample_risk_profile()
        wid = repo.store_workflow(_sample_dag(), risk)
        _, got_risk = repo.get_workflow(wid)
        assert got_risk.sub_scores == risk.sub_scores
        assert got_risk.critical_paths == risk.critical_paths
        assert got_risk.chokepoints == risk.chokepoints

    def test_dag_metadata_preserved(self, repo: SQLiteWorkflowRepository) -> None:
        dag = _sample_dag()
        wid = repo.store_workflow(dag, _sample_risk_profile())
        got_dag, _ = repo.get_workflow(wid)
        assert got_dag.metadata == {"version": "1.0"}
        assert got_dag.nodes[0].params == {"path": "/tmp/x"}
        assert got_dag.nodes[1].metadata == {"retries": 3}

    def test_edge_type_preserved(self, repo: SQLiteWorkflowRepository) -> None:
        dag = _sample_dag()
        wid = repo.store_workflow(dag, _sample_risk_profile())
        got_dag, _ = repo.get_workflow(wid)
        assert got_dag.edges[0].edge_type == EdgeType.DATA_FLOW


class TestExecutionRoundTrip:
    def test_full_execution_round_trip(self, repo: SQLiteWorkflowRepository) -> None:
        wid = repo.store_workflow(_sample_dag(), _sample_risk_profile())
        execution = _sample_execution()
        repo.store_execution(wid, execution)
        got = repo.get_execution("e1")
        assert got.id == execution.id
        assert got.workflow_name == execution.workflow_name
        assert got.dag == execution.dag
        assert got.records == execution.records
        assert got.predicted_risk == execution.predicted_risk
        assert got.actual_outcome == execution.actual_outcome

    def test_records_fields_preserved(self, repo: SQLiteWorkflowRepository) -> None:
        wid = repo.store_workflow(_sample_dag(), _sample_risk_profile())
        repo.store_execution(wid, _sample_execution())
        got = repo.get_execution("e1")
        assert got.records[0].duration_ms == 12.5
        assert got.records[1].error == "timeout"

    def test_nullable_fields(self, repo: SQLiteWorkflowRepository) -> None:
        """Execution with None predicted_risk and actual_outcome."""
        wid = repo.store_workflow(_sample_dag(), _sample_risk_profile())
        execution = WorkflowExecution(
            id="e-null",
            workflow_name="test-wf",
            dag=_sample_dag(),
            records=(),
            predicted_risk=None,
            actual_outcome=None,
        )
        repo.store_execution(wid, execution)
        got = repo.get_execution("e-null")
        assert got.predicted_risk is None
        assert got.actual_outcome is None


# ---------------------------------------------------------------------------
# AC: Missing key raises KeyError
# ---------------------------------------------------------------------------


class TestMissingKey:
    def test_get_workflow_missing(self, repo: SQLiteWorkflowRepository) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            repo.get_workflow("nonexistent")

    def test_get_execution_missing(self, repo: SQLiteWorkflowRepository) -> None:
        with pytest.raises(KeyError, match="no-such-exec"):
            repo.get_execution("no-such-exec")


# ---------------------------------------------------------------------------
# AC: List returns correct subset
# ---------------------------------------------------------------------------


class TestListExecutions:
    def test_list_filters_by_workflow(self, repo: SQLiteWorkflowRepository) -> None:
        wid1 = repo.store_workflow(_sample_dag("wf-1"), _sample_risk_profile())
        wid2 = repo.store_workflow(_sample_dag("wf-2"), _sample_risk_profile())
        repo.store_execution(wid1, _sample_execution("e1"))
        repo.store_execution(wid1, _sample_execution("e2"))
        repo.store_execution(wid2, _sample_execution("e3"))

        result = repo.list_executions(wid1)
        assert len(result) == 2
        assert {e.id for e in result} == {"e1", "e2"}

    def test_list_empty_for_unknown_workflow(self, repo: SQLiteWorkflowRepository) -> None:
        result = repo.list_executions("no-such-workflow")
        assert result == []

    def test_list_preserves_order(self, repo: SQLiteWorkflowRepository) -> None:
        """Executions returned in creation order."""
        wid = repo.store_workflow(_sample_dag(), _sample_risk_profile())
        for i in range(5):
            repo.store_execution(wid, _sample_execution(f"e{i}"))
        result = repo.list_executions(wid)
        assert [e.id for e in result] == [f"e{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# AC: Works with :memory: and file-backed SQLite
# ---------------------------------------------------------------------------


class TestFileBacked:
    def test_file_backed_round_trip(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        repo = SQLiteWorkflowRepository(db_path)
        dag = _sample_dag()
        risk = _sample_risk_profile()
        wid = repo.store_workflow(dag, risk)
        repo.store_execution(wid, _sample_execution())
        repo.close()

        # Reopen from disk
        repo2 = SQLiteWorkflowRepository(db_path)
        got_dag, got_risk = repo2.get_workflow(wid)
        assert got_dag == dag
        assert got_risk == risk
        execs = repo2.list_executions(wid)
        assert len(execs) == 1
        assert execs[0].id == "e1"
        repo2.close()

    def test_memory_works(self) -> None:
        """Explicit test that :memory: is the default and functional."""
        repo = SQLiteWorkflowRepository()
        wid = repo.store_workflow(_sample_dag(), _sample_risk_profile())
        got_dag, _ = repo.get_workflow(wid)
        assert got_dag.name == "test-wf"
        repo.close()


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(SQLiteWorkflowRepository(), WorkflowRepository)
