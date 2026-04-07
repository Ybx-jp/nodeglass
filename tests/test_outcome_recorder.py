"""Tests for outcome recording and storage persistence (NOD-34).

AC:
- [x] After context exit, repository.get_execution(id) returns the complete record
- [x] DAG in execution matches the observed operations
- [x] predicted_risk is populated
"""

import pytest

from workflow_eval.instrumentation.outcome import derive_outcome
from workflow_eval.instrumentation.recorder import WorkflowRecorder
from workflow_eval.instrumentation.sdk import workflow_context
from workflow_eval.storage.repository import SQLiteWorkflowRepository
from workflow_eval.types import (
    ExecutionOutcome,
    ExecutionRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo() -> SQLiteWorkflowRepository:
    r = SQLiteWorkflowRepository(":memory:")
    yield r
    r.close()


@pytest.fixture()
def recorder(repo: SQLiteWorkflowRepository) -> WorkflowRecorder:
    return WorkflowRecorder(repo)


# ---------------------------------------------------------------------------
# AC: After context exit, repository.get_execution(id) returns complete record
# ---------------------------------------------------------------------------


class TestPersistenceOnExit:
    @pytest.mark.asyncio()
    async def test_execution_persisted_on_exit(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        async with workflow_context("test-wf", recorder=recorder) as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
        # Recorder was called on exit — find the execution
        execution = wf.get_execution()
        got = repo.get_execution(execution.id)
        assert got.id == execution.id
        assert got.workflow_name == "test-wf"

    @pytest.mark.asyncio()
    async def test_manual_record(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        """Can also call recorder.record() manually."""
        async with workflow_context("manual") as wf:
            async with wf.operation("read_file"):
                pass
        exec_id = recorder.record(wf)
        got = repo.get_execution(exec_id)
        assert got.workflow_name == "manual"

    @pytest.mark.asyncio()
    async def test_records_preserved(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        async with workflow_context("test", recorder=recorder) as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api") as op:
                op.record_failure("timeout")
        got = repo.get_execution(wf.get_execution().id)
        assert len(got.records) == 2
        assert got.records[0].outcome == ExecutionOutcome.SUCCESS
        assert got.records[1].outcome == ExecutionOutcome.FAILURE
        assert got.records[1].error == "timeout"


# ---------------------------------------------------------------------------
# AC: DAG in execution matches the observed operations
# ---------------------------------------------------------------------------


class TestDAGMatches:
    @pytest.mark.asyncio()
    async def test_dag_matches_operations(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        async with workflow_context("test", recorder=recorder) as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
            async with wf.operation("send_email"):
                pass
        got = repo.get_execution(wf.get_execution().id)
        assert len(got.dag.nodes) == 3
        assert len(got.dag.edges) == 2
        ops = [n.operation for n in got.dag.nodes]
        assert ops == ["read_file", "invoke_api", "send_email"]

    @pytest.mark.asyncio()
    async def test_dag_equals_context_dag(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        async with workflow_context("test", recorder=recorder) as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("delete_record"):
                pass
        got = repo.get_execution(wf.get_execution().id)
        assert got.dag == wf.get_dag()


# ---------------------------------------------------------------------------
# AC: predicted_risk is populated
# ---------------------------------------------------------------------------


class TestPredictedRisk:
    @pytest.mark.asyncio()
    async def test_predicted_risk_populated(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        async with workflow_context("test", recorder=recorder) as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("delete_record"):
                pass
        got = repo.get_execution(wf.get_execution().id)
        assert got.predicted_risk is not None
        assert got.predicted_risk > 0.0

    @pytest.mark.asyncio()
    async def test_predicted_risk_matches_scoring(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        async with workflow_context("test", recorder=recorder) as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
            risk = wf.get_current_risk()
        got = repo.get_execution(wf.get_execution().id)
        assert got.predicted_risk == pytest.approx(risk.aggregate_score)

    @pytest.mark.asyncio()
    async def test_actual_outcome_derived(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        async with workflow_context("test", recorder=recorder) as wf:
            async with wf.operation("read_file"):
                pass
        got = repo.get_execution(wf.get_execution().id)
        assert got.actual_outcome == ExecutionOutcome.SUCCESS

    @pytest.mark.asyncio()
    async def test_actual_outcome_failure(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        async with workflow_context("test", recorder=recorder) as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api") as op:
                op.record_failure("bad gateway")
        got = repo.get_execution(wf.get_execution().id)
        assert got.actual_outcome == ExecutionOutcome.FAILURE


# ---------------------------------------------------------------------------
# derive_outcome unit tests
# ---------------------------------------------------------------------------


class TestDeriveOutcome:
    def test_all_success(self) -> None:
        records = (
            ExecutionRecord(node_id="a", operation="read_file", outcome=ExecutionOutcome.SUCCESS),
            ExecutionRecord(node_id="b", operation="invoke_api", outcome=ExecutionOutcome.SUCCESS),
        )
        assert derive_outcome(records) == ExecutionOutcome.SUCCESS

    def test_any_failure(self) -> None:
        records = (
            ExecutionRecord(node_id="a", operation="read_file", outcome=ExecutionOutcome.SUCCESS),
            ExecutionRecord(node_id="b", operation="invoke_api", outcome=ExecutionOutcome.FAILURE),
        )
        assert derive_outcome(records) == ExecutionOutcome.FAILURE

    def test_all_skipped(self) -> None:
        records = (
            ExecutionRecord(node_id="a", operation="read_file", outcome=ExecutionOutcome.SKIPPED),
        )
        assert derive_outcome(records) == ExecutionOutcome.SKIPPED

    def test_empty_records(self) -> None:
        assert derive_outcome(()) is None

    def test_success_and_skipped(self) -> None:
        records = (
            ExecutionRecord(node_id="a", operation="read_file", outcome=ExecutionOutcome.SUCCESS),
            ExecutionRecord(node_id="b", operation="invoke_api", outcome=ExecutionOutcome.SKIPPED),
        )
        assert derive_outcome(records) == ExecutionOutcome.SUCCESS
