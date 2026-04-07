"""End-to-end instrumentation tests (NOD-35).

AC:
- [x] pytest tests/test_instrumentation.py -v passes
- [x] Instrumented DAG scores identically to manually-constructed equivalent
- [x] Covers context manager, decorator, and storage persistence
"""

import pytest

from workflow_eval.dag.builder import DAGBuilder
from workflow_eval.dag.models import to_networkx
from workflow_eval.instrumentation.recorder import WorkflowRecorder
from workflow_eval.instrumentation.sdk import (
    track_operation,
    workflow_context,
)
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.scoring.engine import RiskScoringEngine
from workflow_eval.storage.repository import SQLiteWorkflowRepository
from workflow_eval.types import (
    ExecutionOutcome,
    RiskProfile,
    ScoringConfig,
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


def _score_declarative(name: str, ops: list[tuple[str, str]]) -> RiskProfile:
    """Build a DAG declaratively and score it."""
    builder = DAGBuilder(name)
    builder.add_step(ops[0][0], ops[0][1])
    for nid, op in ops[1:]:
        builder.then(nid, op)
    dag = builder.build()
    nx_dag = to_networkx(dag)
    engine = RiskScoringEngine(ScoringConfig(), get_default_registry())
    return engine.score(nx_dag)


# ---------------------------------------------------------------------------
# AC: Instrumented DAG scores identically to manually-constructed equivalent
# ---------------------------------------------------------------------------


class TestScoreEquivalence:
    @pytest.mark.asyncio()
    async def test_two_node_pipeline_scores_match(self) -> None:
        """read_file → invoke_api: instrumented == declarative."""
        async with workflow_context("equiv-test") as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
            instrumented_risk = wf.get_current_risk()

        declarative_risk = _score_declarative(
            "equiv-test",
            [("read_file_0", "read_file"), ("invoke_api_1", "invoke_api")],
        )
        assert instrumented_risk.aggregate_score == pytest.approx(
            declarative_risk.aggregate_score,
        )

    @pytest.mark.asyncio()
    async def test_three_node_delete_cascade_scores_match(self) -> None:
        """authenticate → invoke_api → delete_record."""
        async with workflow_context("cascade") as wf:
            async with wf.operation("authenticate"):
                pass
            async with wf.operation("invoke_api"):
                pass
            async with wf.operation("delete_record"):
                pass
            instrumented_risk = wf.get_current_risk()

        declarative_risk = _score_declarative(
            "cascade",
            [
                ("authenticate_0", "authenticate"),
                ("invoke_api_1", "invoke_api"),
                ("delete_record_2", "delete_record"),
            ],
        )
        assert instrumented_risk.aggregate_score == pytest.approx(
            declarative_risk.aggregate_score,
        )
        assert instrumented_risk.risk_level == declarative_risk.risk_level

    @pytest.mark.asyncio()
    async def test_sub_scores_match(self) -> None:
        """All 6 sub-scores should be identical."""
        async with workflow_context("sub-score-test") as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("delete_record"):
                pass
            instrumented_risk = wf.get_current_risk()

        declarative_risk = _score_declarative(
            "sub-score-test",
            [("read_file_0", "read_file"), ("delete_record_1", "delete_record")],
        )
        assert len(instrumented_risk.sub_scores) == len(declarative_risk.sub_scores)
        for i_sub, d_sub in zip(
            instrumented_risk.sub_scores, declarative_risk.sub_scores,
        ):
            assert i_sub.name == d_sub.name
            assert i_sub.score == pytest.approx(d_sub.score)

    @pytest.mark.asyncio()
    async def test_single_pure_op_low_risk(self) -> None:
        """A single read_file should be low risk in both paths."""
        async with workflow_context("pure") as wf:
            async with wf.operation("read_file"):
                pass
            risk = wf.get_current_risk()
        declarative = _score_declarative("pure", [("read_file_0", "read_file")])
        assert risk.aggregate_score == pytest.approx(declarative.aggregate_score)
        assert risk.risk_level == declarative.risk_level


# ---------------------------------------------------------------------------
# AC: Covers context manager
# ---------------------------------------------------------------------------


class TestContextManagerE2E:
    @pytest.mark.asyncio()
    async def test_full_pipeline_context_manager(self) -> None:
        """Full workflow with mixed outcomes via context manager."""
        async with workflow_context("e2e-cm") as wf:
            async with wf.operation("authenticate"):
                pass
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api") as op:
                op.record_failure("503 Service Unavailable")
            async with wf.operation("send_email"):
                pass

        dag = wf.get_dag()
        assert len(dag.nodes) == 4
        assert len(dag.edges) == 3

        execution = wf.get_execution()
        assert execution.records[0].outcome == ExecutionOutcome.SUCCESS
        assert execution.records[1].outcome == ExecutionOutcome.SUCCESS
        assert execution.records[2].outcome == ExecutionOutcome.FAILURE
        assert execution.records[2].error == "503 Service Unavailable"
        assert execution.records[3].outcome == ExecutionOutcome.SUCCESS


# ---------------------------------------------------------------------------
# AC: Covers decorator
# ---------------------------------------------------------------------------


class TestDecoratorE2E:
    @pytest.mark.asyncio()
    async def test_decorator_pipeline(self) -> None:
        """Three decorated functions produce a correct linear DAG."""
        @track_operation("authenticate")
        async def auth():
            return "token-123"

        @track_operation("invoke_api", params={"method": "GET"})
        async def fetch_data():
            return {"users": []}

        @track_operation("send_email")
        async def notify():
            pass

        async with workflow_context("e2e-dec") as wf:
            token = await auth()
            data = await fetch_data()
            await notify()

        assert token == "token-123"
        assert data == {"users": []}
        dag = wf.get_dag()
        assert len(dag.nodes) == 3
        assert len(dag.edges) == 2
        ops = [n.operation for n in dag.nodes]
        assert ops == ["authenticate", "invoke_api", "send_email"]

    @pytest.mark.asyncio()
    async def test_decorator_failure_records_and_continues(self) -> None:
        @track_operation("invoke_api")
        async def flaky_call():
            raise ConnectionError("reset")

        @track_operation("send_email")
        async def fallback():
            pass

        async with workflow_context("e2e-fail") as wf:
            with pytest.raises(ConnectionError):
                await flaky_call()
            await fallback()

        execution = wf.get_execution()
        assert len(execution.records) == 2
        assert execution.records[0].outcome == ExecutionOutcome.FAILURE
        assert execution.records[1].outcome == ExecutionOutcome.SUCCESS

    @pytest.mark.asyncio()
    async def test_mixed_decorator_and_context_manager(self) -> None:
        @track_operation("invoke_api")
        async def api_call():
            return "ok"

        async with workflow_context("mixed") as wf:
            async with wf.operation("authenticate"):
                pass
            result = await api_call()
            async with wf.operation("send_email"):
                pass

        assert result == "ok"
        dag = wf.get_dag()
        assert len(dag.nodes) == 3
        assert len(dag.edges) == 2
        ops = [n.operation for n in dag.nodes]
        assert ops == ["authenticate", "invoke_api", "send_email"]


# ---------------------------------------------------------------------------
# AC: Covers storage persistence
# ---------------------------------------------------------------------------


class TestStoragePersistenceE2E:
    @pytest.mark.asyncio()
    async def test_auto_persist_full_pipeline(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        """Full pipeline: instrument → auto-persist → retrieve from storage."""
        async with workflow_context("e2e-storage", recorder=recorder) as wf:
            async with wf.operation("authenticate"):
                pass
            async with wf.operation("read_file"):
                pass
            async with wf.operation("delete_record"):
                pass

        exec_id = wf.get_execution().id
        got = repo.get_execution(exec_id)

        # DAG preserved
        assert len(got.dag.nodes) == 3
        assert len(got.dag.edges) == 2
        ops = [n.operation for n in got.dag.nodes]
        assert ops == ["authenticate", "read_file", "delete_record"]

        # Risk populated — match against live scoring
        live_risk = wf.get_current_risk()
        assert got.predicted_risk == pytest.approx(live_risk.aggregate_score)

        # Outcome derived
        assert got.actual_outcome == ExecutionOutcome.SUCCESS

        # Records preserved
        assert len(got.records) == 3
        assert all(r.outcome == ExecutionOutcome.SUCCESS for r in got.records)

    @pytest.mark.asyncio()
    async def test_persisted_risk_matches_live_scoring(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        """predicted_risk in storage matches get_current_risk() at exit."""
        async with workflow_context("risk-check", recorder=recorder) as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
            live_risk = wf.get_current_risk()

        got = repo.get_execution(wf.get_execution().id)
        assert got.predicted_risk == pytest.approx(live_risk.aggregate_score)

    @pytest.mark.asyncio()
    async def test_persisted_failure_outcome(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        async with workflow_context("fail-store", recorder=recorder) as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api") as op:
                op.record_failure("network error")

        got = repo.get_execution(wf.get_execution().id)
        assert got.actual_outcome == ExecutionOutcome.FAILURE

    @pytest.mark.asyncio()
    async def test_decorator_with_persistence(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        """Decorator + auto-persist: full round-trip."""
        @track_operation("authenticate")
        async def auth():
            pass

        @track_operation("delete_record")
        async def delete():
            pass

        async with workflow_context("dec-store", recorder=recorder) as wf:
            await auth()
            await delete()

        got = repo.get_execution(wf.get_execution().id)
        assert len(got.dag.nodes) == 2
        assert got.predicted_risk == pytest.approx(wf.get_current_risk().aggregate_score)
        assert got.actual_outcome == ExecutionOutcome.SUCCESS


# ---------------------------------------------------------------------------
# Coverage gaps (review fixes)
# ---------------------------------------------------------------------------


class TestCoverageGaps:
    @pytest.mark.asyncio()
    async def test_dag_structure_after_failed_operation(self) -> None:
        """Failed op remains in DAG; subsequent op chains from it."""
        @track_operation("invoke_api")
        async def fail():
            raise ConnectionError("reset")

        @track_operation("send_email")
        async def recover():
            pass

        async with workflow_context("fail-dag") as wf:
            with pytest.raises(ConnectionError):
                await fail()
            await recover()

        dag = wf.get_dag()
        assert len(dag.nodes) == 2
        assert len(dag.edges) == 1
        # The edge goes FROM the failed node TO the recovery node
        assert dag.edges[0].source_id == dag.nodes[0].id
        assert dag.edges[0].target_id == dag.nodes[1].id

    @pytest.mark.asyncio()
    async def test_params_preserved_through_storage(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        """Params survive the full instrument → persist → retrieve pipeline."""
        @track_operation("invoke_api", params={"method": "POST", "endpoint": "/users"})
        async def api_call():
            pass

        async with workflow_context("params-test", recorder=recorder) as wf:
            await api_call()

        got = repo.get_execution(wf.get_execution().id)
        assert got.dag.nodes[0].params == {"method": "POST", "endpoint": "/users"}

    @pytest.mark.asyncio()
    async def test_workflow_name_preserved_through_storage(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        """workflow_name survives the full round-trip through storage."""
        async with workflow_context("e2e-name-check", recorder=recorder) as wf:
            async with wf.operation("read_file"):
                pass

        got = repo.get_execution(wf.get_execution().id)
        assert got.workflow_name == "e2e-name-check"
