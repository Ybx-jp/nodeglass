"""Tests for workflow_context context manager (NOD-32).

AC:
- [x] Three sequential wf.operation() calls produce a 3-node linear DAG
      with 2 control_flow edges
- [x] get_current_risk() returns a valid RiskProfile
"""

import pytest

from workflow_eval.instrumentation.sdk import WorkflowContext, workflow_context
from workflow_eval.types import (
    EdgeType,
    ExecutionOutcome,
    RiskProfile,
)


# ---------------------------------------------------------------------------
# AC: Three sequential operations produce a 3-node linear DAG with 2 edges
# ---------------------------------------------------------------------------


class TestLinearDAG:
    @pytest.mark.asyncio()
    async def test_three_ops_produce_three_nodes(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
            async with wf.operation("send_email"):
                pass
        dag = wf.get_dag()
        assert len(dag.nodes) == 3

    @pytest.mark.asyncio()
    async def test_three_ops_produce_two_edges(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
            async with wf.operation("send_email"):
                pass
        dag = wf.get_dag()
        assert len(dag.edges) == 2

    @pytest.mark.asyncio()
    async def test_edges_are_control_flow(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
            async with wf.operation("send_email"):
                pass
        dag = wf.get_dag()
        for edge in dag.edges:
            assert edge.edge_type == EdgeType.CONTROL_FLOW

    @pytest.mark.asyncio()
    async def test_edges_chain_sequentially(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
            async with wf.operation("send_email"):
                pass
        dag = wf.get_dag()
        # Edge 0: node 0 -> node 1, Edge 1: node 1 -> node 2
        assert dag.edges[0].source_id == dag.nodes[0].id
        assert dag.edges[0].target_id == dag.nodes[1].id
        assert dag.edges[1].source_id == dag.nodes[1].id
        assert dag.edges[1].target_id == dag.nodes[2].id

    @pytest.mark.asyncio()
    async def test_node_ids_contain_operation_name(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
        dag = wf.get_dag()
        assert "read_file" in dag.nodes[0].id
        assert "invoke_api" in dag.nodes[1].id

    @pytest.mark.asyncio()
    async def test_node_operations_set(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
        dag = wf.get_dag()
        assert dag.nodes[0].operation == "read_file"

    @pytest.mark.asyncio()
    async def test_params_forwarded(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file", params={"path": "/tmp"}):
                pass
        dag = wf.get_dag()
        assert dag.nodes[0].params == {"path": "/tmp"}

    @pytest.mark.asyncio()
    async def test_single_op_no_edges(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
        dag = wf.get_dag()
        assert len(dag.nodes) == 1
        assert len(dag.edges) == 0


# ---------------------------------------------------------------------------
# AC: get_current_risk() returns a valid RiskProfile
# ---------------------------------------------------------------------------


class TestGetCurrentRisk:
    @pytest.mark.asyncio()
    async def test_returns_risk_profile(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
            profile = wf.get_current_risk()
        assert isinstance(profile, RiskProfile)

    @pytest.mark.asyncio()
    async def test_risk_profile_has_correct_name(self) -> None:
        async with workflow_context("my-workflow") as wf:
            async with wf.operation("read_file"):
                pass
            profile = wf.get_current_risk()
        assert profile.workflow_name == "my-workflow"

    @pytest.mark.asyncio()
    async def test_risk_profile_node_count(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
            async with wf.operation("delete_record"):
                pass
            profile = wf.get_current_risk()
        assert profile.node_count == 3
        assert profile.edge_count == 2

    @pytest.mark.asyncio()
    async def test_risk_increases_with_dangerous_ops(self) -> None:
        async with workflow_context("safe") as wf:
            async with wf.operation("read_file"):
                pass
            safe_risk = wf.get_current_risk()

        async with workflow_context("risky") as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("delete_record"):
                pass
            async with wf.operation("destroy_resource"):
                pass
            risky_risk = wf.get_current_risk()
        assert risky_risk.aggregate_score > safe_risk.aggregate_score

    @pytest.mark.asyncio()
    async def test_mid_workflow_scoring(self) -> None:
        """Can score after each operation — risk grows as ops are added."""
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
            risk_1 = wf.get_current_risk()
            assert risk_1.node_count == 1

            async with wf.operation("delete_record"):
                pass
            risk_2 = wf.get_current_risk()
            assert risk_2.node_count == 2


# ---------------------------------------------------------------------------
# Recording outcomes
# ---------------------------------------------------------------------------


class TestOutcomeRecording:
    @pytest.mark.asyncio()
    async def test_auto_success_on_clean_exit(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
        execution = wf.get_execution()
        assert execution.records[0].outcome == ExecutionOutcome.SUCCESS

    @pytest.mark.asyncio()
    async def test_explicit_success(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file") as op:
                op.record_success()
        execution = wf.get_execution()
        assert execution.records[0].outcome == ExecutionOutcome.SUCCESS

    @pytest.mark.asyncio()
    async def test_explicit_failure(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file") as op:
                op.record_failure("disk full")
        execution = wf.get_execution()
        assert execution.records[0].outcome == ExecutionOutcome.FAILURE
        assert execution.records[0].error == "disk full"

    @pytest.mark.asyncio()
    async def test_auto_failure_on_exception(self) -> None:
        async with workflow_context("test") as wf:
            with pytest.raises(RuntimeError, match="boom"):
                async with wf.operation("invoke_api"):
                    raise RuntimeError("boom")
        execution = wf.get_execution()
        assert execution.records[0].outcome == ExecutionOutcome.FAILURE
        assert "boom" in execution.records[0].error

    @pytest.mark.asyncio()
    async def test_handle_exposes_node_id(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file") as op:
                assert "read_file" in op.node_id


# ---------------------------------------------------------------------------
# WorkflowExecution snapshot
# ---------------------------------------------------------------------------


class TestExecutionSnapshot:
    @pytest.mark.asyncio()
    async def test_execution_has_all_records(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
        execution = wf.get_execution()
        assert len(execution.records) == 2
        assert execution.workflow_name == "test"

    @pytest.mark.asyncio()
    async def test_execution_dag_matches(self) -> None:
        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
            async with wf.operation("invoke_api"):
                pass
        execution = wf.get_execution()
        assert execution.dag == wf.get_dag()
