"""Tests for @track_operation decorator (NOD-33).

AC:
- [x] Decorated async function within a workflow_context adds a node to the DAG
- [x] Exception in decorated function records failure outcome
"""

import pytest

from workflow_eval.instrumentation.sdk import track_operation, workflow_context
from workflow_eval.types import ExecutionOutcome


# ---------------------------------------------------------------------------
# AC: Decorated async function within a workflow_context adds a node to DAG
# ---------------------------------------------------------------------------


class TestDecoratorAddsNode:
    @pytest.mark.asyncio()
    async def test_single_decorated_function_adds_node(self) -> None:
        @track_operation("read_file")
        async def do_read():
            return "data"

        async with workflow_context("test") as wf:
            result = await do_read()
        assert result == "data"
        dag = wf.get_dag()
        assert len(dag.nodes) == 1
        assert dag.nodes[0].operation == "read_file"

    @pytest.mark.asyncio()
    async def test_multiple_decorated_functions_chain(self) -> None:
        @track_operation("read_file")
        async def step1():
            pass

        @track_operation("invoke_api")
        async def step2():
            pass

        @track_operation("send_email")
        async def step3():
            pass

        async with workflow_context("test") as wf:
            await step1()
            await step2()
            await step3()
        dag = wf.get_dag()
        assert len(dag.nodes) == 3
        assert len(dag.edges) == 2

    @pytest.mark.asyncio()
    async def test_params_forwarded(self) -> None:
        @track_operation("invoke_api", params={"endpoint": "/users"})
        async def call_api():
            pass

        async with workflow_context("test") as wf:
            await call_api()
        dag = wf.get_dag()
        assert dag.nodes[0].params == {"endpoint": "/users"}

    @pytest.mark.asyncio()
    async def test_records_success_on_return(self) -> None:
        @track_operation("read_file")
        async def do_read():
            return 42

        async with workflow_context("test") as wf:
            await do_read()
        execution = wf.get_execution()
        assert execution.records[0].outcome == ExecutionOutcome.SUCCESS

    @pytest.mark.asyncio()
    async def test_mixed_with_context_manager_ops(self) -> None:
        """Decorator and context manager operations interleave correctly."""
        @track_operation("invoke_api")
        async def api_call():
            pass

        async with workflow_context("test") as wf:
            async with wf.operation("read_file"):
                pass
            await api_call()
            async with wf.operation("send_email"):
                pass
        dag = wf.get_dag()
        assert len(dag.nodes) == 3
        assert len(dag.edges) == 2


# ---------------------------------------------------------------------------
# AC: Exception in decorated function records failure outcome
# ---------------------------------------------------------------------------


class TestDecoratorFailure:
    @pytest.mark.asyncio()
    async def test_exception_records_failure(self) -> None:
        @track_operation("invoke_api")
        async def bad_call():
            raise ValueError("connection refused")

        async with workflow_context("test") as wf:
            with pytest.raises(ValueError, match="connection refused"):
                await bad_call()
        execution = wf.get_execution()
        assert execution.records[0].outcome == ExecutionOutcome.FAILURE
        assert "connection refused" in execution.records[0].error

    @pytest.mark.asyncio()
    async def test_exception_propagates(self) -> None:
        @track_operation("delete_record")
        async def fail():
            raise RuntimeError("boom")

        async with workflow_context("test") as wf:
            with pytest.raises(RuntimeError, match="boom"):
                await fail()

    @pytest.mark.asyncio()
    async def test_failure_still_adds_node(self) -> None:
        @track_operation("invoke_api")
        async def fail():
            raise RuntimeError("oops")

        async with workflow_context("test") as wf:
            with pytest.raises(RuntimeError):
                await fail()
        dag = wf.get_dag()
        assert len(dag.nodes) == 1

    @pytest.mark.asyncio()
    async def test_partial_failure_records_both(self) -> None:
        """First op succeeds, second fails — both recorded."""
        @track_operation("read_file")
        async def good():
            pass

        @track_operation("delete_record")
        async def bad():
            raise RuntimeError("denied")

        async with workflow_context("test") as wf:
            await good()
            with pytest.raises(RuntimeError):
                await bad()
        execution = wf.get_execution()
        assert len(execution.records) == 2
        assert execution.records[0].outcome == ExecutionOutcome.SUCCESS
        assert execution.records[1].outcome == ExecutionOutcome.FAILURE


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio()
    async def test_outside_context_raises(self) -> None:
        @track_operation("read_file")
        async def orphan():
            pass

        with pytest.raises(RuntimeError, match="outside of a workflow_context"):
            await orphan()

    @pytest.mark.asyncio()
    async def test_preserves_function_name(self) -> None:
        @track_operation("read_file")
        async def my_function():
            """My docstring."""

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    @pytest.mark.asyncio()
    async def test_return_value_preserved(self) -> None:
        @track_operation("read_file")
        async def compute():
            return {"key": "value"}

        async with workflow_context("test") as wf:
            result = await compute()
        assert result == {"key": "value"}
