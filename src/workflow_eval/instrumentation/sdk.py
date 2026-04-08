"""Decorators and context managers for runtime workflow tracking (NOD-32, NOD-33).

NOD-32: workflow_context + wf.operation() context managers
NOD-33: @track_operation decorator

AC (NOD-32):
- [x] Three sequential wf.operation() calls produce a 3-node linear DAG
      with 2 control_flow edges
- [x] get_current_risk() returns a valid RiskProfile

AC (NOD-33):
- [x] Decorated async function within a workflow_context adds a node to the DAG
- [x] Exception in decorated function records failure outcome

Behavioral constraints (NOD-33):
- @track_operation(op_name, params=...) decorator for async functions
- Registers with active workflow_context via contextvars
- Records success if function returns, failure if it raises
"""

from __future__ import annotations

import functools
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator, Callable, TypeVar

from workflow_eval.dag.models import to_networkx
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.engine import RiskScoringEngine
from workflow_eval.types import (
    DAGEdge,
    DAGNode,
    EdgeType,
    ExecutionOutcome,
    ExecutionRecord,
    RiskProfile,
    ScoringConfig,
    WorkflowDAG,
    WorkflowExecution,
)


_active_context: ContextVar[WorkflowContext | None] = ContextVar(
    "_active_context", default=None,
)


class OperationHandle:
    """Handle for a single operation within a workflow context."""

    def __init__(self, node_id: str, operation: str) -> None:
        self._node_id = node_id
        self._operation = operation
        self._outcome: ExecutionOutcome | None = None
        self._error: str | None = None

    @property
    def node_id(self) -> str:
        return self._node_id

    def record_success(self) -> None:
        """Mark this operation as successful."""
        self._outcome = ExecutionOutcome.SUCCESS

    def record_failure(self, error: str) -> None:
        """Mark this operation as failed with an error message."""
        self._outcome = ExecutionOutcome.FAILURE
        self._error = error

    def _to_record(self) -> ExecutionRecord:
        return ExecutionRecord(
            node_id=self._node_id,
            operation=self._operation,
            outcome=self._outcome or ExecutionOutcome.SUCCESS,
            error=self._error,
        )


class WorkflowContext:
    """Runtime workflow tracker that auto-builds a DAG from observed operations."""

    def __init__(
        self,
        name: str,
        *,
        registry: OperationRegistry | None = None,
        scoring_config: ScoringConfig | None = None,
    ) -> None:
        self._name = name
        self._registry = registry or get_default_registry()
        self._scoring_config = scoring_config or ScoringConfig()
        self._execution_id = uuid.uuid4().hex
        self._nodes: list[DAGNode] = []
        self._edges: list[DAGEdge] = []
        self._records: list[ExecutionRecord] = []
        self._prev_node_id: str | None = None
        self._counter = 0

    @asynccontextmanager
    async def operation(
        self, op_name: str, *, params: dict[str, Any] | None = None,
    ) -> AsyncIterator[OperationHandle]:
        """Add an operation node to the DAG, linked from the previous node."""
        node_id = f"{op_name}_{self._counter}"
        self._counter += 1

        # Build DAG incrementally: add node + edge from predecessor
        self._nodes.append(
            DAGNode(id=node_id, operation=op_name, params=params or {}),
        )
        if self._prev_node_id is not None:
            self._edges.append(DAGEdge(
                source_id=self._prev_node_id,
                target_id=node_id,
                edge_type=EdgeType.CONTROL_FLOW,
            ))
        self._prev_node_id = node_id

        handle = OperationHandle(node_id, op_name)
        try:
            yield handle
        except Exception as exc:
            if handle._outcome is None:
                handle.record_failure(str(exc))
            raise
        else:
            if handle._outcome is None:
                handle.record_success()
        finally:
            self._records.append(handle._to_record())

    def get_current_risk(self) -> RiskProfile:
        """Score the in-progress DAG and return a RiskProfile."""
        nx_dag = to_networkx(self.get_dag())
        engine = RiskScoringEngine(self._scoring_config, self._registry)
        return engine.score(nx_dag)

    # -- DAG access ------------------------------------------------------------

    def get_dag(self) -> WorkflowDAG:
        """Return the current in-progress DAG."""
        return WorkflowDAG(
            name=self._name,
            nodes=tuple(self._nodes),
            edges=tuple(self._edges),
        )

    def get_execution(self) -> WorkflowExecution:
        """Return a WorkflowExecution snapshot of the current state."""
        return WorkflowExecution(
            id=self._execution_id,
            workflow_name=self._name,
            dag=self.get_dag(),
            records=tuple(self._records),
        )


@asynccontextmanager
async def workflow_context(
    name: str,
    *,
    registry: OperationRegistry | None = None,
    scoring_config: ScoringConfig | None = None,
    recorder: Any | None = None,
) -> AsyncIterator[WorkflowContext]:
    """Top-level async context manager for runtime workflow tracking.

    If a ``WorkflowRecorder`` is provided, the execution is automatically
    persisted to storage on context exit.
    """
    wf = WorkflowContext(name, registry=registry, scoring_config=scoring_config)
    token = _active_context.set(wf)
    try:
        yield wf
    finally:
        _active_context.reset(token)
        if recorder is not None:
            recorder.record(wf)


def track_operation(
    op_name: str, *, params: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Decorator that registers an async function as a tracked operation.

    Usage::

        @track_operation("invoke_api", params={"endpoint": "/users"})
        async def call_users_api():
            ...

    Must be called within an active ``workflow_context``.
    Records success if the function returns, failure if it raises.
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            wf = _active_context.get()
            if wf is None:
                raise RuntimeError(
                    f"@track_operation({op_name!r}) called outside of a workflow_context"
                )
            async with wf.operation(op_name, params=params):
                return await fn(*args, **kwargs)
        return wrapper
    return decorator
