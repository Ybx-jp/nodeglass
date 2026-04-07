"""Decorators and context managers for runtime workflow tracking (NOD-32).

NOD-32 spec (Linear):
- instrumentation/sdk.py
- async with workflow_context(name) as wf: creates a WorkflowExecution
- Nested async with wf.operation(op_name, params=...) as op: adds DAGNode +
  control_flow edge from previous operation
- op.record_success() / op.record_failure(error)
- wf.get_current_risk() runs scoring on in-progress DAG

AC:
- [x] Three sequential wf.operation() calls produce a 3-node linear DAG
      with 2 control_flow edges
- [x] get_current_risk() returns a valid RiskProfile

Behavioral constraints from description:
- operation() is an async context manager yielding an OperationHandle
- Sequential operations create a linear chain of control_flow edges
- get_current_risk() runs scoring on the in-progress DAG
- Auto-records success on clean exit, failure on exception
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

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
            id=uuid.uuid4().hex,
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
) -> AsyncIterator[WorkflowContext]:
    """Top-level async context manager for runtime workflow tracking."""
    wf = WorkflowContext(name, registry=registry, scoring_config=scoring_config)
    yield wf
