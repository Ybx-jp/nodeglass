"""Fluent DAG construction API (NOD-12).

AC:
- [x] Builder produces valid WorkflowDAG
- [x] then() creates control_flow edge from previous step
- [x] parallel() fans out from current to multiple steps
- [x] join() converges parallel branches
- [x] build() on empty builder raises ValueError

Design decisions (agreed with user):
- parallel() accumulates pending heads from a single cursor; chaining
  .parallel(...).parallel(...) fans all branches from the same predecessor.
- then() after parallel() is an error (ambiguous); must join() first.
- join() with no pending parallel heads acts like then() from the cursor.
- All methods accept an optional edge_type (default control_flow). The builder
  has no ontology coupling — callers specify edge semantics explicitly.
"""

from __future__ import annotations

from typing import Any

from workflow_eval.types import DAGEdge, DAGNode, EdgeType, WorkflowDAG


class DAGBuilder:
    """Fluent builder for WorkflowDAG construction.

    Usage::

        dag = (DAGBuilder("my-workflow")
            .add_step("auth", "authenticate", params={"method": "jwt"})
            .then("lookup", "read_database", params={"table": "users"})
            .parallel(["delete_posts", "delete_files"], operation="delete_record")
            .join("notify", "send_notification")
            .build())
    """

    def __init__(self, name: str, *, metadata: dict[str, Any] | None = None) -> None:
        self._name = name
        self._metadata: dict[str, Any] = metadata or {}
        self._nodes: list[DAGNode] = []
        self._edges: list[DAGEdge] = []
        self._node_ids: set[str] = set()
        # Single cursor: the node that then()/parallel() fan out from.
        self._cursor: str | None = None
        # Pending parallel heads: accumulated by parallel(), consumed by join().
        self._parallel_heads: list[str] = []

    def _add_node(self, node_id: str, operation: str, params: dict[str, Any]) -> None:
        if node_id in self._node_ids:
            raise ValueError(f"Duplicate node ID: {node_id!r}")
        node = DAGNode(id=node_id, operation=operation, params=params)
        self._nodes.append(node)
        self._node_ids.add(node_id)

    def _add_edge(self, source: str, target: str, edge_type: EdgeType) -> None:
        self._edges.append(DAGEdge(source=source, target=target, edge_type=edge_type))

    def add_step(
        self,
        node_id: str,
        operation: str,
        *,
        params: dict[str, Any] | None = None,
        edge_type: EdgeType = EdgeType.CONTROL_FLOW,
    ) -> DAGBuilder:
        """Add a node. If a cursor exists, creates an edge from it.

        First call sets the root node (no edge).
        """
        self._add_node(node_id, operation, params or {})
        if self._cursor is not None:
            if self._parallel_heads:
                raise ValueError(
                    "Cannot add_step() with pending parallel branches; call join() first"
                )
            self._add_edge(self._cursor, node_id, edge_type)
        self._cursor = node_id
        self._parallel_heads = []
        return self

    def then(
        self,
        node_id: str,
        operation: str,
        *,
        params: dict[str, Any] | None = None,
        edge_type: EdgeType = EdgeType.CONTROL_FLOW,
    ) -> DAGBuilder:
        """Add a node with an edge from the previous step.

        Raises ValueError if no preceding step or if parallel branches are pending.
        """
        if self._cursor is None:
            raise ValueError("then() requires a preceding add_step()")
        if self._parallel_heads:
            raise ValueError(
                "Cannot then() with pending parallel branches; call join() first"
            )
        self._add_node(node_id, operation, params or {})
        self._add_edge(self._cursor, node_id, edge_type)
        self._cursor = node_id
        return self

    def parallel(
        self,
        node_ids: list[str],
        operation: str,
        *,
        params: dict[str, Any] | None = None,
        edge_type: EdgeType = EdgeType.CONTROL_FLOW,
    ) -> DAGBuilder:
        """Fan out from the current cursor to multiple nodes.

        All nodes share the same operation and params. Chaining multiple
        parallel() calls accumulates branches from the same predecessor.

        Raises ValueError if no cursor exists.
        """
        if self._cursor is None:
            raise ValueError("parallel() requires a preceding add_step()")
        resolved_params = params or {}
        for nid in node_ids:
            self._add_node(nid, operation, resolved_params)
            self._add_edge(self._cursor, nid, edge_type)
            self._parallel_heads.append(nid)
        return self

    def join(
        self,
        node_id: str,
        operation: str,
        *,
        params: dict[str, Any] | None = None,
        edge_type: EdgeType = EdgeType.CONTROL_FLOW,
    ) -> DAGBuilder:
        """Converge parallel branches into a single node.

        Creates edges from all pending parallel heads to the new node.
        If no parallel heads are pending, acts like then() from the cursor.
        """
        if self._cursor is None:
            raise ValueError("join() requires a preceding step")
        self._add_node(node_id, operation, params or {})
        if self._parallel_heads:
            for head in self._parallel_heads:
                self._add_edge(head, node_id, edge_type)
            self._parallel_heads = []
        else:
            self._add_edge(self._cursor, node_id, edge_type)
        self._cursor = node_id
        return self

    def build(self) -> WorkflowDAG:
        """Build and return the WorkflowDAG.

        Raises ValueError if no nodes have been added.
        """
        if not self._nodes:
            raise ValueError("Cannot build an empty DAG; add at least one step")
        return WorkflowDAG(
            name=self._name,
            nodes=tuple(self._nodes),
            edges=tuple(self._edges),
            metadata=self._metadata,
        )
