"""WorkflowDAG networkx conversion and validation.

NOD-11 spec (Linear):
- to_networkx() -> nx.DiGraph with operation metadata on nodes, edge_type on edges
- from_networkx(g) -> WorkflowDAG class method
- Node IDs unique within a DAG (validated)
- JSON round-trip preserves structure

AC:
- [x] Construct DAG from Pydantic models (nodes + edges lists)
- [x] to_networkx() produces valid nx.DiGraph with all metadata
- [x] from_networkx() reconstructs equivalent WorkflowDAG
- [x] JSON round-trip: model_dump_json() -> model_validate_json()
- [x] Node IDs are unique within a DAG (validated)
"""

from __future__ import annotations

import networkx as nx

from workflow_eval.types import DAGEdge, DAGNode, EdgeType, WorkflowDAG


def to_networkx(dag: WorkflowDAG) -> nx.DiGraph:
    """Convert a WorkflowDAG to a networkx DiGraph.

    Node attributes: operation, params (from DAGNode).
    Edge attributes: edge_type (from DAGEdge).
    Graph attributes: name, metadata (from WorkflowDAG).
    """
    g = nx.DiGraph(name=dag.name, metadata=dag.metadata)

    for node in dag.nodes:
        g.add_node(
            node.id,
            operation=node.operation,
            params=node.params,
            metadata=node.metadata,
        )

    for edge in dag.edges:
        g.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value,
            condition=edge.condition,
            metadata=edge.metadata,
        )

    return g


def from_networkx(g: nx.DiGraph) -> WorkflowDAG:
    """Reconstruct a WorkflowDAG from a networkx DiGraph.

    Expects node attributes: operation, params (optional).
    Expects edge attributes: edge_type (optional, defaults to control_flow).
    Expects graph attributes: name, metadata (optional).
    """
    nodes = tuple(
        DAGNode(
            id=str(node_id),
            operation=attrs["operation"],
            params=attrs.get("params", {}),
            metadata=attrs.get("metadata", {}),
        )
        for node_id, attrs in g.nodes(data=True)
    )

    edges = tuple(
        DAGEdge(
            source_id=str(src),
            target_id=str(tgt),
            edge_type=EdgeType(attrs.get("edge_type", EdgeType.CONTROL_FLOW.value)),
            condition=attrs.get("condition"),
            metadata=attrs.get("metadata", {}),
        )
        for src, tgt, attrs in g.edges(data=True)
    )

    return WorkflowDAG(
        name=g.graph.get("name", ""),
        nodes=nodes,
        edges=edges,
        metadata=g.graph.get("metadata", {}),
    )


def validate_unique_node_ids(dag: WorkflowDAG) -> None:
    """Raise ValueError if any node IDs are duplicated."""
    seen: set[str] = set()
    for node in dag.nodes:
        if node.id in seen:
            raise ValueError(f"Duplicate node ID: {node.id!r}")
        seen.add(node.id)
