"""DAG validation — cycle detection, orphan nodes, operation resolution (NOD-14).

NOD-14 spec (Linear):
- validate_dag(dag: WorkflowDAG, registry: OperationRegistry) -> list[ValidationIssue]
- Checks:
  1. Cycle detection — flags cycles as warnings (does NOT reject the DAG)
  2. Orphan nodes — nodes with no incoming AND no outgoing edges
  3. Operation resolution — every node's operation resolves to a registered OperationDefinition
  4. Edge reference integrity — all source_id/target_id point to existing nodes
  5. Root detection — at least one node with no incoming edges
- ValidationIssue model: level (warning/error), code, message, node_ids

AC:
- [x] Cyclic graph returns cycle warning (level=warning), not error
- [x] Orphan node flagged with relevant node ID
- [x] Unknown operation name flagged as error
- [x] Valid DAG returns empty issue list
- [x] Multiple issues can be returned simultaneously
"""

from __future__ import annotations

from enum import StrEnum

import networkx as nx
from pydantic import BaseModel, ConfigDict

from workflow_eval.dag.models import to_networkx
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import WorkflowDAG


class ValidationLevel(StrEnum):
    """Severity of a validation issue."""

    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(BaseModel):
    """A single validation finding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: ValidationLevel
    code: str
    message: str
    node_ids: tuple[str, ...] = ()


def validate_dag(
    dag: WorkflowDAG, registry: OperationRegistry
) -> list[ValidationIssue]:
    """Run all validation checks on a WorkflowDAG.

    Returns a list of issues (may be empty if the DAG is valid).
    Does NOT raise — callers decide how to handle issues.
    """
    issues: list[ValidationIssue] = []

    node_ids = {n.id for n in dag.nodes}

    _check_edge_integrity(dag, node_ids, issues)
    _check_operation_resolution(dag, registry, issues)
    _check_orphan_nodes(dag, node_ids, issues)
    _check_root_detection(dag, node_ids, issues)
    _check_cycles(dag, issues)

    return issues


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_edge_integrity(
    dag: WorkflowDAG, node_ids: set[str], issues: list[ValidationIssue]
) -> None:
    """Check #4: all edge source_id/target_id reference existing nodes."""
    for edge in dag.edges:
        for field, nid in [("source_id", edge.source_id), ("target_id", edge.target_id)]:
            if nid not in node_ids:
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.ERROR,
                        code="dangling_edge",
                        message=f"Edge {field} {nid!r} does not match any node ID",
                        node_ids=(nid,),
                    )
                )


def _check_operation_resolution(
    dag: WorkflowDAG, registry: OperationRegistry, issues: list[ValidationIssue]
) -> None:
    """Check #3: every node's operation resolves to a registered definition."""
    for node in dag.nodes:
        if node.operation not in registry:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    code="unknown_operation",
                    message=f"Node {node.id!r} references unknown operation {node.operation!r}",
                    node_ids=(node.id,),
                )
            )


def _check_orphan_nodes(
    dag: WorkflowDAG, node_ids: set[str], issues: list[ValidationIssue]
) -> None:
    """Check #2: nodes with no incoming AND no outgoing edges."""
    if len(dag.nodes) <= 1:
        return  # single-node DAG is not an orphan

    has_incoming: set[str] = set()
    has_outgoing: set[str] = set()
    for edge in dag.edges:
        has_outgoing.add(edge.source_id)
        has_incoming.add(edge.target_id)

    for nid in node_ids:
        if nid not in has_incoming and nid not in has_outgoing:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="orphan_node",
                    message=f"Node {nid!r} has no incoming or outgoing edges",
                    node_ids=(nid,),
                )
            )


def _check_root_detection(
    dag: WorkflowDAG, node_ids: set[str], issues: list[ValidationIssue]
) -> None:
    """Check #5: at least one node with no incoming edges."""
    if not dag.nodes:
        return  # empty DAG — nothing to flag

    has_incoming = {edge.target_id for edge in dag.edges}
    roots = node_ids - has_incoming

    if not roots:
        issues.append(
            ValidationIssue(
                level=ValidationLevel.WARNING,
                code="no_root",
                message="No root node found (every node has incoming edges)",
                node_ids=(),
            )
        )


def _check_cycles(dag: WorkflowDAG, issues: list[ValidationIssue]) -> None:
    """Check #1: detect cycles and flag as warnings."""
    if not dag.edges:
        return

    g = to_networkx(dag)
    try:
        cycle = nx.find_cycle(g, orientation="original")
        cycle_nodes = tuple(dict.fromkeys(u for u, _v, _dir in cycle))
        issues.append(
            ValidationIssue(
                level=ValidationLevel.WARNING,
                code="cycle_detected",
                message=f"Cycle detected involving nodes: {', '.join(cycle_nodes)}",
                node_ids=cycle_nodes,
            )
        )
    except nx.NetworkXNoCycle:
        pass
