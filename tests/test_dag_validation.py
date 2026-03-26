"""Tests for DAG validation (NOD-14).

AC:
- [x] Cyclic graph returns cycle warning (level=warning), not error
- [x] Orphan node flagged with relevant node ID
- [x] Unknown operation name flagged as error
- [x] Valid DAG returns empty issue list
- [x] Multiple issues can be returned simultaneously
"""

import pytest

from workflow_eval.dag.validation import ValidationIssue, ValidationLevel, validate_dag
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import DAGEdge, DAGNode, EdgeType, WorkflowDAG


@pytest.fixture()
def registry() -> OperationRegistry:
    return get_default_registry()


def _make_dag(
    name: str = "test",
    nodes: tuple[DAGNode, ...] = (),
    edges: tuple[DAGEdge, ...] = (),
) -> WorkflowDAG:
    return WorkflowDAG(name=name, nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# AC: Valid DAG returns empty issue list
# ---------------------------------------------------------------------------


class TestValidDAG:
    def test_linear_chain_valid(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(
                DAGNode(id="a", operation="read_file"),
                DAGNode(id="b", operation="write_database"),
            ),
            edges=(DAGEdge(source_id="a", target_id="b"),),
        )
        assert validate_dag(dag, registry) == []

    def test_empty_dag_valid(self, registry: OperationRegistry) -> None:
        dag = _make_dag(nodes=(), edges=())
        assert validate_dag(dag, registry) == []

    def test_single_node_valid(self, registry: OperationRegistry) -> None:
        dag = _make_dag(nodes=(DAGNode(id="a", operation="read_file"),))
        assert validate_dag(dag, registry) == []

    def test_fan_out_valid(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(
                DAGNode(id="root", operation="authenticate"),
                DAGNode(id="a", operation="delete_record"),
                DAGNode(id="b", operation="delete_file"),
                DAGNode(id="end", operation="send_notification"),
            ),
            edges=(
                DAGEdge(source_id="root", target_id="a"),
                DAGEdge(source_id="root", target_id="b"),
                DAGEdge(source_id="a", target_id="end"),
                DAGEdge(source_id="b", target_id="end"),
            ),
        )
        assert validate_dag(dag, registry) == []


# ---------------------------------------------------------------------------
# AC: Cyclic graph returns cycle warning (level=warning), not error
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_simple_cycle_is_warning(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(
                DAGNode(id="a", operation="read_file"),
                DAGNode(id="b", operation="write_file"),
            ),
            edges=(
                DAGEdge(source_id="a", target_id="b"),
                DAGEdge(source_id="b", target_id="a"),
            ),
        )
        issues = validate_dag(dag, registry)
        cycle_issues = [i for i in issues if i.code == "cycle_detected"]
        assert len(cycle_issues) == 1
        assert cycle_issues[0].level == ValidationLevel.WARNING

    def test_cycle_includes_node_ids(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(
                DAGNode(id="a", operation="read_file"),
                DAGNode(id="b", operation="write_file"),
                DAGNode(id="c", operation="read_database"),
            ),
            edges=(
                DAGEdge(source_id="a", target_id="b"),
                DAGEdge(source_id="b", target_id="c"),
                DAGEdge(source_id="c", target_id="a"),
            ),
        )
        issues = validate_dag(dag, registry)
        cycle_issues = [i for i in issues if i.code == "cycle_detected"]
        assert len(cycle_issues) == 1
        assert set(cycle_issues[0].node_ids) == {"a", "b", "c"}

    def test_self_loop_is_warning(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(DAGNode(id="a", operation="read_file"),),
            edges=(DAGEdge(source_id="a", target_id="a"),),
        )
        issues = validate_dag(dag, registry)
        cycle_issues = [i for i in issues if i.code == "cycle_detected"]
        assert len(cycle_issues) == 1
        assert cycle_issues[0].level == ValidationLevel.WARNING

    def test_phantom_node_cycle_from_dangling_edges(self, registry: OperationRegistry) -> None:
        """Dangling edges that form a cycle via phantom nodes should still detect the cycle.

        to_networkx auto-creates 'ghost' from the edge, forming a→ghost→a cycle.
        Both dangling_edge and cycle_detected should fire.
        """
        dag = _make_dag(
            nodes=(DAGNode(id="a", operation="read_file"),),
            edges=(
                DAGEdge(source_id="a", target_id="ghost"),
                DAGEdge(source_id="ghost", target_id="a"),
            ),
        )
        issues = validate_dag(dag, registry)
        codes = {i.code for i in issues}
        assert "dangling_edge" in codes
        assert "cycle_detected" in codes

    def test_acyclic_dag_no_cycle_warning(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(
                DAGNode(id="a", operation="read_file"),
                DAGNode(id="b", operation="write_file"),
            ),
            edges=(DAGEdge(source_id="a", target_id="b"),),
        )
        issues = validate_dag(dag, registry)
        assert not any(i.code == "cycle_detected" for i in issues)


# ---------------------------------------------------------------------------
# AC: Orphan node flagged with relevant node ID
# ---------------------------------------------------------------------------


class TestOrphanNodes:
    def test_orphan_flagged(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(
                DAGNode(id="a", operation="read_file"),
                DAGNode(id="b", operation="write_file"),
                DAGNode(id="orphan", operation="read_database"),
            ),
            edges=(DAGEdge(source_id="a", target_id="b"),),
        )
        issues = validate_dag(dag, registry)
        orphan_issues = [i for i in issues if i.code == "orphan_node"]
        assert len(orphan_issues) == 1
        assert orphan_issues[0].node_ids == ("orphan",)
        assert orphan_issues[0].level == ValidationLevel.WARNING

    def test_root_node_not_orphan(self, registry: OperationRegistry) -> None:
        """Root has outgoing edges — not an orphan."""
        dag = _make_dag(
            nodes=(
                DAGNode(id="root", operation="read_file"),
                DAGNode(id="leaf", operation="write_file"),
            ),
            edges=(DAGEdge(source_id="root", target_id="leaf"),),
        )
        issues = validate_dag(dag, registry)
        assert not any(i.code == "orphan_node" for i in issues)

    def test_leaf_node_not_orphan(self, registry: OperationRegistry) -> None:
        """Leaf has incoming edges — not an orphan."""
        dag = _make_dag(
            nodes=(
                DAGNode(id="a", operation="read_file"),
                DAGNode(id="leaf", operation="write_file"),
            ),
            edges=(DAGEdge(source_id="a", target_id="leaf"),),
        )
        issues = validate_dag(dag, registry)
        assert not any(i.code == "orphan_node" for i in issues)

    def test_multiple_orphans_each_reported(self, registry: OperationRegistry) -> None:
        """Multiple isolated nodes should each get their own orphan_node issue."""
        dag = _make_dag(
            nodes=(
                DAGNode(id="a", operation="read_file"),
                DAGNode(id="b", operation="write_file"),
                DAGNode(id="x", operation="read_database"),
                DAGNode(id="y", operation="send_notification"),
            ),
            edges=(DAGEdge(source_id="a", target_id="b"),),
        )
        issues = validate_dag(dag, registry)
        orphan_issues = [i for i in issues if i.code == "orphan_node"]
        assert len(orphan_issues) == 2
        orphan_ids = {i.node_ids[0] for i in orphan_issues}
        assert orphan_ids == {"x", "y"}

    def test_single_node_not_orphan(self, registry: OperationRegistry) -> None:
        """A single-node DAG is valid, not an orphan."""
        dag = _make_dag(nodes=(DAGNode(id="a", operation="read_file"),))
        issues = validate_dag(dag, registry)
        assert not any(i.code == "orphan_node" for i in issues)


# ---------------------------------------------------------------------------
# AC: Unknown operation name flagged as error
# ---------------------------------------------------------------------------


class TestOperationResolution:
    def test_unknown_operation_flagged(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(DAGNode(id="a", operation="nonexistent_op"),),
        )
        issues = validate_dag(dag, registry)
        op_issues = [i for i in issues if i.code == "unknown_operation"]
        assert len(op_issues) == 1
        assert op_issues[0].level == ValidationLevel.ERROR
        assert op_issues[0].node_ids == ("a",)
        assert "nonexistent_op" in op_issues[0].message

    def test_known_operation_no_error(self, registry: OperationRegistry) -> None:
        dag = _make_dag(nodes=(DAGNode(id="a", operation="read_file"),))
        issues = validate_dag(dag, registry)
        assert not any(i.code == "unknown_operation" for i in issues)


# ---------------------------------------------------------------------------
# Edge reference integrity (deferred from NOD-11, check #4)
# ---------------------------------------------------------------------------


class TestEdgeIntegrity:
    def test_dangling_target_flagged(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(DAGNode(id="a", operation="read_file"),),
            edges=(DAGEdge(source_id="a", target_id="ghost"),),
        )
        issues = validate_dag(dag, registry)
        edge_issues = [i for i in issues if i.code == "dangling_edge"]
        assert len(edge_issues) == 1
        assert edge_issues[0].level == ValidationLevel.ERROR
        assert "ghost" in edge_issues[0].message

    def test_dangling_source_flagged(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(DAGNode(id="b", operation="write_file"),),
            edges=(DAGEdge(source_id="ghost", target_id="b"),),
        )
        issues = validate_dag(dag, registry)
        edge_issues = [i for i in issues if i.code == "dangling_edge"]
        assert len(edge_issues) == 1
        assert edge_issues[0].level == ValidationLevel.ERROR
        assert "ghost" in edge_issues[0].message

    def test_valid_edges_no_error(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(
                DAGNode(id="a", operation="read_file"),
                DAGNode(id="b", operation="write_file"),
            ),
            edges=(DAGEdge(source_id="a", target_id="b"),),
        )
        issues = validate_dag(dag, registry)
        assert not any(i.code == "dangling_edge" for i in issues)


# ---------------------------------------------------------------------------
# Root detection (check #5)
# ---------------------------------------------------------------------------


class TestRootDetection:
    def test_no_root_flagged(self, registry: OperationRegistry) -> None:
        """All nodes have incoming edges — no root."""
        dag = _make_dag(
            nodes=(
                DAGNode(id="a", operation="read_file"),
                DAGNode(id="b", operation="write_file"),
            ),
            edges=(
                DAGEdge(source_id="a", target_id="b"),
                DAGEdge(source_id="b", target_id="a"),
            ),
        )
        issues = validate_dag(dag, registry)
        root_issues = [i for i in issues if i.code == "no_root"]
        assert len(root_issues) == 1
        assert root_issues[0].level == ValidationLevel.WARNING
        assert root_issues[0].node_ids == ()

    def test_has_root_no_warning(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(
                DAGNode(id="a", operation="read_file"),
                DAGNode(id="b", operation="write_file"),
            ),
            edges=(DAGEdge(source_id="a", target_id="b"),),
        )
        issues = validate_dag(dag, registry)
        assert not any(i.code == "no_root" for i in issues)


# ---------------------------------------------------------------------------
# AC: Multiple issues can be returned simultaneously
# ---------------------------------------------------------------------------


class TestMultipleIssues:
    def test_multiple_issues_returned(self, registry: OperationRegistry) -> None:
        """DAG with unknown op + dangling edge + 2 orphans returns all four."""
        dag = _make_dag(
            nodes=(
                DAGNode(id="a", operation="nonexistent_op"),
                DAGNode(id="b", operation="read_file"),
                DAGNode(id="orphan", operation="read_database"),
            ),
            edges=(DAGEdge(source_id="a", target_id="ghost"),),
        )
        issues = validate_dag(dag, registry)
        codes = [i.code for i in issues]
        assert codes.count("unknown_operation") == 1
        assert codes.count("dangling_edge") == 1
        assert codes.count("orphan_node") == 2  # "b" and "orphan"
        assert len(issues) == 4

    def test_multiple_unknown_ops(self, registry: OperationRegistry) -> None:
        dag = _make_dag(
            nodes=(
                DAGNode(id="a", operation="fake_op_1"),
                DAGNode(id="b", operation="fake_op_2"),
            ),
            edges=(DAGEdge(source_id="a", target_id="b"),),
        )
        issues = validate_dag(dag, registry)
        op_issues = [i for i in issues if i.code == "unknown_operation"]
        assert len(op_issues) == 2


# ---------------------------------------------------------------------------
# ValidationIssue model
# ---------------------------------------------------------------------------


class TestValidationIssueModel:
    def test_frozen(self) -> None:
        from pydantic import ValidationError

        issue = ValidationIssue(
            level=ValidationLevel.ERROR,
            code="test",
            message="msg",
            node_ids=("a",),
        )
        with pytest.raises(ValidationError, match="frozen"):
            issue.code = "changed"  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
        issue = ValidationIssue(
            level=ValidationLevel.WARNING,
            code="cycle_detected",
            message="cycle",
            node_ids=("a", "b"),
        )
        restored = ValidationIssue.model_validate_json(issue.model_dump_json())
        assert restored == issue
