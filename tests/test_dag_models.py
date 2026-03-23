"""Tests for DAG model networkx conversion (NOD-11).

AC:
- [x] Construct DAG from Pydantic models (nodes + edges lists)
- [x] to_networkx() produces valid nx.DiGraph with all metadata
- [x] from_networkx() reconstructs equivalent WorkflowDAG
- [x] JSON round-trip: model_dump_json() -> model_validate_json()
- [x] Node IDs are unique within a DAG (validated)
"""

import networkx as nx
import pytest
from pydantic import ValidationError

from workflow_eval.dag.models import from_networkx, to_networkx, validate_unique_node_ids
from workflow_eval.types import DAGEdge, DAGNode, EdgeType, WorkflowDAG


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _linear_dag() -> WorkflowDAG:
    """A -> B -> C linear pipeline."""
    return WorkflowDAG(
        name="linear",
        nodes=(
            DAGNode(id="a", operation="read_file", params={"path": "/tmp/x"}),
            DAGNode(id="b", operation="write_database", params={"table": "t"}),
            DAGNode(id="c", operation="send_email"),
        ),
        edges=(
            DAGEdge(source="a", target="b", edge_type=EdgeType.DATA_FLOW),
            DAGEdge(source="b", target="c"),
        ),
        metadata={"version": 1},
    )


def _single_node_dag() -> WorkflowDAG:
    return WorkflowDAG(name="single", nodes=(DAGNode(id="x", operation="read_file"),), edges=())


def _empty_dag() -> WorkflowDAG:
    return WorkflowDAG(name="empty", nodes=(), edges=())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestWorkflowDAGConstruction:
    def test_construct_from_pydantic(self) -> None:
        dag = _linear_dag()
        assert len(dag.nodes) == 3
        assert len(dag.edges) == 2
        assert dag.name == "linear"

    def test_single_node(self) -> None:
        dag = _single_node_dag()
        assert len(dag.nodes) == 1
        assert len(dag.edges) == 0

    def test_empty_dag(self) -> None:
        dag = _empty_dag()
        assert len(dag.nodes) == 0
        assert len(dag.edges) == 0


# ---------------------------------------------------------------------------
# Node ID uniqueness validation
# ---------------------------------------------------------------------------


class TestNodeIdUniqueness:
    def test_duplicate_ids_rejected_at_construction(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate node IDs"):
            WorkflowDAG(
                name="bad",
                nodes=(
                    DAGNode(id="a", operation="read_file"),
                    DAGNode(id="a", operation="write_file"),
                ),
                edges=(),
            )

    def test_validate_unique_node_ids_helper_happy(self) -> None:
        validate_unique_node_ids(_linear_dag())  # should not raise

    def test_validate_unique_node_ids_helper_raises(self) -> None:
        """Build a DAG bypassing the model validator to test the standalone helper."""
        # Use object.__setattr__ to bypass frozen model and inject duplicate nodes
        dag = _single_node_dag()
        dupe_nodes = (
            DAGNode(id="z", operation="read_file"),
            DAGNode(id="z", operation="write_file"),
        )
        # Construct via model_construct to skip validator
        bad_dag = WorkflowDAG.model_construct(
            name="bad", nodes=dupe_nodes, edges=(), metadata={},
        )
        with pytest.raises(ValueError, match="Duplicate node ID"):
            validate_unique_node_ids(bad_dag)


# ---------------------------------------------------------------------------
# to_networkx()
# ---------------------------------------------------------------------------


class TestToNetworkx:
    def test_produces_digraph(self) -> None:
        g = to_networkx(_linear_dag())
        assert isinstance(g, nx.DiGraph)

    def test_node_count(self) -> None:
        g = to_networkx(_linear_dag())
        assert g.number_of_nodes() == 3

    def test_edge_count(self) -> None:
        g = to_networkx(_linear_dag())
        assert g.number_of_edges() == 2

    def test_node_attributes(self) -> None:
        g = to_networkx(_linear_dag())
        assert g.nodes["a"]["operation"] == "read_file"
        assert g.nodes["a"]["params"] == {"path": "/tmp/x"}
        assert g.nodes["b"]["operation"] == "write_database"
        assert g.nodes["c"]["operation"] == "send_email"

    def test_edge_attributes(self) -> None:
        g = to_networkx(_linear_dag())
        assert g.edges["a", "b"]["edge_type"] == "data_flow"
        assert g.edges["b", "c"]["edge_type"] == "control_flow"

    def test_graph_attributes(self) -> None:
        g = to_networkx(_linear_dag())
        assert g.graph["name"] == "linear"
        assert g.graph["metadata"] == {"version": 1}

    def test_empty_dag(self) -> None:
        g = to_networkx(_empty_dag())
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_single_node(self) -> None:
        g = to_networkx(_single_node_dag())
        assert g.number_of_nodes() == 1
        assert g.number_of_edges() == 0


# ---------------------------------------------------------------------------
# from_networkx()
# ---------------------------------------------------------------------------


class TestFromNetworkx:
    def test_round_trip(self) -> None:
        original = _linear_dag()
        g = to_networkx(original)
        restored = from_networkx(g)
        assert restored.name == original.name
        assert restored.metadata == original.metadata
        assert len(restored.nodes) == len(original.nodes)
        assert len(restored.edges) == len(original.edges)

    def test_node_data_preserved(self) -> None:
        original = _linear_dag()
        restored = from_networkx(to_networkx(original))
        orig_by_id = {n.id: n for n in original.nodes}
        for node in restored.nodes:
            assert node.operation == orig_by_id[node.id].operation
            assert node.params == orig_by_id[node.id].params

    def test_edge_data_preserved(self) -> None:
        original = _linear_dag()
        restored = from_networkx(to_networkx(original))
        orig_edges = {(e.source, e.target): e for e in original.edges}
        for edge in restored.edges:
            orig = orig_edges[(edge.source, edge.target)]
            assert edge.edge_type == orig.edge_type

    def test_empty_dag_round_trip(self) -> None:
        restored = from_networkx(to_networkx(_empty_dag()))
        assert restored.name == "empty"
        assert len(restored.nodes) == 0

    def test_from_manually_built_graph(self) -> None:
        g = nx.DiGraph(name="manual", metadata={})
        g.add_node("n1", operation="invoke_api", params={"url": "http://x"})
        g.add_node("n2", operation="write_file", params={})
        g.add_edge("n1", "n2", edge_type="data_flow")

        dag = from_networkx(g)
        assert dag.name == "manual"
        assert len(dag.nodes) == 2
        assert len(dag.edges) == 1
        assert dag.edges[0].edge_type == EdgeType.DATA_FLOW

    def test_missing_params_defaults_to_empty(self) -> None:
        g = nx.DiGraph(name="minimal")
        g.add_node("n1", operation="read_file")
        dag = from_networkx(g)
        assert dag.nodes[0].params == {}

    def test_missing_edge_type_defaults_to_control_flow(self) -> None:
        g = nx.DiGraph(name="no_type")
        g.add_node("n1", operation="read_file")
        g.add_node("n2", operation="write_file")
        g.add_edge("n1", "n2")
        dag = from_networkx(g)
        assert dag.edges[0].edge_type == EdgeType.CONTROL_FLOW

    def test_node_missing_operation_raises(self) -> None:
        """A node without an 'operation' attribute should raise KeyError."""
        g = nx.DiGraph(name="bad")
        g.add_node("n1")  # no operation attribute
        with pytest.raises(KeyError):
            from_networkx(g)

    def test_invalid_edge_type_raises(self) -> None:
        """An edge with an unrecognised edge_type string should raise ValueError."""
        g = nx.DiGraph(name="bad_edge")
        g.add_node("n1", operation="read_file")
        g.add_node("n2", operation="write_file")
        g.add_edge("n1", "n2", edge_type="bogus")
        with pytest.raises(ValueError):
            from_networkx(g)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_linear_dag(self) -> None:
        original = _linear_dag()
        restored = WorkflowDAG.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_empty_dag(self) -> None:
        original = _empty_dag()
        restored = WorkflowDAG.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_single_node(self) -> None:
        original = _single_node_dag()
        restored = WorkflowDAG.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_full_cycle_pydantic_nx_pydantic_json(self) -> None:
        """Pydantic -> networkx -> Pydantic -> JSON -> Pydantic."""
        original = _linear_dag()
        via_nx = from_networkx(to_networkx(original))
        via_json = WorkflowDAG.model_validate_json(via_nx.model_dump_json())

        assert via_json.name == original.name
        assert via_json.metadata == original.metadata
        # Compare nodes by id
        orig_by_id = {n.id: n for n in original.nodes}
        for node in via_json.nodes:
            assert node.id in orig_by_id
            assert node.operation == orig_by_id[node.id].operation
            assert node.params == orig_by_id[node.id].params
        # Compare edges
        orig_edges = {(e.source, e.target): e for e in original.edges}
        for edge in via_json.edges:
            orig = orig_edges[(edge.source, edge.target)]
            assert edge.edge_type == orig.edge_type
