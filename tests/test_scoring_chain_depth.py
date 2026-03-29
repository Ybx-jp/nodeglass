"""Tests for chain depth scorer (NOD-19).

AC:
- [x] Pure-only DAG (all `pure` effect_type) -> 0.0
- [x] All-stateful chain of depth 5 in DAG of depth 5 -> 1.0
- [x] Mixed DAG: 2 side-effect nodes in depth-4 DAG -> ~0.5
- [x] Single-node DAG -> 0.0
- [x] Details include the longest side-effect path (node IDs)
"""

import networkx as nx
import pytest

from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.chain_depth import ChainDepthScorer
from workflow_eval.scoring.protocols import Scorer
from workflow_eval.types import SubScore


@pytest.fixture()
def registry() -> OperationRegistry:
    return get_default_registry()


@pytest.fixture()
def scorer() -> ChainDepthScorer:
    return ChainDepthScorer()


def _make_graph(nodes: dict[str, str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Helper: nodes is {id: operation}, edges is [(src, tgt)]."""
    g = nx.DiGraph(name="test", metadata={})
    for nid, op in nodes.items():
        g.add_node(nid, operation=op, params={}, metadata={})
    for src, tgt in edges:
        g.add_edge(src, tgt, edge_type="control_flow")
    return g


# ---------------------------------------------------------------------------
# AC: Pure-only DAG (all pure effect_type) -> 0.0
# ---------------------------------------------------------------------------


class TestPureOnly:
    def test_all_pure_linear(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        # read_file and read_database are both PURE
        g = _make_graph(
            {"a": "read_file", "b": "read_database", "c": "read_state"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.score == 0.0

    def test_all_pure_branching(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "read_database", "c": "read_state", "d": "branch"},
            [("a", "b"), ("a", "c"), ("b", "d")],
        )
        result = scorer.score(g, registry)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# AC: All-stateful chain of depth 5 in DAG of depth 5 -> 1.0
# ---------------------------------------------------------------------------


class TestAllStateful:
    def test_all_stateful_chain_depth_5(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        # 5 nodes, all stateful: write_file(STATEFUL), write_database(STATEFUL), mutate_state(STATEFUL)
        g = _make_graph(
            {
                "n1": "write_file",
                "n2": "write_database",
                "n3": "mutate_state",
                "n4": "write_file",
                "n5": "write_database",
            },
            [("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n5")],
        )
        result = scorer.score(g, registry)
        # side_effect_chain = 5 nodes, max_dag_depth = 5 nodes -> 1.0
        assert result.score == 1.0

    def test_all_irreversible_chain(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        # IRREVERSIBLE is also a side-effect type
        g = _make_graph(
            {"a": "delete_file", "b": "delete_record", "c": "delete_file"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.score == 1.0

    def test_all_external_chain(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        # EXTERNAL is also a side-effect type
        g = _make_graph(
            {"a": "invoke_api", "b": "send_notification"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# AC: Mixed DAG: 2 side-effect nodes in depth-4 DAG -> ~0.5
# ---------------------------------------------------------------------------


class TestMixed:
    def test_two_side_effect_in_depth_4(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        # 4-node chain: pure -> stateful -> stateful -> pure
        # Full depth = 4 nodes. Side-effect subgraph: {b, c}, edge b->c. Path = 2 nodes.
        # Score = 2/4 = 0.5
        g = _make_graph(
            {"a": "read_file", "b": "write_database", "c": "mutate_state", "d": "read_state"},
            [("a", "b"), ("b", "c"), ("c", "d")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.5, abs=0.01)

    def test_non_adjacent_side_effects_lower_score(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        # 4-node chain: stateful -> pure -> pure -> stateful
        # Subgraph has {a, d} with no edge between them. Longest path = 1 node.
        # Score = 1/4 = 0.25
        g = _make_graph(
            {"a": "write_file", "b": "read_file", "c": "read_database", "d": "write_database"},
            [("a", "b"), ("b", "c"), ("c", "d")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.25, abs=0.01)

    def test_mixed_effect_types_all_count(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        # stateful -> external -> irreversible (all side-effect types)
        # 3 side-effect nodes in 3-node DAG -> 1.0
        g = _make_graph(
            {"a": "write_file", "b": "invoke_api", "c": "delete_file"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# AC: Single-node DAG -> 0.0
# ---------------------------------------------------------------------------


class TestSingleNode:
    def test_single_pure_node(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file"}, [])
        result = scorer.score(g, registry)
        assert result.score == 0.0

    def test_single_stateful_node(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        # Even a stateful single node has no chain -> 0.0
        g = _make_graph({"a": "write_file"}, [])
        result = scorer.score(g, registry)
        assert result.score == 0.0

    def test_empty_dag(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = scorer.score(g, registry)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# AC: Details include the longest side-effect path (node IDs)
# ---------------------------------------------------------------------------


class TestDetails:
    def test_path_included_in_details(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "write_database", "c": "mutate_state", "d": "read_state"},
            [("a", "b"), ("b", "c"), ("c", "d")],
        )
        result = scorer.score(g, registry)
        assert "longest_side_effect_path" in result.details
        path = result.details["longest_side_effect_path"]
        assert path == ["b", "c"]

    def test_empty_path_when_all_pure(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "read_database"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.details["longest_side_effect_path"] == []

    def test_full_path_when_all_stateful(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "write_file", "b": "write_database", "c": "mutate_state"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.details["longest_side_effect_path"] == ["a", "b", "c"]

    def test_single_node_path_empty(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "write_file"}, [])
        result = scorer.score(g, registry)
        assert result.details["longest_side_effect_path"] == []

    def test_diamond_picks_longest_side_effect_path(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        # Diamond: a -> b, a -> c, b -> d, c -> d
        # Left path (a->b->d): b is STATEFUL, d is STATEFUL -> subgraph path [b,d] = 2 nodes
        # Right path (a->c->d): c is PURE, d is STATEFUL -> subgraph path [d] = 1 node
        # Full DAG longest path = 3 nodes (a->b->d or a->c->d)
        # Side-effect subgraph: {b, d} with edge b->d. Longest path = [b, d] = 2 nodes.
        # Score = 2/3 ≈ 0.667
        g = _make_graph(
            {"a": "read_file", "b": "write_database", "c": "read_file", "d": "mutate_state"},
            [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(2 / 3, abs=0.01)
        assert result.details["longest_side_effect_path"] == ["b", "d"]


# ---------------------------------------------------------------------------
# Flagged nodes for edge cases
# ---------------------------------------------------------------------------


class TestFlaggedNodesEdgeCases:
    def test_pure_only_flagged_nodes_empty(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "read_database"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ()

    def test_single_node_flagged_nodes_empty(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "write_file"}, [])
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ()

    def test_empty_dag_flagged_nodes_empty(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_cyclic_graph_raises(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        """Cyclic input raises NetworkXUnfeasible from dag_longest_path."""
        g = _make_graph(
            {"a": "write_file", "b": "write_database"},
            [("a", "b"), ("b", "a")],
        )
        with pytest.raises(nx.NetworkXUnfeasible):
            scorer.score(g, registry)

    def test_unknown_operation_propagates_key_error(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "nonexistent_op", "b": "read_file"},
            [("a", "b")],
        )
        with pytest.raises(KeyError, match="nonexistent_op"):
            scorer.score(g, registry)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_satisfies_scorer_protocol(self) -> None:
        assert isinstance(ChainDepthScorer(), Scorer)

    def test_name_is_chain_depth(self) -> None:
        assert ChainDepthScorer().name == "chain_depth"

    def test_returns_subscore_with_correct_name(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file"}, [])
        result = scorer.score(g, registry)
        assert isinstance(result, SubScore)
        assert result.name == "chain_depth"
        assert result.weight == 0.0

    def test_flagged_nodes_exact_match_side_effect_path(
        self, scorer: ChainDepthScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "write_database", "c": "mutate_state", "d": "read_state"},
            [("a", "b"), ("b", "c"), ("c", "d")],
        )
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ("b", "c")
        # Pure nodes a and d must NOT be flagged
        assert "a" not in result.flagged_nodes
        assert "d" not in result.flagged_nodes
