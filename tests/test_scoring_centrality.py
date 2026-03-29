"""Tests for centrality scorer (NOD-21).

AC:
- [x] Uniform low-risk DAG -> low score
- [x] DAG with single high-risk chokepoint (high betweenness + high risk weight) -> significantly higher
- [x] Linear chain -> moderate (middle nodes have highest betweenness)
- [x] Single-node DAG -> 0.0 (no betweenness)
- [x] Flagged nodes = those with centrality_risk > mean + 1*std
"""

import networkx as nx
import pytest

from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.centrality import CentralityScorer
from workflow_eval.scoring.protocols import Scorer
from workflow_eval.types import SubScore


@pytest.fixture()
def registry() -> OperationRegistry:
    return get_default_registry()


@pytest.fixture()
def scorer() -> CentralityScorer:
    return CentralityScorer()


def _make_graph(nodes: dict[str, str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Helper: nodes is {id: operation}, edges is [(src, tgt)]."""
    g = nx.DiGraph(name="test", metadata={})
    for nid, op in nodes.items():
        g.add_node(nid, operation=op, params={}, metadata={})
    for src, tgt in edges:
        g.add_edge(src, tgt, edge_type="control_flow")
    return g


# ---------------------------------------------------------------------------
# AC: Uniform low-risk DAG -> low score
# ---------------------------------------------------------------------------


class TestUniformLowRisk:
    def test_all_pure_chain_low_score(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # 5-node chain, all read_file (risk=0.05)
        # bc = [0, 0.25, 0.333, 0.25, 0]
        # cr = [0, 0.0125, 0.01667, 0.0125, 0]
        # mean ≈ 0.00833, std ≈ 0.00697
        # SCORE ≈ 0.00833 * 1.00697 ≈ 0.0084
        g = _make_graph(
            {f"n{i}": "read_file" for i in range(5)},
            [(f"n{i}", f"n{i+1}") for i in range(4)],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.0084, abs=0.001)

    def test_all_pure_branching_low_score(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # Star: root -> 3 leaves (all read_file). No intermediate nodes -> all bc = 0.
        # cr = all 0. mean = 0, std = 0. SCORE = 0.
        g = _make_graph(
            {"root": "read_file", "l1": "read_file", "l2": "read_file", "l3": "read_file"},
            [("root", "l1"), ("root", "l2"), ("root", "l3")],
        )
        result = scorer.score(g, registry)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# AC: DAG with single high-risk chokepoint -> significantly higher
# ---------------------------------------------------------------------------


class TestHighRiskChokepoint:
    def test_bowtie_with_high_risk_center(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # Bow-tie: a,b -> c -> d,e. c = delete_file (0.80), others = read_file (0.05)
        # bc(c) = 1/3, all others = 0
        # cr(c) = 1/3 * 0.80 = 0.2667. Others = 0.
        # mean = 0.2667/5 = 0.05333
        # std ≈ 0.1067
        # SCORE ≈ 0.05333 * 1.1067 ≈ 0.059
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "delete_file", "d": "read_file", "e": "read_file"},
            [("a", "c"), ("b", "c"), ("c", "d"), ("c", "e")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.059, abs=0.005)

    def test_chokepoint_significantly_higher_than_uniform(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # Compare: uniform low-risk chain vs bow-tie with high-risk center
        uniform = _make_graph(
            {f"n{i}": "read_file" for i in range(5)},
            [(f"n{i}", f"n{i+1}") for i in range(4)],
        )
        chokepoint = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "delete_file", "d": "read_file", "e": "read_file"},
            [("a", "c"), ("b", "c"), ("c", "d"), ("c", "e")],
        )
        uniform_result = scorer.score(uniform, registry)
        choke_result = scorer.score(chokepoint, registry)
        # Pin absolute values to catch drift
        assert uniform_result.score == pytest.approx(0.0084, abs=0.001)
        assert choke_result.score == pytest.approx(0.059, abs=0.005)
        # Chokepoint should be at least 5x higher
        assert choke_result.score > uniform_result.score * 5

    def test_high_risk_node_without_betweenness_stays_low(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # delete_file as leaf: bc = 0, so cr = 0 regardless of risk weight.
        # Betweenness only rewards INTERMEDIATE nodes.
        g = _make_graph(
            {"a": "read_file", "b": "delete_file"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# AC: Linear chain -> moderate (middle nodes have highest betweenness)
# ---------------------------------------------------------------------------


class TestLinearChain:
    def test_chain_with_risky_middle_moderate(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # 3-node chain: read_file -> delete_file -> read_file
        # bc = [0, 0.5, 0]
        # cr = [0, 0.5*0.80, 0] = [0, 0.40, 0]
        # mean = 0.40/3 = 0.1333
        # std ≈ 0.1886
        # SCORE ≈ 0.1333 * 1.1886 ≈ 0.158
        g = _make_graph(
            {"a": "read_file", "b": "delete_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.158, abs=0.005)

    def test_middle_nodes_have_highest_centrality_risk(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # 5-node chain: middle node (n2) has highest betweenness (0.333)
        g = _make_graph(
            {f"n{i}": "write_database" for i in range(5)},
            [(f"n{i}", f"n{i+1}") for i in range(4)],
        )
        result = scorer.score(g, registry)
        cr = result.details["centrality_risks"]
        # n2 (center) should have the highest centrality_risk
        assert cr["n2"] > cr["n1"]
        assert cr["n2"] > cr["n3"]
        assert cr["n0"] == 0.0
        assert cr["n4"] == 0.0

    def test_chain_score_between_low_and_high(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # Chain with moderate-risk ops should score between uniform-low and chokepoint-high
        uniform_low = _make_graph(
            {f"n{i}": "read_file" for i in range(5)},
            [(f"n{i}", f"n{i+1}") for i in range(4)],
        )
        moderate_chain = _make_graph(
            {f"n{i}": "write_database" for i in range(5)},
            [(f"n{i}", f"n{i+1}") for i in range(4)],
        )
        # 3-node chain with delete_file center is a high-scoring topology
        high_chain = _make_graph(
            {"a": "read_file", "b": "delete_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        low_result = scorer.score(uniform_low, registry)
        mod_result = scorer.score(moderate_chain, registry)
        high_result = scorer.score(high_chain, registry)
        assert low_result.score < mod_result.score < high_result.score


# ---------------------------------------------------------------------------
# AC: Single-node DAG -> 0.0 (no betweenness)
# ---------------------------------------------------------------------------


class TestSingleNode:
    def test_single_node_score_zero(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "delete_file"}, [])
        result = scorer.score(g, registry)
        assert result.score == 0.0
        # n<=1 early return produces empty details
        assert result.details["centrality_risks"] == {}

    def test_empty_dag_score_zero(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = scorer.score(g, registry)
        assert result.score == 0.0

    def test_two_node_dag_score_zero(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # Two nodes, no intermediate -> bc = 0 for both -> score = 0
        # Unlike n<=1 early return, n=2 goes through normal path -> details populated
        g = _make_graph(
            {"a": "delete_file", "b": "delete_file"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.score == 0.0
        # n=2 goes through bc computation, so details are populated (not empty)
        cr = result.details["centrality_risks"]
        assert set(cr.keys()) == {"a", "b"}
        assert cr["a"] == 0.0
        assert cr["b"] == 0.0


# ---------------------------------------------------------------------------
# AC: Flagged nodes = those with centrality_risk > mean + 1*std
# ---------------------------------------------------------------------------


class TestFlaggedNodes:
    def test_chokepoint_flagged(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # Bow-tie with high-risk center: cr(c) = 0.2667, threshold ≈ 0.160
        # c is flagged, others are not
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "delete_file", "d": "read_file", "e": "read_file"},
            [("a", "c"), ("b", "c"), ("c", "d"), ("c", "e")],
        )
        result = scorer.score(g, registry)
        assert "c" in result.flagged_nodes
        assert "a" not in result.flagged_nodes
        assert "d" not in result.flagged_nodes

    def test_uniform_risk_center_node_flagged(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # All same operation -> centrality_risk is proportional to bc only.
        # In a symmetric chain, middle node has slightly higher cr but it's just
        # the natural bc distribution, not a dramatic outlier.
        # For 5-node chain with all read_file:
        #   cr = [0, 0.0125, 0.01667, 0.0125, 0]
        #   mean = 0.00833, std = 0.00697
        #   threshold = 0.01530
        #   cr(n2) = 0.01667 > 0.01530 -> n2 flagged (it IS the statistical outlier)
        # This is correct behavior: even in a uniform-risk chain, the center
        # node is a structural outlier.
        g = _make_graph(
            {f"n{i}": "read_file" for i in range(5)},
            [(f"n{i}", f"n{i+1}") for i in range(4)],
        )
        result = scorer.score(g, registry)
        # n2 is flagged as the structural center (highest bc)
        assert "n2" in result.flagged_nodes
        # Endpoints never flagged (cr = 0)
        assert "n0" not in result.flagged_nodes
        assert "n4" not in result.flagged_nodes

    def test_star_no_flagged(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # Star: all bc = 0, all cr = 0, mean = 0, std = 0, threshold = 0
        # No node has cr > 0 -> no flagged nodes
        g = _make_graph(
            {"root": "read_file", "l1": "read_file", "l2": "read_file"},
            [("root", "l1"), ("root", "l2")],
        )
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ()

    def test_flagged_nodes_exact_match_on_chain(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        # 3-node chain: read_file -> delete_file -> read_file
        # cr = [0, 0.40, 0], mean = 0.1333, std = 0.1886
        # threshold = 0.1333 + 0.1886 = 0.3219
        # cr(b) = 0.40 > 0.3219 -> flagged
        g = _make_graph(
            {"a": "read_file", "b": "delete_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ("b",)

    def test_empty_dag_flagged_empty(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ()


# ---------------------------------------------------------------------------
# Details
# ---------------------------------------------------------------------------


class TestDetails:
    def test_centrality_risks_in_details(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "delete_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert "centrality_risks" in result.details
        cr = result.details["centrality_risks"]
        assert set(cr.keys()) == {"a", "b", "c"}
        # b has bc=0.5, risk=0.80 -> cr = 0.40
        assert cr["b"] == pytest.approx(0.40, abs=0.01)
        assert cr["a"] == 0.0
        assert cr["c"] == 0.0

    def test_empty_details_when_empty_dag(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = scorer.score(g, registry)
        assert result.details["centrality_risks"] == {}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_unknown_operation_propagates_key_error(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "nonexistent_op", "b": "read_file"},
            [("a", "b")],
        )
        with pytest.raises(KeyError, match="nonexistent_op"):
            scorer.score(g, registry)

    def test_cyclic_graph_does_not_raise(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        """Unlike DAG-specific scorers, betweenness_centrality accepts cycles."""
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c"), ("c", "a")],
        )
        result = scorer.score(g, registry)
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_satisfies_scorer_protocol(self) -> None:
        assert isinstance(CentralityScorer(), Scorer)

    def test_name_is_centrality(self) -> None:
        assert CentralityScorer().name == "centrality"

    def test_returns_subscore_with_correct_name(
        self, scorer: CentralityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file"}, [])
        result = scorer.score(g, registry)
        assert isinstance(result, SubScore)
        assert result.name == "centrality"
        assert result.weight == 0.0
