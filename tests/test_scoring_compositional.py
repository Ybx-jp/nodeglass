"""Tests for compositional risk scorer (NOD-23).

AC:
- [x] read_credentials -> invoke_api (C=2.5) scores high
- [x] read_file -> read_file (C=1.0, low weights) scores low
- [x] Custom matrix entries override defaults
- [x] No edges -> score 0.0
- [x] Details include the highest-risk edge pair and its multiplier
"""

import networkx as nx
import pytest

from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.compositional import CompositionalScorer
from workflow_eval.scoring.protocols import Scorer
from workflow_eval.types import SubScore


@pytest.fixture()
def registry() -> OperationRegistry:
    return get_default_registry()


@pytest.fixture()
def scorer() -> CompositionalScorer:
    return CompositionalScorer()


def _make_graph(nodes: dict[str, str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Helper: nodes is {id: operation}, edges is [(src, tgt)]."""
    g = nx.DiGraph(name="test", metadata={})
    for nid, op in nodes.items():
        g.add_node(nid, operation=op, params={}, metadata={})
    for src, tgt in edges:
        g.add_edge(src, tgt, edge_type="control_flow")
    return g


# ---------------------------------------------------------------------------
# AC: read_credentials -> invoke_api (C=2.5) scores high
# ---------------------------------------------------------------------------


class TestHighComposition:
    def test_cred_to_api_scores_high(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        # edge_risk = 0.45 * 0.30 * 2.5 = 0.3375, score = 0.3375 / 3.0 = 0.1125
        g = _make_graph(
            {"a": "read_credentials", "b": "invoke_api"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.1125, abs=0.001)

    def test_cred_to_api_higher_than_neutral_pair(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        # Same ops but reversed (no composition entry) -> C=1.0
        # invoke_api -> read_credentials: 0.30 * 0.45 * 1.0 = 0.135, score = 0.045
        high = _make_graph(
            {"a": "read_credentials", "b": "invoke_api"},
            [("a", "b")],
        )
        neutral = _make_graph(
            {"a": "invoke_api", "b": "read_credentials"},
            [("a", "b")],
        )
        high_result = scorer.score(high, registry)
        neutral_result = scorer.score(neutral, registry)
        assert high_result.score > neutral_result.score * 2

    def test_authenticate_to_destroy_resource(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        # edge_risk = 0.20 * 0.90 * 2.3 = 0.414, score = 0.414 / 3.0 = 0.138
        g = _make_graph(
            {"a": "authenticate", "b": "destroy_resource"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.138, abs=0.001)


# ---------------------------------------------------------------------------
# AC: read_file -> read_file (C=1.0, low weights) scores low
# ---------------------------------------------------------------------------


class TestLowComposition:
    def test_read_to_read_scores_low(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        # edge_risk = 0.05 * 0.05 * 1.0 = 0.0025, score = 0.000833
        g = _make_graph(
            {"a": "read_file", "b": "read_file"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.000833, abs=0.0005)

    def test_low_much_less_than_high(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        low = _make_graph(
            {"a": "read_file", "b": "read_file"},
            [("a", "b")],
        )
        high = _make_graph(
            {"a": "read_credentials", "b": "invoke_api"},
            [("a", "b")],
        )
        low_result = scorer.score(low, registry)
        high_result = scorer.score(high, registry)
        # High should be >100x greater
        assert high_result.score > low_result.score * 100


# ---------------------------------------------------------------------------
# AC: Custom matrix entries override defaults
# ---------------------------------------------------------------------------


class TestCustomMatrix:
    def test_override_existing_entry(
        self, registry: OperationRegistry,
    ) -> None:
        # Override read_credentials -> invoke_api from 2.5 to 1.0
        custom = CompositionalScorer(compositions={("read_credentials", "invoke_api"): 1.0})
        g = _make_graph(
            {"a": "read_credentials", "b": "invoke_api"},
            [("a", "b")],
        )
        result = custom.score(g, registry)
        # edge_risk = 0.45 * 0.30 * 1.0 = 0.135, score = 0.045
        assert result.score == pytest.approx(0.045, abs=0.001)

    def test_add_new_entry(
        self, registry: OperationRegistry,
    ) -> None:
        # Add a new pair not in defaults
        custom = CompositionalScorer(compositions={("read_state", "write_database"): 3.0})
        g = _make_graph(
            {"a": "read_state", "b": "write_database"},
            [("a", "b")],
        )
        result = custom.score(g, registry)
        # edge_risk = 0.02 * 0.40 * 3.0 = 0.024, score = 0.008
        assert result.score == pytest.approx(0.008, abs=0.001)
        assert result.details["composition_multiplier"] == 3.0

    def test_default_scorer_unaffected_by_custom(
        self, scorer: CompositionalScorer, registry: OperationRegistry,
    ) -> None:
        # Verify default scorer still has original multiplier
        g = _make_graph(
            {"a": "read_credentials", "b": "invoke_api"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.details["composition_multiplier"] == 2.5


# ---------------------------------------------------------------------------
# AC: No edges -> score 0.0
# ---------------------------------------------------------------------------


class TestNoEdges:
    def test_no_edges_score_zero(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "read_credentials", "b": "invoke_api"},
            [],
        )
        result = scorer.score(g, registry)
        assert result.score == 0.0

    def test_single_node_score_zero(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "delete_file"}, [])
        result = scorer.score(g, registry)
        assert result.score == 0.0

    def test_empty_dag_score_zero(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = scorer.score(g, registry)
        assert result.score == 0.0
        assert result.details["highest_risk_edge"] is None


# ---------------------------------------------------------------------------
# AC: Details include the highest-risk edge pair and its multiplier
# ---------------------------------------------------------------------------


class TestDetails:
    def test_details_contain_edge_and_multiplier(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "read_credentials", "b": "invoke_api"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.details["highest_risk_edge"] == ("a", "b")
        assert result.details["highest_risk_ops"] == ("read_credentials", "invoke_api")
        assert result.details["composition_multiplier"] == 2.5
        assert result.details["edge_risk"] == pytest.approx(0.3375, abs=0.001)

    def test_picks_highest_risk_among_multiple_edges(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        # Two edges: low (read_file->read_file) and high (read_credentials->invoke_api)
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_credentials", "d": "invoke_api"},
            [("a", "b"), ("c", "d")],
        )
        result = scorer.score(g, registry)
        assert result.details["highest_risk_edge"] == ("c", "d")
        assert result.details["composition_multiplier"] == 2.5

    def test_no_edges_details_null(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file"}, [])
        result = scorer.score(g, registry)
        assert result.details["highest_risk_edge"] is None
        assert result.details["highest_risk_ops"] is None
        assert result.details["composition_multiplier"] == 0.0
        assert result.details["edge_risk"] == 0.0


# ---------------------------------------------------------------------------
# Wildcard matching
# ---------------------------------------------------------------------------


class TestWildcard:
    def test_branch_to_delete_file_matches_wildcard(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        # ("branch", "delete_*"): 2.0 should match delete_file
        # edge_risk = 0.10 * 0.80 * 2.0 = 0.16, score = 0.0533
        g = _make_graph(
            {"a": "branch", "b": "delete_file"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.0533, abs=0.001)
        assert result.details["composition_multiplier"] == 2.0

    def test_branch_to_delete_record_matches_wildcard(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        # ("branch", "delete_*"): 2.0 should also match delete_record
        # edge_risk = 0.10 * 0.85 * 2.0 = 0.17, score = 0.0567
        g = _make_graph(
            {"a": "branch", "b": "delete_record"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.0567, abs=0.001)
        assert result.details["composition_multiplier"] == 2.0

    def test_exact_match_wins_over_wildcard(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        # invoke_api -> delete_record has exact entry (2.2)
        # It should NOT fall through to any wildcard
        g = _make_graph(
            {"a": "invoke_api", "b": "delete_record"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.details["composition_multiplier"] == 2.2


# ---------------------------------------------------------------------------
# Multi-edge scoring
# ---------------------------------------------------------------------------


class TestMultiEdge:
    def test_max_edge_wins_in_chain(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        # Chain: read_file -> read_credentials -> invoke_api
        # Edge 1: read_file->read_credentials C=1.0, er = 0.05*0.45*1.0 = 0.0225
        # Edge 2: read_credentials->invoke_api C=2.5, er = 0.45*0.30*2.5 = 0.3375
        # SCORE = 0.3375 / 3.0 = 0.1125
        g = _make_graph(
            {"a": "read_file", "b": "read_credentials", "c": "invoke_api"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.1125, abs=0.001)
        assert result.details["highest_risk_edge"] == ("b", "c")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_unknown_operation_propagates_key_error(
        self, scorer: CompositionalScorer, registry: OperationRegistry
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
        assert isinstance(CompositionalScorer(), Scorer)

    def test_name_is_compositional(self) -> None:
        assert CompositionalScorer().name == "compositional"

    def test_returns_subscore_with_correct_name(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file"}, [])
        result = scorer.score(g, registry)
        assert isinstance(result, SubScore)
        assert result.name == "compositional"
        assert result.weight == 0.0

    def test_score_in_range(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        # authenticate -> destroy_resource is one of the highest: score ≈ 0.138
        g = _make_graph(
            {"a": "authenticate", "b": "destroy_resource"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert 0.0 <= result.score <= 1.0
        assert result.score == pytest.approx(0.138, abs=0.001)

    def test_flagged_nodes_empty(
        self, scorer: CompositionalScorer, registry: OperationRegistry
    ) -> None:
        """Compositional scorer flags edges, not nodes — flagged_nodes is empty."""
        g = _make_graph(
            {"a": "read_credentials", "b": "invoke_api"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ()
