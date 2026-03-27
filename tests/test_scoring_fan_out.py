"""Tests for fan-out (blast radius) scorer (NOD-18).

AC:
- [x] Single-node DAG -> score 0.0
- [x] Linear A->B->C with root risk_weight 0.5 -> (2/3)*0.5 = ~0.33
- [x] Star topology: root (risk 0.8) fanning to 5 leaves -> high score
- [x] `details` contains top-5 nodes ranked by weighted fan-out
- [x] `flagged_nodes` lists nodes exceeding threshold
"""

import networkx as nx
import pytest

from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.fan_out import FanOutScorer
from workflow_eval.scoring.protocols import Scorer
from workflow_eval.types import SubScore


@pytest.fixture()
def registry() -> OperationRegistry:
    return get_default_registry()


@pytest.fixture()
def scorer() -> FanOutScorer:
    return FanOutScorer()


def _make_graph(nodes: dict[str, str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Helper: nodes is {id: operation}, edges is [(src, tgt)]."""
    g = nx.DiGraph(name="test", metadata={})
    for nid, op in nodes.items():
        g.add_node(nid, operation=op, params={}, metadata={})
    for src, tgt in edges:
        g.add_edge(src, tgt, edge_type="control_flow")
    return g


# ---------------------------------------------------------------------------
# AC: Single-node DAG -> score 0.0
# ---------------------------------------------------------------------------


class TestSingleNode:
    def test_single_node_score_zero(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file"}, [])
        result = scorer.score(g, registry)
        assert result.score == 0.0

    def test_empty_dag_score_zero(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = scorer.score(g, registry)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# AC: Linear A->B->C with root risk_weight 0.5 -> (2/3)*0.5 = ~0.33
# ---------------------------------------------------------------------------


class TestLinearChain:
    def test_linear_abc_root_risk_05(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        # send_email has base_risk_weight=0.50
        g = _make_graph(
            {"a": "send_email", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        # fan_out(a) = 2/3, weighted = (2/3) * 0.5 = 0.3333...
        assert result.score == pytest.approx(2 / 3 * 0.5, abs=1e-4)

    def test_linear_max_comes_from_root(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        # Root has highest reachability, so it drives the max
        g = _make_graph(
            {"a": "send_email", "b": "send_email", "c": "send_email"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        # fan_out(a) = 2/3 * 0.5 = 0.333
        # fan_out(b) = 1/3 * 0.5 = 0.167
        # fan_out(c) = 0/3 * 0.5 = 0.0
        assert result.score == pytest.approx(2 / 3 * 0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# AC: Star topology: root (risk 0.8) fanning to 5 leaves -> high score
# ---------------------------------------------------------------------------


class TestStarTopology:
    def test_star_root_high_risk(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        # delete_file has base_risk_weight=0.80
        g = _make_graph(
            {
                "root": "delete_file",
                "l1": "read_file",
                "l2": "read_file",
                "l3": "read_file",
                "l4": "read_file",
                "l5": "read_file",
            },
            [("root", f"l{i}") for i in range(1, 6)],
        )
        result = scorer.score(g, registry)
        # fan_out(root) = 5/6, weighted = (5/6) * 0.8 = 0.6667
        expected = (5 / 6) * 0.8
        assert result.score == pytest.approx(expected, abs=1e-4)
        assert result.score > 0.5  # "high score"

    def test_star_leaves_score_zero(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {
                "root": "delete_file",
                "l1": "read_file",
                "l2": "read_file",
            },
            [("root", "l1"), ("root", "l2")],
        )
        result = scorer.score(g, registry)
        # Leaves have fan_out=0, so only root contributes
        top = result.details["top_nodes"]
        leaf_scores = [n for n in top if n["node_id"] in ("l1", "l2")]
        for leaf in leaf_scores:
            assert leaf["weighted_fan_out"] == 0.0


# ---------------------------------------------------------------------------
# AC: `details` contains top-5 nodes ranked by weighted fan-out
# ---------------------------------------------------------------------------


class TestDetails:
    def test_top_5_present(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        # 7 nodes — details should contain exactly 5
        nodes = {f"n{i}": "read_file" for i in range(7)}
        edges = [(f"n{i}", f"n{i+1}") for i in range(6)]
        g = _make_graph(nodes, edges)
        result = scorer.score(g, registry)
        assert "top_nodes" in result.details
        assert len(result.details["top_nodes"]) == 5

    def test_top_5_ranked_descending(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        nodes = {f"n{i}": "read_file" for i in range(7)}
        edges = [(f"n{i}", f"n{i+1}") for i in range(6)]
        g = _make_graph(nodes, edges)
        result = scorer.score(g, registry)
        scores = [n["weighted_fan_out"] for n in result.details["top_nodes"]]
        assert scores == sorted(scores, reverse=True)

    def test_fewer_than_5_nodes(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file", "b": "read_file"}, [("a", "b")])
        result = scorer.score(g, registry)
        assert len(result.details["top_nodes"]) == 2

    def test_top_nodes_have_correct_fields_and_types(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file", "b": "read_file"}, [("a", "b")])
        result = scorer.score(g, registry)
        for entry in result.details["top_nodes"]:
            assert isinstance(entry["node_id"], str)
            assert isinstance(entry["weighted_fan_out"], float)
            assert entry["node_id"] in ("a", "b")


# ---------------------------------------------------------------------------
# AC: `flagged_nodes` lists nodes exceeding threshold
# ---------------------------------------------------------------------------


class TestFlaggedNodes:
    def test_high_risk_node_flagged(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        # delete_file (0.8) as root of star with 5 leaves: weighted = 5/6 * 0.8 ≈ 0.667 > 0.5
        g = _make_graph(
            {
                "root": "delete_file",
                "l1": "read_file",
                "l2": "read_file",
                "l3": "read_file",
                "l4": "read_file",
                "l5": "read_file",
            },
            [("root", f"l{i}") for i in range(1, 6)],
        )
        result = scorer.score(g, registry)
        assert "root" in result.flagged_nodes

    def test_low_risk_node_not_flagged(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        # read_file (0.05) as root of chain: 2/3 * 0.05 = 0.033 < 0.5
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ()

    def test_threshold_boundary_exactly_05_not_flagged(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        """Exactly 0.5 weighted_fan_out should NOT be flagged (strict > check)."""
        from workflow_eval.ontology.effect_types import EffectTarget, EffectType
        from workflow_eval.types import OperationDefinition

        custom_reg = OperationRegistry()
        custom_reg.register(OperationDefinition(
            name="risky",
            category="test",
            base_risk_weight=1.0,
            effect_type=EffectType.IRREVERSIBLE,
            effect_targets=frozenset({EffectTarget.DATABASE}),
        ))
        custom_reg.register(OperationDefinition(
            name="safe",
            category="test",
            base_risk_weight=0.0,
            effect_type=EffectType.PURE,
            effect_targets=frozenset({EffectTarget.DATABASE}),
        ))
        # root(risky)->leaf(safe): fan_out(root) = 1/2, weighted = 1/2 * 1.0 = 0.5 exactly
        g = _make_graph({"root": "risky", "leaf": "safe"}, [("root", "leaf")])
        result = scorer.score(g, custom_reg)
        assert result.score == pytest.approx(0.5, abs=1e-6)
        assert "root" not in result.flagged_nodes  # strict >, not >=

    def test_leaves_not_flagged_even_when_root_is(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        # Root is flagged (5/6 * 0.8 = 0.667 > 0.5) but leaves are not (fan_out=0)
        g = _make_graph(
            {
                "root": "delete_file",
                "l1": "delete_file",
                "l2": "delete_file",
                "l3": "delete_file",
                "l4": "delete_file",
                "l5": "delete_file",
            },
            [("root", f"l{i}") for i in range(1, 6)],
        )
        result = scorer.score(g, registry)
        assert "root" in result.flagged_nodes
        for i in range(1, 6):
            assert f"l{i}" not in result.flagged_nodes


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_satisfies_scorer_protocol(self) -> None:
        scorer = FanOutScorer()
        assert isinstance(scorer, Scorer)

    def test_name_is_fan_out(self) -> None:
        assert FanOutScorer().name == "fan_out"

    def test_returns_subscore_with_correct_name(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file"}, [])
        result = scorer.score(g, registry)
        assert isinstance(result, SubScore)
        assert result.name == "fan_out"

    def test_weight_is_zero_placeholder(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        """Scorer returns weight=0.0; the engine assigns the real weight."""
        g = _make_graph({"a": "read_file"}, [])
        result = scorer.score(g, registry)
        assert result.weight == 0.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_unknown_operation_propagates_key_error(
        self, scorer: FanOutScorer, registry: OperationRegistry
    ) -> None:
        """If a node's operation is not in the registry, KeyError propagates."""
        g = _make_graph({"a": "nonexistent_op"}, [])
        with pytest.raises(KeyError, match="nonexistent_op"):
            scorer.score(g, registry)
