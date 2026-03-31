"""Tests for aggregator and RiskScoringEngine (NOD-24).

AC:
- [x] Invalid weights (sum != 1.0) raise ValueError
- [x] Engine returns RiskProfile with all 6 sub-scores + aggregate
- [x] Risk level thresholds correct: 0.24 -> low, 0.25 -> medium, 0.74 -> high, 0.75 -> critical
- [x] critical_paths populated (longest risk-weighted paths)
- [x] chokepoints populated (high centrality + high risk nodes)
"""

import networkx as nx
import pytest

from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.aggregator import aggregate, apply_weights, classify_risk
from workflow_eval.scoring.engine import RiskScoringEngine
from workflow_eval.types import (
    RiskLevel,
    RiskProfile,
    ScoringConfig,
    SubScore,
)


@pytest.fixture()
def registry() -> OperationRegistry:
    return get_default_registry()


@pytest.fixture()
def config() -> ScoringConfig:
    return ScoringConfig()


@pytest.fixture()
def engine(config: ScoringConfig, registry: OperationRegistry) -> RiskScoringEngine:
    return RiskScoringEngine(config, registry)


def _make_graph(nodes: dict[str, str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Helper: nodes is {id: operation}, edges is [(src, tgt)]."""
    g = nx.DiGraph(name="test", metadata={})
    for nid, op in nodes.items():
        g.add_node(nid, operation=op, params={}, metadata={})
    for src, tgt in edges:
        g.add_edge(src, tgt, edge_type="control_flow")
    return g


# ---------------------------------------------------------------------------
# AC: Invalid weights (sum != 1.0) raise ValueError
# ---------------------------------------------------------------------------


class TestInvalidWeights:
    def test_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringConfig(fan_out=0.5, chain_depth=0.5, irreversibility=0.5)

    def test_all_zero_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringConfig(
                fan_out=0.0, chain_depth=0.0, irreversibility=0.0,
                centrality=0.0, spectral=0.0, compositional=0.0,
            )

    def test_valid_custom_weights_accepted(self) -> None:
        config = ScoringConfig(
            fan_out=0.10, chain_depth=0.10, irreversibility=0.30,
            centrality=0.20, spectral=0.20, compositional=0.10,
        )
        assert config.fan_out == 0.10


# ---------------------------------------------------------------------------
# AC: Engine returns RiskProfile with all 6 sub-scores + aggregate
# ---------------------------------------------------------------------------


class TestEngineReturnsProfile:
    def test_returns_risk_profile(
        self, engine: RiskScoringEngine,
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = engine.score(g)
        assert isinstance(result, RiskProfile)

    def test_all_six_subscores_present(
        self, engine: RiskScoringEngine,
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = engine.score(g)
        names = {s.name for s in result.sub_scores}
        assert names == {
            "fan_out", "chain_depth", "irreversibility",
            "centrality", "spectral", "compositional",
        }

    def test_weights_populated_on_subscores(
        self, engine: RiskScoringEngine, config: ScoringConfig,
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = engine.score(g)
        weight_map = {s.name: s.weight for s in result.sub_scores}
        assert weight_map["fan_out"] == config.fan_out
        assert weight_map["chain_depth"] == config.chain_depth
        assert weight_map["irreversibility"] == config.irreversibility
        assert weight_map["centrality"] == config.centrality
        assert weight_map["spectral"] == config.spectral
        assert weight_map["compositional"] == config.compositional

    def test_aggregate_matches_weighted_sum(
        self, engine: RiskScoringEngine, config: ScoringConfig,
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = engine.score(g)
        expected = sum(s.weight * s.score for s in result.sub_scores)
        assert result.aggregate_score == pytest.approx(expected, abs=1e-6)
        # Pin absolute value to guard against uniform weight bugs
        assert result.aggregate_score == pytest.approx(0.0609, abs=0.005)

    def test_low_risk_chain_aggregate(
        self, engine: RiskScoringEngine,
    ) -> None:
        # 3-node all read_file chain: aggregate ≈ 0.0609
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = engine.score(g)
        assert result.aggregate_score == pytest.approx(0.0609, abs=0.005)
        assert result.risk_level == RiskLevel.LOW

    def test_node_and_edge_counts(
        self, engine: RiskScoringEngine,
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = engine.score(g)
        assert result.node_count == 3
        assert result.edge_count == 2

    def test_empty_dag(
        self, engine: RiskScoringEngine,
    ) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = engine.score(g)
        assert result.aggregate_score == 0.0
        assert result.risk_level == RiskLevel.LOW
        assert result.node_count == 0
        assert result.edge_count == 0

    def test_workflow_name_from_graph(
        self, engine: RiskScoringEngine,
    ) -> None:
        g = nx.DiGraph(name="my_workflow", metadata={})
        result = engine.score(g)
        assert result.workflow_name == "my_workflow"


# ---------------------------------------------------------------------------
# AC: Risk level thresholds correct
# ---------------------------------------------------------------------------


class TestRiskLevelThresholds:
    def test_0_24_is_low(self) -> None:
        assert classify_risk(0.24) == RiskLevel.LOW

    def test_0_25_is_medium(self) -> None:
        assert classify_risk(0.25) == RiskLevel.MEDIUM

    def test_0_49_is_medium(self) -> None:
        assert classify_risk(0.49) == RiskLevel.MEDIUM

    def test_0_50_is_high(self) -> None:
        assert classify_risk(0.50) == RiskLevel.HIGH

    def test_0_74_is_high(self) -> None:
        assert classify_risk(0.74) == RiskLevel.HIGH

    def test_0_75_is_critical(self) -> None:
        assert classify_risk(0.75) == RiskLevel.CRITICAL

    def test_1_0_is_critical(self) -> None:
        assert classify_risk(1.0) == RiskLevel.CRITICAL

    def test_0_0_is_low(self) -> None:
        assert classify_risk(0.0) == RiskLevel.LOW


# ---------------------------------------------------------------------------
# AC: critical_paths populated (longest risk-weighted paths)
# ---------------------------------------------------------------------------


class TestCriticalPaths:
    def test_chain_produces_full_path(
        self, engine: RiskScoringEngine,
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = engine.score(g)
        assert len(result.critical_paths) == 1
        assert result.critical_paths[0] == ("a", "b", "c")

    def test_picks_highest_risk_path(
        self, engine: RiskScoringEngine,
    ) -> None:
        # Two paths: a->b (read_file->read_file, risk=0.10)
        #            a->c (read_file->delete_file, risk=0.85)
        # Critical path should go through c (higher risk sink)
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "delete_file"},
            [("a", "b"), ("a", "c")],
        )
        result = engine.score(g)
        assert len(result.critical_paths) == 1
        assert result.critical_paths[0] == ("a", "c")

    def test_longer_risky_chain(
        self, engine: RiskScoringEngine,
    ) -> None:
        # Bow-tie: b has read_credentials (0.45), c has delete_file (0.80), d has invoke_api (0.30)
        # Path b->c->d has risk 0.45+0.80+0.30 = 1.55 (highest)
        g = _make_graph(
            {"a": "read_file", "b": "read_credentials", "c": "delete_file",
             "d": "invoke_api", "e": "read_file"},
            [("a", "c"), ("b", "c"), ("c", "d"), ("c", "e")],
        )
        result = engine.score(g)
        assert result.critical_paths[0] == ("b", "c", "d")

    def test_empty_dag_no_critical_paths(
        self, engine: RiskScoringEngine,
    ) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = engine.score(g)
        assert result.critical_paths == ()

    def test_single_node_path(
        self, engine: RiskScoringEngine,
    ) -> None:
        g = _make_graph({"a": "delete_file"}, [])
        result = engine.score(g)
        assert result.critical_paths == (("a",),)


# ---------------------------------------------------------------------------
# AC: chokepoints populated (high centrality + high risk nodes)
# ---------------------------------------------------------------------------


class TestChokepoints:
    def test_bowtie_center_is_chokepoint(
        self, engine: RiskScoringEngine,
    ) -> None:
        # c = delete_file (risk=0.80) at bow-tie center (high betweenness)
        g = _make_graph(
            {"a": "read_file", "b": "read_credentials", "c": "delete_file",
             "d": "invoke_api", "e": "read_file"},
            [("a", "c"), ("b", "c"), ("c", "d"), ("c", "e")],
        )
        result = engine.score(g)
        assert result.chokepoints == ("c",)

    def test_low_risk_center_not_chokepoint(
        self, engine: RiskScoringEngine,
    ) -> None:
        # All read_file (risk=0.05) — even if center has high betweenness,
        # risk weight is below threshold
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file",
             "d": "read_file", "e": "read_file"},
            [("a", "c"), ("b", "c"), ("c", "d"), ("c", "e")],
        )
        result = engine.score(g)
        assert result.chokepoints == ()

    def test_empty_dag_no_chokepoints(
        self, engine: RiskScoringEngine,
    ) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = engine.score(g)
        assert result.chokepoints == ()

    def test_leaf_high_risk_not_chokepoint(
        self, engine: RiskScoringEngine,
    ) -> None:
        # delete_file as leaf: high risk but zero betweenness -> not flagged by centrality
        g = _make_graph(
            {"a": "read_file", "b": "delete_file"},
            [("a", "b")],
        )
        result = engine.score(g)
        assert result.chokepoints == ()


# ---------------------------------------------------------------------------
# Aggregator unit tests
# ---------------------------------------------------------------------------


class TestAggregator:
    def test_weighted_sum(self) -> None:
        scores = (
            SubScore(name="fan_out", score=0.5, weight=0.0),
            SubScore(name="chain_depth", score=0.5, weight=0.0),
            SubScore(name="irreversibility", score=0.5, weight=0.0),
            SubScore(name="centrality", score=0.5, weight=0.0),
            SubScore(name="spectral", score=0.5, weight=0.0),
            SubScore(name="compositional", score=0.5, weight=0.0),
        )
        config = ScoringConfig()
        result = aggregate(scores, config)
        # All scores = 0.5, weights sum to 1.0 -> aggregate = 0.5
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_single_scorer_dominates(self) -> None:
        scores = (
            SubScore(name="fan_out", score=0.0, weight=0.0),
            SubScore(name="chain_depth", score=0.0, weight=0.0),
            SubScore(name="irreversibility", score=1.0, weight=0.0),
            SubScore(name="centrality", score=0.0, weight=0.0),
            SubScore(name="spectral", score=0.0, weight=0.0),
            SubScore(name="compositional", score=0.0, weight=0.0),
        )
        config = ScoringConfig()
        result = aggregate(scores, config)
        # Only irreversibility scores 1.0 * 0.25 = 0.25
        assert result == pytest.approx(0.25, abs=1e-6)

    def test_all_zero_scores(self) -> None:
        scores = tuple(
            SubScore(name=n, score=0.0, weight=0.0)
            for n in ["fan_out", "chain_depth", "irreversibility",
                       "centrality", "spectral", "compositional"]
        )
        assert aggregate(scores, ScoringConfig()) == 0.0


# ---------------------------------------------------------------------------
# Coverage gaps from review
# ---------------------------------------------------------------------------


class TestCoverageGaps:
    def test_weight_tolerance_boundary_accepted(self) -> None:
        # Spec says tolerance 0.001 — weights summing to 1.0005 should be accepted
        config = ScoringConfig(
            fan_out=0.15, chain_depth=0.20, irreversibility=0.2505,
            centrality=0.15, spectral=0.10, compositional=0.15,
        )
        assert config.irreversibility == 0.2505

    def test_weight_tolerance_boundary_rejected(self) -> None:
        # Weights summing to 1.002 (just beyond 0.001 tolerance) should be rejected
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringConfig(
                fan_out=0.15, chain_depth=0.202, irreversibility=0.25,
                centrality=0.15, spectral=0.10, compositional=0.15,
            )

    def test_chokepoint_at_exact_threshold(
        self, engine: RiskScoringEngine,
    ) -> None:
        # mutate_state (risk=0.25) at bow-tie center: exactly at _CHOKEPOINT_RISK_THRESHOLD
        # >= 0.25 means it should be included
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "mutate_state",
             "d": "read_file", "e": "read_file"},
            [("a", "c"), ("b", "c"), ("c", "d"), ("c", "e")],
        )
        result = engine.score(g)
        assert "c" in result.chokepoints

    def test_negative_weight_rejected(self) -> None:
        # Individual weights must be >= 0.0
        with pytest.raises(ValueError):
            ScoringConfig(
                fan_out=-0.5, chain_depth=0.65, irreversibility=0.25,
                centrality=0.15, spectral=0.10, compositional=0.35,
            )

    def test_apply_weights_direct(self) -> None:
        scores = (
            SubScore(name="fan_out", score=0.5, weight=0.0),
            SubScore(name="chain_depth", score=0.8, weight=0.0),
            SubScore(name="irreversibility", score=0.3, weight=0.0),
            SubScore(name="centrality", score=0.1, weight=0.0),
            SubScore(name="spectral", score=0.6, weight=0.0),
            SubScore(name="compositional", score=0.2, weight=0.0),
        )
        config = ScoringConfig()
        result = apply_weights(scores, config)
        weight_map = {s.name: s for s in result}
        # Weights populated from config
        assert weight_map["fan_out"].weight == 0.15
        assert weight_map["chain_depth"].weight == 0.20
        assert weight_map["irreversibility"].weight == 0.25
        assert weight_map["centrality"].weight == 0.15
        assert weight_map["spectral"].weight == 0.10
        assert weight_map["compositional"].weight == 0.15
        # Scores unchanged
        assert weight_map["fan_out"].score == 0.5
        assert weight_map["chain_depth"].score == 0.8

    def test_cyclic_graph_raises(
        self, engine: RiskScoringEngine,
    ) -> None:
        # Engine assumes valid DAG — cycles cause NetworkXUnfeasible from topological_sort
        g = _make_graph(
            {"a": "read_file", "b": "read_file"},
            [("a", "b"), ("b", "a")],
        )
        with pytest.raises(nx.NetworkXUnfeasible):
            engine.score(g)
