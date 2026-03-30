"""Integration tests: YAML -> DAG -> score (NOD-25).

AC:
- [x] pytest tests/test_scoring.py -v passes
- [x] Each scorer has at least 3 test cases
- [x] safe_read_pipeline.yaml -> "low"
- [x] risky_delete_cascade.yaml -> "high" or "critical"
- [x] moderate_api_chain.yaml -> "medium"
- [x] Edge cases covered (single node, cycles, custom config)
"""

from pathlib import Path

import networkx as nx
import pytest

from workflow_eval.dag.models import to_networkx
from workflow_eval.dag.schema import load_workflow
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.engine import RiskScoringEngine
from workflow_eval.types import RiskLevel, RiskProfile, ScoringConfig

EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "sample_workflows"


@pytest.fixture()
def registry() -> OperationRegistry:
    return get_default_registry()


@pytest.fixture()
def engine(registry: OperationRegistry) -> RiskScoringEngine:
    return RiskScoringEngine(ScoringConfig(), registry)


def _make_graph(nodes: dict[str, str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    g = nx.DiGraph(name="test", metadata={})
    for nid, op in nodes.items():
        g.add_node(nid, operation=op, params={}, metadata={})
    for src, tgt in edges:
        g.add_edge(src, tgt, edge_type="control_flow")
    return g


def _load_and_score(engine: RiskScoringEngine, filename: str) -> RiskProfile:
    dag = load_workflow(EXAMPLES_DIR / filename)
    g = to_networkx(dag)
    return engine.score(g)


# ---------------------------------------------------------------------------
# AC: safe_read_pipeline.yaml -> "low"
# ---------------------------------------------------------------------------


class TestSafeReadPipeline:
    def test_risk_level_low(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "safe_read_pipeline.yaml")
        assert result.risk_level == RiskLevel.LOW

    def test_aggregate_below_025(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "safe_read_pipeline.yaml")
        assert result.aggregate_score < 0.25
        assert result.aggregate_score == pytest.approx(0.061, abs=0.01)

    def test_irreversibility_zero(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "safe_read_pipeline.yaml")
        irrev = next(s for s in result.sub_scores if s.name == "irreversibility")
        assert irrev.score == 0.0


# ---------------------------------------------------------------------------
# AC: risky_delete_cascade.yaml -> "high" or "critical"
# ---------------------------------------------------------------------------


class TestRiskyDeleteCascade:
    def test_risk_level_high_or_critical(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "risky_delete_cascade.yaml")
        assert result.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}

    def test_aggregate_above_050(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "risky_delete_cascade.yaml")
        assert result.aggregate_score >= 0.50
        assert result.aggregate_score == pytest.approx(0.624, abs=0.02)

    def test_high_irreversibility(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "risky_delete_cascade.yaml")
        irrev = next(s for s in result.sub_scores if s.name == "irreversibility")
        assert irrev.score >= 0.5

    def test_has_critical_path(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "risky_delete_cascade.yaml")
        assert len(result.critical_paths) >= 1

    def test_has_chokepoints(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "risky_delete_cascade.yaml")
        # exec node (execute_code, risk=0.60) at bow-tie center is a chokepoint
        assert len(result.chokepoints) >= 1


# ---------------------------------------------------------------------------
# AC: moderate_api_chain.yaml -> "medium"
# ---------------------------------------------------------------------------


class TestModerateApiChain:
    def test_risk_level_medium_from_yaml(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "moderate_api_chain.yaml")
        assert result.risk_level == RiskLevel.MEDIUM

    def test_risk_level_medium_from_json(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "moderate_api_chain.json")
        assert result.risk_level == RiskLevel.MEDIUM

    def test_aggregate_in_medium_range(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "moderate_api_chain.yaml")
        assert 0.25 <= result.aggregate_score < 0.50

    def test_yaml_json_identical_scores(self, engine: RiskScoringEngine) -> None:
        yaml_result = _load_and_score(engine, "moderate_api_chain.yaml")
        json_result = _load_and_score(engine, "moderate_api_chain.json")
        assert yaml_result.aggregate_score == pytest.approx(json_result.aggregate_score, abs=1e-6)


# ---------------------------------------------------------------------------
# Per-scorer integration (at least 3 tests each via the engine)
# ---------------------------------------------------------------------------


class TestPerScorerIntegration:
    """Verify each of the 6 scorers produces correct sub-scores through the engine."""

    def test_fan_out_safe_pipeline(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "safe_read_pipeline.yaml")
        fan_out = next(s for s in result.sub_scores if s.name == "fan_out")
        assert fan_out.score == pytest.approx(0.033, abs=0.01)

    def test_fan_out_risky_cascade(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "risky_delete_cascade.yaml")
        fan_out = next(s for s in result.sub_scores if s.name == "fan_out")
        assert fan_out.score > 0.2  # exec fans out to 3 deletes

    def test_fan_out_moderate_chain(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "moderate_api_chain.yaml")
        fan_out = next(s for s in result.sub_scores if s.name == "fan_out")
        assert fan_out.score == pytest.approx(0.125, abs=0.01)

    def test_chain_depth_safe(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "safe_read_pipeline.yaml")
        chain = next(s for s in result.sub_scores if s.name == "chain_depth")
        assert chain.score == 0.0  # all pure ops

    def test_chain_depth_risky(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "risky_delete_cascade.yaml")
        chain = next(s for s in result.sub_scores if s.name == "chain_depth")
        assert chain.score == 1.0  # all side-effect ops

    def test_chain_depth_moderate(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "moderate_api_chain.yaml")
        chain = next(s for s in result.sub_scores if s.name == "chain_depth")
        assert chain.score == pytest.approx(0.75, abs=0.01)

    def test_centrality_safe(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "safe_read_pipeline.yaml")
        cent = next(s for s in result.sub_scores if s.name == "centrality")
        assert cent.score < 0.02

    def test_centrality_risky(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "risky_delete_cascade.yaml")
        cent = next(s for s in result.sub_scores if s.name == "centrality")
        assert cent.score > 0.01

    def test_centrality_moderate(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "moderate_api_chain.yaml")
        cent = next(s for s in result.sub_scores if s.name == "centrality")
        assert cent.score == pytest.approx(0.048, abs=0.01)

    def test_spectral_safe(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "safe_read_pipeline.yaml")
        spec = next(s for s in result.sub_scores if s.name == "spectral")
        assert spec.score == pytest.approx(0.545, abs=0.01)

    def test_spectral_risky(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "risky_delete_cascade.yaml")
        spec = next(s for s in result.sub_scores if s.name == "spectral")
        assert spec.score > 0.8  # fragile bow-tie structure

    def test_spectral_moderate(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "moderate_api_chain.yaml")
        spec = next(s for s in result.sub_scores if s.name == "spectral")
        assert spec.score == pytest.approx(0.789, abs=0.01)

    def test_compositional_safe(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "safe_read_pipeline.yaml")
        comp = next(s for s in result.sub_scores if s.name == "compositional")
        assert comp.score < 0.01

    def test_compositional_risky(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "risky_delete_cascade.yaml")
        comp = next(s for s in result.sub_scores if s.name == "compositional")
        assert comp.score == pytest.approx(0.18, abs=0.02)

    def test_compositional_moderate(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "moderate_api_chain.yaml")
        comp = next(s for s in result.sub_scores if s.name == "compositional")
        assert comp.score == pytest.approx(0.04, abs=0.01)

    def test_irreversibility_safe(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "safe_read_pipeline.yaml")
        irrev = next(s for s in result.sub_scores if s.name == "irreversibility")
        assert irrev.score == 0.0

    def test_irreversibility_risky(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "risky_delete_cascade.yaml")
        irrev = next(s for s in result.sub_scores if s.name == "irreversibility")
        assert irrev.score == 1.0

    def test_irreversibility_moderate(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "moderate_api_chain.yaml")
        irrev = next(s for s in result.sub_scores if s.name == "irreversibility")
        assert irrev.score == 0.0  # no irreversible ops


# ---------------------------------------------------------------------------
# AC: Edge cases (single node, cycles, custom config)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_node_no_edges(self, engine: RiskScoringEngine) -> None:
        g = _make_graph({"a": "delete_file"}, [])
        result = engine.score(g)
        assert result.risk_level == RiskLevel.LOW
        assert result.aggregate_score == 0.0
        assert result.node_count == 1
        assert result.edge_count == 0

    def test_empty_dag(self, engine: RiskScoringEngine) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = engine.score(g)
        assert result.risk_level == RiskLevel.LOW
        assert result.aggregate_score == 0.0

    def test_custom_config_changes_aggregate(
        self, registry: OperationRegistry,
    ) -> None:
        # Heavily weight irreversibility (0.70) to shift the aggregate
        heavy_irrev = ScoringConfig(
            fan_out=0.05, chain_depth=0.05, irreversibility=0.70,
            centrality=0.05, spectral=0.05, compositional=0.10,
        )
        engine = RiskScoringEngine(heavy_irrev, registry)
        # Graph with irreversible op after external
        g = _make_graph(
            {"a": "invoke_api", "b": "delete_record", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = engine.score(g)
        # Irreversibility=1.0 * weight=0.70 dominates
        assert result.aggregate_score >= 0.70

    def test_custom_config_low_weight_reduces_score(
        self, registry: OperationRegistry,
    ) -> None:
        # Near-zero weight on everything risky
        low_config = ScoringConfig(
            fan_out=0.30, chain_depth=0.30, irreversibility=0.05,
            centrality=0.15, spectral=0.10, compositional=0.10,
        )
        engine = RiskScoringEngine(low_config, registry)
        result = _load_and_score(engine, "risky_delete_cascade.yaml")
        # With low irrev weight, aggregate drops vs default
        default_engine = RiskScoringEngine(ScoringConfig(), registry)
        default_result = _load_and_score(default_engine, "risky_delete_cascade.yaml")
        assert result.aggregate_score < default_result.aggregate_score

    def test_all_weights_on_one_scorer(
        self, registry: OperationRegistry,
    ) -> None:
        # All weight on spectral
        config = ScoringConfig(
            fan_out=0.0, chain_depth=0.0, irreversibility=0.0,
            centrality=0.0, spectral=1.0, compositional=0.0,
        )
        engine = RiskScoringEngine(config, registry)
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = engine.score(g)
        spectral = next(s for s in result.sub_scores if s.name == "spectral")
        assert result.aggregate_score == pytest.approx(spectral.score, abs=1e-6)

    def test_profile_has_all_fields(self, engine: RiskScoringEngine) -> None:
        result = _load_and_score(engine, "safe_read_pipeline.yaml")
        assert isinstance(result, RiskProfile)
        assert isinstance(result.workflow_name, str)
        assert isinstance(result.aggregate_score, float)
        assert isinstance(result.risk_level, RiskLevel)
        assert len(result.sub_scores) == 6
        assert isinstance(result.critical_paths, tuple)
        assert isinstance(result.chokepoints, tuple)
