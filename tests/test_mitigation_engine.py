"""Tests for MitigationEngine (NOD-27).

AC:
- [x] Low-risk DAG -> empty mitigations or optional-only
- [x] risky_delete_cascade -> multiple required mitigations (add_confirmation on deletes, add_audit_log on auth)
- [x] Residual risk < original aggregate risk
- [x] No duplicate mitigations in output
- [x] Mitigations sorted: required first, then recommended, then optional
"""

import networkx as nx
import pytest

from workflow_eval.mitigation.engine import MitigationEngine
from workflow_eval.mitigation.strategies import (
    MitigateExternalOps,
    MitigateIrreversibleOps,
    get_default_strategies,
)
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import (
    Mitigation,
    MitigationAction,
    MitigationPlan,
    MitigationPriority,
    RiskLevel,
    RiskProfile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> OperationRegistry:
    return get_default_registry()


@pytest.fixture()
def engine() -> MitigationEngine:
    return MitigationEngine()


def _build_dag(nodes: list[tuple[str, str]], edges: list[tuple[str, str]]) -> nx.DiGraph:
    dag = nx.DiGraph()
    for nid, op in nodes:
        dag.add_node(nid, operation=op)
    dag.add_edges_from(edges)
    return dag


def _low_risk_profile() -> RiskProfile:
    return RiskProfile(
        workflow_name="low-risk",
        aggregate_score=0.10,
        risk_level=RiskLevel.LOW,
        sub_scores=(),
        node_count=2,
        edge_count=1,
    )


def _high_risk_profile() -> RiskProfile:
    return RiskProfile(
        workflow_name="risky-delete-cascade",
        aggregate_score=0.80,
        risk_level=RiskLevel.CRITICAL,
        sub_scores=(),
        node_count=5,
        edge_count=4,
    )


# ---------------------------------------------------------------------------
# AC: Low-risk DAG -> empty mitigations or optional-only
# ---------------------------------------------------------------------------


class TestLowRiskDAG:
    def test_pure_only_dag_empty_plan(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        """A DAG with only pure ops should produce no mitigations."""
        dag = _build_dag(
            nodes=[("a", "read_file"), ("b", "read_state")],
            edges=[("a", "b")],
        )
        plan = engine.generate_plan(_low_risk_profile(), dag, registry)
        assert len(plan.mitigations) == 0

    def test_low_risk_no_required(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        """A low-risk DAG with a single external op gets only recommended/optional."""
        dag = _build_dag(
            nodes=[("a", "read_file"), ("b", "invoke_api")],
            edges=[("a", "b")],
        )
        plan = engine.generate_plan(_low_risk_profile(), dag, registry)
        for m in plan.mitigations:
            assert m.priority in (MitigationPriority.RECOMMENDED, MitigationPriority.OPTIONAL)


# ---------------------------------------------------------------------------
# AC: risky_delete_cascade -> multiple required mitigations
# ---------------------------------------------------------------------------


class TestRiskyDeleteCascade:
    def _cascade_dag(self) -> nx.DiGraph:
        """auth -> lookup -> delete_posts -> delete_files (classic delete cascade)."""
        return _build_dag(
            nodes=[
                ("auth", "authenticate"),
                ("lookup", "invoke_api"),
                ("delete_posts", "delete_record"),
                ("delete_files", "delete_file"),
            ],
            edges=[
                ("auth", "lookup"),
                ("lookup", "delete_posts"),
                ("delete_posts", "delete_files"),
            ],
        )

    def test_has_required_confirmations(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        plan = engine.generate_plan(
            _high_risk_profile(), self._cascade_dag(), registry,
        )
        confirmations = [
            m for m in plan.mitigations
            if m.action == MitigationAction.ADD_CONFIRMATION
        ]
        # Both delete_record and delete_file are irreversible
        confirmed_nodes = {nid for m in confirmations for nid in m.target_node_ids}
        assert confirmed_nodes == {"delete_posts", "delete_files"}

    def test_has_audit_on_auth(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        plan = engine.generate_plan(
            _high_risk_profile(), self._cascade_dag(), registry,
        )
        audits = [
            m for m in plan.mitigations
            if m.action == MitigationAction.ADD_AUDIT_LOG
        ]
        audit_nodes = {nid for m in audits for nid in m.target_node_ids}
        assert audit_nodes == {"auth"}

    def test_multiple_required(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        plan = engine.generate_plan(
            _high_risk_profile(), self._cascade_dag(), registry,
        )
        required = [m for m in plan.mitigations if m.priority == MitigationPriority.REQUIRED]
        # Exactly: add_confirmation on 2 deletes + add_audit_log on auth = 3
        assert len(required) == 3


# ---------------------------------------------------------------------------
# AC: Residual risk < original aggregate risk
# ---------------------------------------------------------------------------


class TestResidualRisk:
    def test_residual_less_than_original(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        dag = _build_dag(
            nodes=[("auth", "authenticate"), ("del", "delete_record")],
            edges=[("auth", "del")],
        )
        profile = _high_risk_profile()
        plan = engine.generate_plan(profile, dag, registry)
        assert plan.residual_risk < plan.original_risk

    def test_residual_halves_per_required(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        """Residual = aggregate * 0.5^required_count."""
        dag = _build_dag(
            nodes=[("auth", "authenticate"), ("del", "delete_record")],
            edges=[("auth", "del")],
        )
        profile = _high_risk_profile()
        plan = engine.generate_plan(profile, dag, registry)
        required_count = sum(
            1 for m in plan.mitigations if m.priority == MitigationPriority.REQUIRED
        )
        expected = profile.aggregate_score * (0.5 ** required_count)
        assert plan.residual_risk == pytest.approx(expected)

    def test_no_required_residual_unchanged(
        self, registry: OperationRegistry,
    ) -> None:
        """With no required mitigations, residual == original."""
        dag = _build_dag(
            nodes=[("a", "read_file"), ("b", "invoke_api")],
            edges=[("a", "b")],
        )
        engine = MitigationEngine()
        plan = engine.generate_plan(_low_risk_profile(), dag, registry)
        required = [m for m in plan.mitigations if m.priority == MitigationPriority.REQUIRED]
        assert len(required) == 0
        assert plan.residual_risk == plan.original_risk

    def test_zero_aggregate_stays_zero(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        """0.0 * anything = 0.0 — pure DAGs don't invent residual risk."""
        dag = _build_dag(nodes=[("a", "read_file")], edges=[])
        profile = RiskProfile(
            workflow_name="zero",
            aggregate_score=0.0,
            risk_level=RiskLevel.LOW,
            sub_scores=(),
            node_count=1,
            edge_count=0,
        )
        plan = engine.generate_plan(profile, dag, registry)
        assert plan.residual_risk == 0.0


# ---------------------------------------------------------------------------
# AC: No duplicate mitigations in output
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_no_duplicates(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        dag = _build_dag(
            nodes=[("auth", "authenticate"), ("del", "delete_record")],
            edges=[("auth", "del")],
        )
        plan = engine.generate_plan(_high_risk_profile(), dag, registry)
        keys = [(m.action, frozenset(m.target_node_ids)) for m in plan.mitigations]
        assert len(keys) == len(set(keys))

    def test_duplicate_strategies_deduplicated(
        self, registry: OperationRegistry,
    ) -> None:
        """Running the same strategy twice should still produce unique output."""
        # Use two copies of the irreversible strategy
        double_strategies = (MitigateIrreversibleOps(), MitigateIrreversibleOps())
        engine = MitigationEngine(strategies=double_strategies)
        dag = _build_dag(nodes=[("del", "delete_record")], edges=[])
        plan = engine.generate_plan(_high_risk_profile(), dag, registry)
        # Should have exactly 2 (confirmation + rollback), not 4
        assert len(plan.mitigations) == 2

    def test_order_independent_dedup(
        self, registry: OperationRegistry,
    ) -> None:
        """Target nodes ("a","b") and ("b","a") should deduplicate."""
        m1 = Mitigation(
            action=MitigationAction.ADD_RETRY,
            priority=MitigationPriority.RECOMMENDED,
            target_node_ids=("a", "b"),
            reason="test",
        )
        m2 = Mitigation(
            action=MitigationAction.ADD_RETRY,
            priority=MitigationPriority.RECOMMENDED,
            target_node_ids=("b", "a"),
            reason="test duplicate",
        )

        class FakeStrategy:
            name = "fake"
            def __call__(self, dag, registry, risk_profile):
                return [m1, m2]

        engine = MitigationEngine(strategies=(FakeStrategy(),))
        dag = nx.DiGraph()
        profile = _low_risk_profile()
        plan = engine.generate_plan(profile, dag, registry)
        assert len(plan.mitigations) == 1
        # First-wins: the surviving mitigation should have the first reason
        assert plan.mitigations[0].reason == "test"


# ---------------------------------------------------------------------------
# AC: Mitigations sorted: required first, then recommended, then optional
# ---------------------------------------------------------------------------


class TestSorting:
    def test_required_before_recommended_before_optional(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        """Full cascade DAG should have required → recommended → optional ordering."""
        dag = _build_dag(
            nodes=[
                ("auth", "authenticate"),
                ("exec", "execute_code"),
                ("hub", "invoke_api"),
                ("d1", "delete_record"),
                ("d2", "delete_record"),
                ("d3", "destroy_resource"),
                ("d4", "destroy_resource"),
                ("email", "send_email"),
            ],
            edges=[
                ("auth", "exec"),
                ("exec", "hub"),
                ("hub", "d1"),
                ("hub", "d2"),
                ("hub", "d3"),
                ("hub", "d4"),
                ("hub", "email"),
            ],
        )
        plan = engine.generate_plan(_high_risk_profile(), dag, registry)

        priorities = [m.priority for m in plan.mitigations]
        priority_order = {
            MitigationPriority.REQUIRED: 0,
            MitigationPriority.RECOMMENDED: 1,
            MitigationPriority.OPTIONAL: 2,
        }
        numeric = [priority_order[p] for p in priorities]
        assert numeric == sorted(numeric), (
            f"Mitigations not sorted by priority: {priorities}"
        )

    def test_has_all_three_priority_levels(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        """A rich DAG should produce mitigations at all three priority levels."""
        dag = _build_dag(
            nodes=[
                ("auth", "authenticate"),
                ("exec", "execute_code"),
                ("hub", "invoke_api"),
                ("d1", "delete_record"),
                ("d2", "delete_record"),
                ("d3", "destroy_resource"),
                ("d4", "destroy_resource"),
                ("email", "send_email"),
            ],
            edges=[
                ("auth", "exec"),
                ("exec", "hub"),
                ("hub", "d1"),
                ("hub", "d2"),
                ("hub", "d3"),
                ("hub", "d4"),
                ("hub", "email"),
            ],
        )
        plan = engine.generate_plan(_high_risk_profile(), dag, registry)
        present = {m.priority for m in plan.mitigations}
        assert MitigationPriority.REQUIRED in present
        assert MitigationPriority.RECOMMENDED in present
        assert MitigationPriority.OPTIONAL in present


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dag(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        dag = nx.DiGraph()
        plan = engine.generate_plan(_low_risk_profile(), dag, registry)
        assert len(plan.mitigations) == 0
        assert plan.original_risk == 0.10
        assert plan.residual_risk == 0.10

    def test_custom_strategies(self, registry: OperationRegistry) -> None:
        """Engine accepts custom strategy set."""
        engine = MitigationEngine(strategies=(MitigateExternalOps(),))
        dag = _build_dag(
            nodes=[("api", "invoke_api"), ("del", "delete_record")],
            edges=[("api", "del")],
        )
        plan = engine.generate_plan(_low_risk_profile(), dag, registry)
        # Only external strategy runs -> only sandbox_external
        actions = {m.action for m in plan.mitigations}
        assert actions == {MitigationAction.SANDBOX_EXTERNAL}

    def test_plan_fields_populated(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        dag = _build_dag(
            nodes=[("auth", "authenticate"), ("del", "delete_record")],
            edges=[("auth", "del")],
        )
        profile = _high_risk_profile()
        plan = engine.generate_plan(profile, dag, registry)
        assert isinstance(plan, MitigationPlan)
        assert isinstance(plan.mitigations, tuple)
        assert plan.original_risk == profile.aggregate_score
        # auth(CREDENTIALS) + del(IRREVERSIBLE) → 2 required mitigations
        # residual = 0.80 * (0.5^2) = 0.20
        assert plan.residual_risk == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Coverage gaps (review fixes)
# ---------------------------------------------------------------------------


class TestCoverageGaps:
    def test_unknown_op_error_propagates(
        self, engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        """Engine propagates KeyError for unknown operations."""
        dag = nx.DiGraph()
        dag.add_node("x", operation="nonexistent_op")
        with pytest.raises(KeyError, match="nonexistent_op"):
            engine.generate_plan(_high_risk_profile(), dag, registry)

    def test_empty_strategies_tuple_runs_nothing(self, registry: OperationRegistry) -> None:
        """MitigationEngine(strategies=()) should run zero strategies, not defaults."""
        engine = MitigationEngine(strategies=())
        dag = _build_dag(
            nodes=[("auth", "authenticate"), ("del", "delete_record")],
            edges=[("auth", "del")],
        )
        plan = engine.generate_plan(_high_risk_profile(), dag, registry)
        assert len(plan.mitigations) == 0
        assert plan.residual_risk == plan.original_risk
