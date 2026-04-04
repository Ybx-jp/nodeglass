"""Integration tests for the mitigation subsystem (NOD-28).

NOD-28 spec (Linear):
- tests/test_mitigation.py
- Each strategy rule produces expected mitigations for known DAG patterns
- Engine deduplication: same strategy triggered twice -> single mitigation
- Residual risk estimation: verify < original aggregate
- Edge case: low-risk DAG -> no required mitigations
- Edge case: DAG with only pure ops -> minimal/no mitigations
- Integration: score a risky DAG, generate plan, verify specific mitigations present

AC:
- [ ] pytest tests/test_mitigation.py -v passes
- [ ] Covers all strategy rules (irreversible, external, fan-out, credentials, user-facing)
- [ ] Covers engine deduplication
- [ ] Covers residual risk estimation
- [ ] Covers edge case: no mitigations needed

Unit tests for strategies live in test_mitigation_strategies.py (NOD-26, 34 tests).
Unit tests for the engine live in test_mitigation_engine.py (NOD-27, 17 tests).
This file adds integration tests exercising the full scoring -> mitigation pipeline.
"""

import networkx as nx
import pytest

from workflow_eval.dag.builder import DAGBuilder
from workflow_eval.dag.models import to_networkx
from workflow_eval.mitigation.engine import MitigationEngine
from workflow_eval.mitigation.strategies import (
    MitigateCredentialAccess,
    MitigateExternalOps,
    MitigateHighFanOut,
    MitigateHighRiskExternal,
    MitigateIrreversibleOps,
    MitigateUncertainPredecessors,
    MitigateUserFacingOps,
)
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.engine import RiskScoringEngine
from workflow_eval.types import (
    MitigationAction,
    MitigationPriority,
    RiskLevel,
    RiskProfile,
    ScoringConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> OperationRegistry:
    return get_default_registry()


@pytest.fixture()
def scoring_engine(registry: OperationRegistry) -> RiskScoringEngine:
    return RiskScoringEngine(ScoringConfig(), registry)


@pytest.fixture()
def mitigation_engine() -> MitigationEngine:
    return MitigationEngine()


def _build_dag(nodes: list[tuple[str, str]], edges: list[tuple[str, str]]) -> nx.DiGraph:
    dag = nx.DiGraph()
    for nid, op in nodes:
        dag.add_node(nid, operation=op)
    dag.add_edges_from(edges)
    return dag


# ---------------------------------------------------------------------------
# AC: Covers all strategy rules
# ---------------------------------------------------------------------------


class TestStrategyRules:
    """Each strategy rule produces expected mitigations for known patterns."""

    def test_irreversible_produces_confirmation_and_rollback(self, registry: OperationRegistry) -> None:
        dag = _build_dag(nodes=[("del", "delete_record")], edges=[])
        profile = RiskProfile(
            workflow_name="t", aggregate_score=0.5, risk_level=RiskLevel.MEDIUM,
            sub_scores=(), node_count=1, edge_count=0,
        )
        result = MitigateIrreversibleOps()(dag, registry, profile)
        actions = {m.action for m in result}
        assert MitigationAction.ADD_CONFIRMATION in actions
        assert MitigationAction.ADD_ROLLBACK in actions

    def test_external_produces_sandbox(self, registry: OperationRegistry) -> None:
        dag = _build_dag(nodes=[("api", "invoke_api")], edges=[])
        profile = RiskProfile(
            workflow_name="t", aggregate_score=0.2, risk_level=RiskLevel.LOW,
            sub_scores=(), node_count=1, edge_count=0,
        )
        result = MitigateExternalOps()(dag, registry, profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.SANDBOX_EXTERNAL

    def test_fan_out_produces_reduce_parallelism(self, registry: OperationRegistry) -> None:
        dag = _build_dag(
            nodes=[("hub", "invoke_api"), ("a", "read_file"), ("b", "read_file"),
                   ("c", "read_file"), ("d", "read_file")],
            edges=[("hub", "a"), ("hub", "b"), ("hub", "c"), ("hub", "d")],
        )
        profile = RiskProfile(
            workflow_name="t", aggregate_score=0.3, risk_level=RiskLevel.MEDIUM,
            sub_scores=(), node_count=5, edge_count=4,
        )
        result = MitigateHighFanOut()(dag, registry, profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.REDUCE_PARALLELISM

    def test_credentials_produces_audit_log(self, registry: OperationRegistry) -> None:
        dag = _build_dag(nodes=[("auth", "authenticate")], edges=[])
        profile = RiskProfile(
            workflow_name="t", aggregate_score=0.2, risk_level=RiskLevel.LOW,
            sub_scores=(), node_count=1, edge_count=0,
        )
        result = MitigateCredentialAccess()(dag, registry, profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.ADD_AUDIT_LOG

    def test_user_facing_produces_require_auth(self, registry: OperationRegistry) -> None:
        dag = _build_dag(nodes=[("email", "send_email")], edges=[])
        profile = RiskProfile(
            workflow_name="t", aggregate_score=0.3, risk_level=RiskLevel.MEDIUM,
            sub_scores=(), node_count=1, edge_count=0,
        )
        result = MitigateUserFacingOps()(dag, registry, profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.REQUIRE_AUTHENTICATION

    def test_high_risk_external_produces_rate_limit(self, registry: OperationRegistry) -> None:
        dag = _build_dag(nodes=[("exec", "execute_code")], edges=[])
        profile = RiskProfile(
            workflow_name="t", aggregate_score=0.4, risk_level=RiskLevel.MEDIUM,
            sub_scores=(), node_count=1, edge_count=0,
        )
        result = MitigateHighRiskExternal()(dag, registry, profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.ADD_RATE_LIMIT

    def test_uncertain_predecessor_produces_retry(self, registry: OperationRegistry) -> None:
        dag = _build_dag(
            nodes=[("api", "invoke_api"), ("del", "delete_record")],
            edges=[("api", "del")],
        )
        profile = RiskProfile(
            workflow_name="t", aggregate_score=0.5, risk_level=RiskLevel.MEDIUM,
            sub_scores=(), node_count=2, edge_count=1,
        )
        result = MitigateUncertainPredecessors()(dag, registry, profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.ADD_RETRY


# ---------------------------------------------------------------------------
# AC: Covers engine deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_duplicate_strategy_deduplicates(self, registry: OperationRegistry) -> None:
        """Same strategy run twice should not produce duplicate mitigations."""
        engine = MitigationEngine(
            strategies=(MitigateIrreversibleOps(), MitigateIrreversibleOps()),
        )
        dag = _build_dag(nodes=[("del", "delete_record")], edges=[])
        profile = RiskProfile(
            workflow_name="t", aggregate_score=0.5, risk_level=RiskLevel.MEDIUM,
            sub_scores=(), node_count=1, edge_count=0,
        )
        plan = engine.generate_plan(profile, dag, registry)
        keys = [(m.action, frozenset(m.target_node_ids)) for m in plan.mitigations]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# AC: Covers residual risk estimation
# ---------------------------------------------------------------------------


class TestResidualRisk:
    def test_residual_less_than_original(
        self, mitigation_engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        dag = _build_dag(
            nodes=[("auth", "authenticate"), ("del", "delete_record")],
            edges=[("auth", "del")],
        )
        profile = RiskProfile(
            workflow_name="t", aggregate_score=0.8, risk_level=RiskLevel.CRITICAL,
            sub_scores=(), node_count=2, edge_count=1,
        )
        plan = mitigation_engine.generate_plan(profile, dag, registry)
        assert plan.residual_risk < plan.original_risk

    def test_residual_formula(
        self, mitigation_engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        dag = _build_dag(
            nodes=[("auth", "authenticate"), ("del", "delete_record")],
            edges=[("auth", "del")],
        )
        profile = RiskProfile(
            workflow_name="t", aggregate_score=0.8, risk_level=RiskLevel.CRITICAL,
            sub_scores=(), node_count=2, edge_count=1,
        )
        plan = mitigation_engine.generate_plan(profile, dag, registry)
        required_count = sum(
            1 for m in plan.mitigations if m.priority == MitigationPriority.REQUIRED
        )
        assert plan.residual_risk == pytest.approx(0.8 * (0.5 ** required_count))


# ---------------------------------------------------------------------------
# AC: Covers edge case: no mitigations needed
# ---------------------------------------------------------------------------


class TestNoMitigationsNeeded:
    def test_pure_ops_no_mitigations(
        self, mitigation_engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        dag = _build_dag(
            nodes=[("a", "read_file"), ("b", "read_state"), ("c", "branch")],
            edges=[("a", "b"), ("b", "c")],
        )
        profile = RiskProfile(
            workflow_name="pure", aggregate_score=0.05, risk_level=RiskLevel.LOW,
            sub_scores=(), node_count=3, edge_count=2,
        )
        plan = mitigation_engine.generate_plan(profile, dag, registry)
        assert len(plan.mitigations) == 0

    def test_low_risk_no_required(
        self, mitigation_engine: MitigationEngine, registry: OperationRegistry,
    ) -> None:
        """A single low-risk external op should produce no required mitigations."""
        dag = _build_dag(
            nodes=[("a", "read_file"), ("b", "invoke_api")],
            edges=[("a", "b")],
        )
        profile = RiskProfile(
            workflow_name="low", aggregate_score=0.10, risk_level=RiskLevel.LOW,
            sub_scores=(), node_count=2, edge_count=1,
        )
        plan = mitigation_engine.generate_plan(profile, dag, registry)
        for m in plan.mitigations:
            assert m.priority != MitigationPriority.REQUIRED


# ---------------------------------------------------------------------------
# Integration: score a risky DAG, generate plan, verify specific mitigations
# ---------------------------------------------------------------------------


class TestIntegration:
    """Full pipeline: build DAG -> score -> mitigate -> verify."""

    def test_delete_cascade_full_pipeline(
        self,
        scoring_engine: RiskScoringEngine,
        mitigation_engine: MitigationEngine,
        registry: OperationRegistry,
    ) -> None:
        """Build a delete cascade, score it, generate mitigations, verify plan."""
        wf = (
            DAGBuilder("risky-delete-cascade")
            .add_step("auth", "authenticate", params={"method": "mfa"})
            .then("lookup", "invoke_api", params={"endpoint": "/users"})
            .then("delete_posts", "delete_record", params={"table": "posts"})
            .then("delete_files", "delete_file", params={"path": "/uploads"})
            .then("notify", "send_email", params={"to": "admin"})
            .build()
        )
        dag = to_networkx(wf)

        # Score the workflow
        risk_profile = scoring_engine.score(dag)
        assert risk_profile.aggregate_score > 0

        # Generate mitigation plan
        plan = mitigation_engine.generate_plan(risk_profile, dag, registry)

        # Verify specific mitigations present
        actions = {m.action for m in plan.mitigations}

        # Irreversible ops (delete_record, delete_file, send_email) -> confirmation
        assert MitigationAction.ADD_CONFIRMATION in actions

        # Credential access (authenticate) -> audit log
        assert MitigationAction.ADD_AUDIT_LOG in actions

        # External ops (invoke_api, authenticate, execute_code) -> sandbox
        assert MitigationAction.SANDBOX_EXTERNAL in actions

        # Residual risk should be lower than original
        assert plan.residual_risk < plan.original_risk

        # Sorted: required first
        priorities = [m.priority for m in plan.mitigations]
        order = {MitigationPriority.REQUIRED: 0, MitigationPriority.RECOMMENDED: 1, MitigationPriority.OPTIONAL: 2}
        assert [order[p] for p in priorities] == sorted(order[p] for p in priorities)

    def test_pure_pipeline_minimal_plan(
        self,
        scoring_engine: RiskScoringEngine,
        mitigation_engine: MitigationEngine,
        registry: OperationRegistry,
    ) -> None:
        """A pure-only pipeline should produce an empty or optional-only plan."""
        wf = (
            DAGBuilder("safe-pipeline")
            .add_step("read", "read_file")
            .then("state", "read_state")
            .then("branch", "branch")
            .build()
        )
        dag = to_networkx(wf)

        risk_profile = scoring_engine.score(dag)
        plan = mitigation_engine.generate_plan(risk_profile, dag, registry)

        # No required mitigations for pure ops
        required = [m for m in plan.mitigations if m.priority == MitigationPriority.REQUIRED]
        assert len(required) == 0

    def test_fan_out_destruction_pipeline(
        self,
        scoring_engine: RiskScoringEngine,
        mitigation_engine: MitigationEngine,
        registry: OperationRegistry,
    ) -> None:
        """A fan-out to 4+ destruction ops should trigger reduce_parallelism."""
        wf = (
            DAGBuilder("infra-teardown")
            .add_step("auth", "authenticate")
            .then("exec", "execute_code")
            .parallel(
                ["d1", "d2", "d3", "d4"],
                operation="destroy_resource",
            )
            .join("notify", "send_notification")
            .build()
        )
        dag = to_networkx(wf)

        risk_profile = scoring_engine.score(dag)
        plan = mitigation_engine.generate_plan(risk_profile, dag, registry)

        actions = {m.action for m in plan.mitigations}
        assert MitigationAction.REDUCE_PARALLELISM in actions
        assert MitigationAction.ADD_CONFIRMATION in actions

        # exec fans out to 4 destroy ops -> reduce_parallelism targets exec
        fan_out_mitigations = [
            m for m in plan.mitigations
            if m.action == MitigationAction.REDUCE_PARALLELISM
        ]
        fan_out_targets = {nid for m in fan_out_mitigations for nid in m.target_node_ids}
        assert "exec" in fan_out_targets
