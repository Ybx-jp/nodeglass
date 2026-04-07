"""Tests for mitigation models and strategies (NOD-26).

AC:
- [x] Each MitigationAction enum value is produced by at least one strategy
- [x] Strategies are independent and composable (order doesn't matter)
- [x] Each Mitigation has action, target_node_ids, reason, and priority
- [x] Priority levels: required > recommended > optional
"""

import networkx as nx
import pytest

from workflow_eval.mitigation.models import MitigationStrategy
from workflow_eval.mitigation.strategies import (
    MitigateCredentialAccess,
    MitigateExternalOps,
    MitigateHighFanOut,
    MitigateHighRiskExternal,
    MitigateIrreversibleOps,
    MitigateUncertainPredecessors,
    MitigateUserFacingOps,
    get_default_strategies,
)
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.effect_types import EffectTarget, EffectType
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import (
    Mitigation,
    MitigationAction,
    MitigationPriority,
    OperationDefinition,
    RiskLevel,
    RiskProfile,
    SubScore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> OperationRegistry:
    return get_default_registry()


@pytest.fixture()
def empty_profile() -> RiskProfile:
    """Minimal risk profile for strategy calls that don't inspect it."""
    return RiskProfile(
        workflow_name="test",
        aggregate_score=0.0,
        risk_level=RiskLevel.LOW,
        sub_scores=(),
        node_count=0,
        edge_count=0,
    )


def _build_dag(nodes: list[tuple[str, str]], edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Build a DAG from (id, operation) node tuples and (src, tgt) edge tuples."""
    dag = nx.DiGraph()
    for nid, op in nodes:
        dag.add_node(nid, operation=op)
    dag.add_edges_from(edges)
    return dag


# ---------------------------------------------------------------------------
# AC: Each MitigationAction enum value is produced by at least one strategy
# ---------------------------------------------------------------------------


class TestActionCoverage:
    """Every MitigationAction must be emitted by at least one strategy."""

    def test_all_actions_covered(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        """Run all strategies on a workflow that triggers every rule and check coverage."""
        # Build a DAG that triggers all 7 strategies:
        # - authenticate (EXTERNAL, CREDENTIALS) -> triggers credential_access + external + sandbox
        # - execute_code (EXTERNAL, risk=0.60) -> triggers high_risk_external + external
        # - hub (fan-out=4) -> triggers high_fan_out
        # - delete_record (IRREVERSIBLE) -> triggers irreversible + uncertain predecessor
        # - send_email (IRREVERSIBLE, USER_FACING) -> triggers user_facing + irreversible
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

        strategies = get_default_strategies()
        all_mitigations: list[Mitigation] = []
        for strategy in strategies:
            all_mitigations.extend(strategy(dag, registry, empty_profile))

        produced_actions = {m.action for m in all_mitigations}
        for action in MitigationAction:
            assert action in produced_actions, (
                f"MitigationAction.{action.name} not produced by any strategy"
            )


# ---------------------------------------------------------------------------
# AC: Strategies are independent and composable (order doesn't matter)
# ---------------------------------------------------------------------------


class TestComposability:
    """Strategies must be independent — order of execution doesn't matter."""

    def test_order_independence(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(
            nodes=[
                ("auth", "authenticate"),
                ("del", "delete_record"),
            ],
            edges=[("auth", "del")],
        )

        strategies = get_default_strategies()
        forward: list[Mitigation] = []
        for s in strategies:
            forward.extend(s(dag, registry, empty_profile))

        reverse: list[Mitigation] = []
        for s in reversed(strategies):
            reverse.extend(s(dag, registry, empty_profile))

        # Same mitigations regardless of strategy order (sort for comparison)
        assert sorted(forward, key=lambda m: (m.action, m.target_node_ids)) == \
               sorted(reverse, key=lambda m: (m.action, m.target_node_ids))

    def test_single_strategy_isolation(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        """Each strategy runs without needing output from others."""
        dag = _build_dag(
            nodes=[("del", "delete_record")],
            edges=[],
        )
        # Running a single strategy in isolation should work fine
        result = MitigateIrreversibleOps()(dag, registry, empty_profile)
        assert len(result) == 2  # confirmation + rollback


# ---------------------------------------------------------------------------
# AC: Each Mitigation has action, target_node_ids, reason, and priority
# ---------------------------------------------------------------------------


class TestMitigationFields:
    """Every emitted Mitigation must have all four required fields."""

    def test_fields_present(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(
            nodes=[
                ("auth", "authenticate"),
                ("del", "delete_record"),
            ],
            edges=[("auth", "del")],
        )

        strategies = get_default_strategies()
        for strategy in strategies:
            for m in strategy(dag, registry, empty_profile):
                assert isinstance(m.action, MitigationAction)
                assert isinstance(m.priority, MitigationPriority)
                assert isinstance(m.target_node_ids, tuple)
                assert len(m.target_node_ids) > 0
                assert isinstance(m.reason, str)
                assert len(m.reason) > 0


# ---------------------------------------------------------------------------
# AC: Priority levels: required > recommended > optional
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    """Priority semantics: required > recommended > optional."""

    def test_irreversible_confirmation_is_required(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("del", "delete_record")], edges=[])
        result = MitigateIrreversibleOps()(dag, registry, empty_profile)
        confirmations = [m for m in result if m.action == MitigationAction.ADD_CONFIRMATION]
        assert all(m.priority == MitigationPriority.REQUIRED for m in confirmations)

    def test_irreversible_rollback_is_recommended(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("del", "delete_record")], edges=[])
        result = MitigateIrreversibleOps()(dag, registry, empty_profile)
        rollbacks = [m for m in result if m.action == MitigationAction.ADD_ROLLBACK]
        assert all(m.priority == MitigationPriority.RECOMMENDED for m in rollbacks)

    def test_credential_audit_is_required(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("auth", "authenticate")], edges=[])
        result = MitigateCredentialAccess()(dag, registry, empty_profile)
        assert all(m.priority == MitigationPriority.REQUIRED for m in result)

    def test_external_sandbox_is_recommended(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("api", "invoke_api")], edges=[])
        result = MitigateExternalOps()(dag, registry, empty_profile)
        assert all(m.priority == MitigationPriority.RECOMMENDED for m in result)

    def test_fan_out_is_optional(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(
            nodes=[("hub", "invoke_api"), ("a", "read_file"), ("b", "read_file"),
                   ("c", "read_file"), ("d", "read_file")],
            edges=[("hub", "a"), ("hub", "b"), ("hub", "c"), ("hub", "d")],
        )
        result = MitigateHighFanOut()(dag, registry, empty_profile)
        assert all(m.priority == MitigationPriority.OPTIONAL for m in result)

    def test_rate_limit_is_optional(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("exec", "execute_code")], edges=[])
        result = MitigateHighRiskExternal()(dag, registry, empty_profile)
        assert all(m.priority == MitigationPriority.OPTIONAL for m in result)

    def test_retry_is_recommended(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(
            nodes=[("api", "invoke_api"), ("del", "delete_record")],
            edges=[("api", "del")],
        )
        result = MitigateUncertainPredecessors()(dag, registry, empty_profile)
        assert all(m.priority == MitigationPriority.RECOMMENDED for m in result)

    def test_user_facing_auth_is_recommended(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("email", "send_email")], edges=[])
        result = MitigateUserFacingOps()(dag, registry, empty_profile)
        assert all(m.priority == MitigationPriority.RECOMMENDED for m in result)


# ---------------------------------------------------------------------------
# Individual strategy behavior
# ---------------------------------------------------------------------------


class TestIrreversibleStrategy:
    def test_emits_two_mitigations_per_node(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("del", "delete_record")], edges=[])
        result = MitigateIrreversibleOps()(dag, registry, empty_profile)
        assert len(result) == 2
        actions = {m.action for m in result}
        assert actions == {MitigationAction.ADD_CONFIRMATION, MitigationAction.ADD_ROLLBACK}

    def test_ignores_non_irreversible(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("api", "invoke_api")], edges=[])
        result = MitigateIrreversibleOps()(dag, registry, empty_profile)
        assert result == []

    def test_targets_correct_node(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(
            nodes=[("api", "invoke_api"), ("del", "delete_record")],
            edges=[("api", "del")],
        )
        result = MitigateIrreversibleOps()(dag, registry, empty_profile)
        assert all(m.target_node_ids == ("del",) for m in result)


class TestExternalStrategy:
    def test_emits_for_external_ops(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("api", "invoke_api")], edges=[])
        result = MitigateExternalOps()(dag, registry, empty_profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.SANDBOX_EXTERNAL

    def test_ignores_pure(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("read", "read_file")], edges=[])
        result = MitigateExternalOps()(dag, registry, empty_profile)
        assert result == []


class TestFanOutStrategy:
    def test_triggers_above_threshold(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(
            nodes=[("hub", "invoke_api"), ("a", "read_file"), ("b", "read_file"),
                   ("c", "read_file"), ("d", "read_file")],
            edges=[("hub", "a"), ("hub", "b"), ("hub", "c"), ("hub", "d")],
        )
        result = MitigateHighFanOut()(dag, registry, empty_profile)
        assert len(result) == 1
        assert result[0].target_node_ids == ("hub",)

    def test_no_trigger_at_threshold(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(
            nodes=[("hub", "invoke_api"), ("a", "read_file"), ("b", "read_file"),
                   ("c", "read_file")],
            edges=[("hub", "a"), ("hub", "b"), ("hub", "c")],
        )
        result = MitigateHighFanOut()(dag, registry, empty_profile)
        assert result == []


class TestCredentialStrategy:
    def test_triggers_on_credentials_target(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("auth", "authenticate")], edges=[])
        result = MitigateCredentialAccess()(dag, registry, empty_profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.ADD_AUDIT_LOG

    def test_read_credentials_also_triggers(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("creds", "read_credentials")], edges=[])
        result = MitigateCredentialAccess()(dag, registry, empty_profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.ADD_AUDIT_LOG


class TestUserFacingStrategy:
    def test_triggers_on_user_facing_target(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("email", "send_email")], edges=[])
        result = MitigateUserFacingOps()(dag, registry, empty_profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.REQUIRE_AUTHENTICATION


class TestHighRiskExternalStrategy:
    def test_triggers_above_threshold(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        # execute_code has risk=0.60, EXTERNAL -> should trigger
        dag = _build_dag(nodes=[("exec", "execute_code")], edges=[])
        result = MitigateHighRiskExternal()(dag, registry, empty_profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.ADD_RATE_LIMIT

    def test_ignores_low_risk_external(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        # invoke_api has risk=0.30, EXTERNAL -> below threshold
        dag = _build_dag(nodes=[("api", "invoke_api")], edges=[])
        result = MitigateHighRiskExternal()(dag, registry, empty_profile)
        assert result == []

    def test_ignores_high_risk_non_external(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        # delete_record has risk=0.85, but is IRREVERSIBLE not EXTERNAL
        dag = _build_dag(nodes=[("del", "delete_record")], edges=[])
        result = MitigateHighRiskExternal()(dag, registry, empty_profile)
        assert result == []


class TestUncertainPredecessorStrategy:
    def test_external_predecessor_triggers(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(
            nodes=[("api", "invoke_api"), ("del", "delete_record")],
            edges=[("api", "del")],
        )
        result = MitigateUncertainPredecessors()(dag, registry, empty_profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.ADD_RETRY
        assert result[0].target_node_ids == ("api",)

    def test_stateful_predecessor_triggers(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(
            nodes=[("state", "mutate_state"), ("del", "delete_record")],
            edges=[("state", "del")],
        )
        result = MitigateUncertainPredecessors()(dag, registry, empty_profile)
        assert len(result) == 1
        assert result[0].target_node_ids == ("state",)

    def test_pure_predecessor_no_trigger(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(
            nodes=[("read", "read_file"), ("del", "delete_record")],
            edges=[("read", "del")],
        )
        result = MitigateUncertainPredecessors()(dag, registry, empty_profile)
        assert result == []

    def test_groups_multiple_predecessors(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(
            nodes=[
                ("api1", "invoke_api"),
                ("api2", "send_webhook"),
                ("del", "delete_record"),
            ],
            edges=[("api1", "del"), ("api2", "del")],
        )
        result = MitigateUncertainPredecessors()(dag, registry, empty_profile)
        assert len(result) == 1  # grouped into one mitigation
        assert set(result[0].target_node_ids) == {"api1", "api2"}

    def test_no_irreversible_nodes_empty(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = _build_dag(nodes=[("api", "invoke_api")], edges=[])
        result = MitigateUncertainPredecessors()(dag, registry, empty_profile)
        assert result == []

    def test_irreversible_predecessor_not_uncertain(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        """An irreversible predecessor isn't EXTERNAL/STATEFUL, so no retry."""
        dag = _build_dag(
            nodes=[("del1", "delete_record"), ("del2", "destroy_resource")],
            edges=[("del1", "del2")],
        )
        result = MitigateUncertainPredecessors()(dag, registry, empty_profile)
        assert result == []


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_all_strategies_implement_protocol(self) -> None:
        for strategy in get_default_strategies():
            assert isinstance(strategy, MitigationStrategy)

    def test_all_have_name(self) -> None:
        names = [s.name for s in get_default_strategies()]
        assert len(names) == 7
        assert len(set(names)) == 7  # all unique


# ---------------------------------------------------------------------------
# Empty DAG edge cases
# ---------------------------------------------------------------------------


class TestEmptyDAG:
    def test_all_strategies_return_empty_on_empty_dag(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        dag = nx.DiGraph()
        for strategy in get_default_strategies():
            assert strategy(dag, registry, empty_profile) == []


# ---------------------------------------------------------------------------
# Coverage gaps (review fixes)
# ---------------------------------------------------------------------------


class TestCoverageGaps:
    def test_high_risk_external_boundary_at_exactly_half(self, empty_profile: RiskProfile) -> None:
        """Operation with risk=0.50 and EXTERNAL should trigger (>= boundary)."""
        registry = OperationRegistry()
        registry.register(OperationDefinition(
            name="boundary_external",
            category="test",
            base_risk_weight=0.50,
            effect_type=EffectType.EXTERNAL,
            effect_targets=frozenset({EffectTarget.NETWORK}),
        ))
        dag = _build_dag(nodes=[("b", "boundary_external")], edges=[])
        result = MitigateHighRiskExternal()(dag, registry, empty_profile)
        assert len(result) == 1
        assert result[0].action == MitigationAction.ADD_RATE_LIMIT

    def test_unknown_operation_raises_key_error(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        """Strategies propagate KeyError for unknown operations."""
        dag = nx.DiGraph()
        dag.add_node("x", operation="nonexistent_op")
        with pytest.raises(KeyError, match="nonexistent_op"):
            MitigateIrreversibleOps()(dag, registry, empty_profile)

    def test_reason_contains_node_id_and_op_name(self, registry: OperationRegistry, empty_profile: RiskProfile) -> None:
        """Reason strings include the node ID and operation name."""
        dag = _build_dag(nodes=[("my_node", "delete_record")], edges=[])
        result = MitigateIrreversibleOps()(dag, registry, empty_profile)
        for m in result:
            assert "my_node" in m.reason
            assert "delete_record" in m.reason

    def test_multiple_irreversible_nodes_independent_predecessors(
        self, registry: OperationRegistry, empty_profile: RiskProfile,
    ) -> None:
        """Two irreversible nodes each get their own uncertain-predecessor mitigation."""
        dag = _build_dag(
            nodes=[
                ("api1", "invoke_api"),
                ("del1", "delete_record"),
                ("api2", "send_webhook"),
                ("del2", "destroy_resource"),
            ],
            edges=[("api1", "del1"), ("api2", "del2")],
        )
        result = MitigateUncertainPredecessors()(dag, registry, empty_profile)
        assert len(result) == 2  # one per irreversible node
        target_sets = [set(m.target_node_ids) for m in result]
        assert {"api1"} in target_sets
        assert {"api2"} in target_sets

    def test_external_strategy_ignores_irreversible_and_stateful(
        self, registry: OperationRegistry, empty_profile: RiskProfile,
    ) -> None:
        """IRREVERSIBLE and STATEFUL ops should not trigger sandbox_external."""
        dag = _build_dag(
            nodes=[
                ("del", "delete_record"),
                ("state", "mutate_state"),
            ],
            edges=[],
        )
        result = MitigateExternalOps()(dag, registry, empty_profile)
        assert result == []
