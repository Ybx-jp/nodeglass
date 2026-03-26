"""Tests for ontology module: enums, registry, defaults, YAML round-trip.

Also covers all Pydantic models in types.py per NOD-6 AC.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from workflow_eval.ontology.effect_types import EffectTarget, EffectType
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.ontology.defaults import DEFAULT_OPERATIONS, get_default_registry, inject_defaults
from workflow_eval.types import (
    DAGEdge, DAGNode, EdgeType, ExecutionOutcome, ExecutionRecord,
    Mitigation, MitigationAction, MitigationPlan, MitigationPriority,
    OperationDefinition, RiskLevel, RiskProfile, ScoringConfig,
    SubScore, WorkflowDAG, WorkflowExecution,
)


# ---------------------------------------------------------------------------
# NOD-7: EffectType and EffectTarget enums
# ---------------------------------------------------------------------------


class TestEffectType:
    def test_values(self) -> None:
        assert EffectType.PURE == "pure"
        assert EffectType.STATEFUL == "stateful"
        assert EffectType.EXTERNAL == "external"
        assert EffectType.IRREVERSIBLE == "irreversible"

    def test_serializes_to_string(self) -> None:
        assert str(EffectType.PURE) == "pure"
        assert EffectType.PURE.value == "pure"

    def test_construct_from_string(self) -> None:
        assert EffectType("pure") == EffectType.PURE

    def test_importable_from_ontology(self) -> None:
        from workflow_eval.ontology import EffectType as ET
        assert ET.PURE == EffectType.PURE


class TestEffectTarget:
    def test_all_targets(self) -> None:
        expected = {
            "filesystem", "network", "database", "memory",
            "user_facing", "credentials", "system_config",
        }
        actual = {t.value for t in EffectTarget}
        assert actual == expected

    def test_serializes_to_string(self) -> None:
        assert EffectTarget.FILESYSTEM.value == "filesystem"

    def test_construct_from_string(self) -> None:
        assert EffectTarget("network") == EffectTarget.NETWORK


# ---------------------------------------------------------------------------
# NOD-6: OperationDefinition
# ---------------------------------------------------------------------------


class TestOperationDefinition:
    def test_create_valid(self) -> None:
        op = OperationDefinition(
            name="test_op",
            category="test",
            base_risk_weight=0.5,
            effect_type=EffectType.STATEFUL,
            effect_targets=frozenset({EffectTarget.DATABASE}),
        )
        assert op.name == "test_op"
        assert op.category == "test"
        assert op.base_risk_weight == 0.5
        assert op.effect_type == EffectType.STATEFUL
        assert op.effect_targets == frozenset({EffectTarget.DATABASE})

    def test_frozen(self) -> None:
        op = OperationDefinition(
            name="test_op",
            category="test",
            base_risk_weight=0.5,
            effect_type=EffectType.PURE,
            effect_targets=frozenset(),
        )
        with pytest.raises(ValidationError):
            op.name = "changed"  # type: ignore[misc]

    def test_rejects_invalid_risk_weight(self) -> None:
        with pytest.raises(ValidationError):
            OperationDefinition(
                name="bad", category="test", base_risk_weight=1.5,
                effect_type=EffectType.PURE, effect_targets=frozenset(),
            )
        with pytest.raises(ValidationError):
            OperationDefinition(
                name="bad", category="test", base_risk_weight=-0.1,
                effect_type=EffectType.PURE, effect_targets=frozenset(),
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            OperationDefinition(
                name="bad", category="test", base_risk_weight=0.5,
                effect_type=EffectType.PURE, effect_targets=frozenset(),
                bogus_field="should fail",  # type: ignore[call-arg]
            )

    def test_preconditions_postconditions(self) -> None:
        op = OperationDefinition(
            name="guarded_op", category="test", base_risk_weight=0.5,
            effect_type=EffectType.EXTERNAL,
            effect_targets=frozenset({EffectTarget.NETWORK}),
            preconditions=("authenticated", "rate_limit_ok"),
            postconditions=("audit_logged",),
        )
        assert op.preconditions == ("authenticated", "rate_limit_ok")
        assert op.postconditions == ("audit_logged",)

    def test_preconditions_default_empty(self) -> None:
        op = OperationDefinition(
            name="bare_op", category="test", base_risk_weight=0.1,
            effect_type=EffectType.PURE, effect_targets=frozenset(),
        )
        assert op.preconditions == ()
        assert op.postconditions == ()

    def test_json_round_trip(self) -> None:
        op = OperationDefinition(
            name="round_trip", category="test", base_risk_weight=0.42,
            effect_type=EffectType.EXTERNAL,
            effect_targets=frozenset({EffectTarget.NETWORK, EffectTarget.MEMORY}),
            preconditions=("authed",), postconditions=("logged",),
        )
        restored = OperationDefinition.model_validate_json(op.model_dump_json())
        assert restored == op


# ---------------------------------------------------------------------------
# NOD-8: OperationRegistry
# ---------------------------------------------------------------------------


class TestOperationRegistry:
    def _make_op(self, name: str = "test_op") -> OperationDefinition:
        return OperationDefinition(
            name=name, category="test", base_risk_weight=0.5,
            effect_type=EffectType.PURE, effect_targets=frozenset(),
        )

    def test_register_and_get(self) -> None:
        reg = OperationRegistry()
        op = self._make_op()
        reg.register(op)
        assert reg.get("test_op") is op

    def test_all_returns_registered_ops(self) -> None:
        reg = OperationRegistry()
        op_a = self._make_op("a")
        op_b = self._make_op("b")
        reg.register(op_a)
        reg.register(op_b)
        result = reg.all()
        assert len(result) == 2
        assert {op.name for op in result} == {"a", "b"}

    def test_len_and_contains(self) -> None:
        reg = OperationRegistry()
        reg.register(self._make_op("x"))
        assert len(reg) == 1
        assert "x" in reg
        assert "y" not in reg

    def test_iter(self) -> None:
        reg = OperationRegistry()
        reg.register(self._make_op("a"))
        reg.register(self._make_op("b"))
        names = {op.name for op in reg}
        assert names == {"a", "b"}

    def test_duplicate_raises_value_error(self) -> None:
        reg = OperationRegistry()
        reg.register(self._make_op("dup"))
        with pytest.raises(ValueError, match="Duplicate operation"):
            reg.register(self._make_op("dup"))

    def test_missing_raises_key_error(self) -> None:
        reg = OperationRegistry()
        with pytest.raises(KeyError, match="Unknown operation"):
            reg.get("nonexistent")

    def test_reset(self) -> None:
        reg = OperationRegistry()
        reg.register(self._make_op("a"))
        reg.register(self._make_op("b"))
        assert len(reg) == 2
        reg.reset()
        assert len(reg) == 0
        assert "a" not in reg


# ---------------------------------------------------------------------------
# NOD-8: from_yaml (happy + error paths)
# ---------------------------------------------------------------------------


class TestRegistryFromYaml:
    def test_load_from_yaml(self, tmp_path: Path) -> None:
        data = {
            "operations": [
                {
                    "name": "yaml_op_1", "category": "test",
                    "base_risk_weight": 0.1, "effect_type": "pure",
                    "effect_targets": ["filesystem"],
                },
                {
                    "name": "yaml_op_2", "category": "network",
                    "base_risk_weight": 0.7, "effect_type": "external",
                    "effect_targets": ["network", "memory"],
                },
            ]
        }
        yaml_path = tmp_path / "ops.yaml"
        yaml_path.write_text(yaml.dump(data))

        reg = OperationRegistry.from_yaml(yaml_path)
        assert len(reg) == 2

        op1 = reg.get("yaml_op_1")
        assert op1.effect_type == EffectType.PURE
        assert op1.effect_targets == frozenset({EffectTarget.FILESYSTEM})

        op2 = reg.get("yaml_op_2")
        assert op2.base_risk_weight == 0.7
        assert op2.effect_targets == frozenset({EffectTarget.NETWORK, EffectTarget.MEMORY})

    def test_python_to_yaml_round_trip(self, tmp_path: Path) -> None:
        """Build registry in Python, export to YAML, reload, compare."""
        original = OperationDefinition(
            name="rt_op", category="test", base_risk_weight=0.33,
            effect_type=EffectType.STATEFUL,
            effect_targets=frozenset({EffectTarget.DATABASE}),
        )
        yaml_data = {
            "operations": [{
                "name": original.name,
                "category": original.category,
                "base_risk_weight": original.base_risk_weight,
                "effect_type": original.effect_type.value,
                "effect_targets": sorted(t.value for t in original.effect_targets),
            }]
        }
        yaml_path = tmp_path / "rt.yaml"
        yaml_path.write_text(yaml.dump(yaml_data))

        reg = OperationRegistry.from_yaml(yaml_path)
        restored = reg.get("rt_op")
        assert restored.name == original.name
        assert restored.category == original.category
        assert restored.base_risk_weight == original.base_risk_weight
        assert restored.effect_type == original.effect_type
        assert restored.effect_targets == original.effect_targets

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(yaml.dump({"operations": [{"name": "incomplete"}]}))
        with pytest.raises(Exception):
            OperationRegistry.from_yaml(yaml_path)

    def test_invalid_effect_type_raises(self, tmp_path: Path) -> None:
        data = {"operations": [{
            "name": "bad", "category": "test", "base_risk_weight": 0.1,
            "effect_type": "nonexistent_type", "effect_targets": [],
        }]}
        yaml_path = tmp_path / "bad_type.yaml"
        yaml_path.write_text(yaml.dump(data))
        with pytest.raises(ValueError):
            OperationRegistry.from_yaml(yaml_path)

    def test_empty_operations_list(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text(yaml.dump({"operations": []}))
        reg = OperationRegistry.from_yaml(yaml_path)
        assert len(reg) == 0

    def test_missing_operations_key_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "no_key.yaml"
        yaml_path.write_text(yaml.dump({"something_else": []}))
        with pytest.raises((KeyError, TypeError)):
            OperationRegistry.from_yaml(yaml_path)


# ---------------------------------------------------------------------------
# NOD-9: Default ontology — full spec pinning
# ---------------------------------------------------------------------------

# Complete spec table: every operation, every field.
# If any value drifts, the specific assertion fails with a clear message.
SPEC: dict[str, tuple[float, EffectType, frozenset[EffectTarget]]] = {
    "read_file":        (0.05, EffectType.PURE,         frozenset({EffectTarget.FILESYSTEM})),
    "write_file":       (0.35, EffectType.STATEFUL,     frozenset({EffectTarget.FILESYSTEM})),
    "delete_file":      (0.80, EffectType.IRREVERSIBLE, frozenset({EffectTarget.FILESYSTEM})),
    "read_database":    (0.05, EffectType.PURE,         frozenset({EffectTarget.DATABASE})),
    "write_database":   (0.40, EffectType.STATEFUL,     frozenset({EffectTarget.DATABASE})),
    "delete_record":    (0.85, EffectType.IRREVERSIBLE, frozenset({EffectTarget.DATABASE})),
    "invoke_api":       (0.30, EffectType.EXTERNAL,     frozenset({EffectTarget.NETWORK})),
    "send_webhook":     (0.35, EffectType.EXTERNAL,     frozenset({EffectTarget.NETWORK})),
    "mutate_state":     (0.25, EffectType.STATEFUL,     frozenset({EffectTarget.MEMORY})),
    "read_state":       (0.02, EffectType.PURE,         frozenset({EffectTarget.MEMORY})),
    "branch":           (0.10, EffectType.PURE,         frozenset()),
    "loop":             (0.15, EffectType.PURE,         frozenset()),
    "execute_code":     (0.60, EffectType.EXTERNAL,     frozenset({EffectTarget.MEMORY, EffectTarget.FILESYSTEM, EffectTarget.NETWORK})),
    "authenticate":     (0.20, EffectType.EXTERNAL,     frozenset({EffectTarget.CREDENTIALS, EffectTarget.NETWORK})),
    "authorize":        (0.15, EffectType.PURE,         frozenset({EffectTarget.CREDENTIALS})),
    "read_credentials": (0.45, EffectType.PURE,         frozenset({EffectTarget.CREDENTIALS})),
    "send_email":       (0.50, EffectType.IRREVERSIBLE, frozenset({EffectTarget.NETWORK, EffectTarget.USER_FACING})),
    "send_notification":(0.40, EffectType.EXTERNAL,     frozenset({EffectTarget.NETWORK, EffectTarget.USER_FACING})),
    "create_resource":  (0.55, EffectType.EXTERNAL,     frozenset({EffectTarget.NETWORK, EffectTarget.SYSTEM_CONFIG})),
    "destroy_resource": (0.90, EffectType.IRREVERSIBLE, frozenset({EffectTarget.NETWORK, EffectTarget.SYSTEM_CONFIG})),
}


class TestDefaultOntology:
    def test_default_count(self) -> None:
        assert len(DEFAULT_OPERATIONS) == 20

    def test_all_expected_names_present(self) -> None:
        actual_names = {op.name for op in DEFAULT_OPERATIONS}
        assert actual_names == set(SPEC.keys())

    def test_get_default_registry(self) -> None:
        reg = get_default_registry()
        assert len(reg) == 20

    def test_inject_defaults_into_existing_registry(self) -> None:
        reg = OperationRegistry()
        custom = OperationDefinition(
            name="custom_op", category="custom", base_risk_weight=0.99,
            effect_type=EffectType.IRREVERSIBLE,
            effect_targets=frozenset({EffectTarget.SYSTEM_CONFIG}),
        )
        reg.register(custom)
        inject_defaults(reg)
        assert len(reg) == 21
        assert "custom_op" in reg

    @pytest.mark.parametrize("op_name", sorted(SPEC.keys()))
    def test_risk_weight(self, op_name: str) -> None:
        reg = get_default_registry()
        expected_weight, _, _ = SPEC[op_name]
        assert reg.get(op_name).base_risk_weight == expected_weight

    @pytest.mark.parametrize("op_name", sorted(SPEC.keys()))
    def test_effect_type(self, op_name: str) -> None:
        reg = get_default_registry()
        _, expected_type, _ = SPEC[op_name]
        assert reg.get(op_name).effect_type == expected_type

    @pytest.mark.parametrize("op_name", sorted(SPEC.keys()))
    def test_effect_targets(self, op_name: str) -> None:
        reg = get_default_registry()
        _, _, expected_targets = SPEC[op_name]
        assert reg.get(op_name).effect_targets == expected_targets


# ---------------------------------------------------------------------------
# NOD-6: ScoringConfig validation
# ---------------------------------------------------------------------------


class TestScoringConfig:
    def test_default_weights_valid(self) -> None:
        cfg = ScoringConfig()
        total = (cfg.fan_out + cfg.chain_depth + cfg.irreversibility
                 + cfg.centrality + cfg.spectral + cfg.compositional)
        assert abs(total - 1.0) < 1e-9

    def test_rejects_weights_not_summing_to_one(self) -> None:
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            ScoringConfig(fan_out=0.5, chain_depth=0.5, irreversibility=0.5)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ScoringConfig(bogus=0.1)  # type: ignore[call-arg]

    def test_json_round_trip(self) -> None:
        cfg = ScoringConfig()
        assert ScoringConfig.model_validate_json(cfg.model_dump_json()) == cfg


# ---------------------------------------------------------------------------
# NOD-6: DAG model round-trips
# ---------------------------------------------------------------------------


class TestDAGModels:
    def test_dag_node_round_trip(self) -> None:
        node = DAGNode(id="a", operation="read_file", params={"path": "/tmp"})
        assert DAGNode.model_validate_json(node.model_dump_json()) == node

    def test_dag_edge_round_trip(self) -> None:
        edge = DAGEdge(source_id="a", target_id="b", edge_type=EdgeType.DATA_FLOW)
        assert DAGEdge.model_validate_json(edge.model_dump_json()) == edge

    def test_workflow_dag_round_trip(self) -> None:
        dag = WorkflowDAG(
            name="test",
            nodes=(DAGNode(id="a", operation="read_file"),
                   DAGNode(id="b", operation="write_file")),
            edges=(DAGEdge(source_id="a", target_id="b"),),
            metadata={"version": 1},
        )
        assert WorkflowDAG.model_validate_json(dag.model_dump_json()) == dag

    def test_sub_score_round_trip(self) -> None:
        ss = SubScore(name="fan_out", score=0.3, weight=0.15, details={"top": ["a"]})
        assert SubScore.model_validate_json(ss.model_dump_json()) == ss

    def test_risk_profile_round_trip(self) -> None:
        rp = RiskProfile(
            workflow_name="test", aggregate_score=0.42,
            risk_level=RiskLevel.MEDIUM,
            sub_scores=(SubScore(name="fan_out", score=0.3, weight=0.15),),
            node_count=2, edge_count=1,
        )
        assert RiskProfile.model_validate_json(rp.model_dump_json()) == rp


# ---------------------------------------------------------------------------
# NOD-6: Mitigation model round-trips
# ---------------------------------------------------------------------------


class TestMitigationModels:
    def test_mitigation_round_trip(self) -> None:
        m = Mitigation(
            action=MitigationAction.ADD_CONFIRMATION,
            priority=MitigationPriority.REQUIRED,
            target_node_ids=("b", "c"),
            reason="irreversible downstream",
        )
        assert Mitigation.model_validate_json(m.model_dump_json()) == m

    def test_mitigation_plan_round_trip(self) -> None:
        mp = MitigationPlan(
            mitigations=(Mitigation(
                action=MitigationAction.ADD_ROLLBACK,
                priority=MitigationPriority.RECOMMENDED,
                target_node_ids=("x",), reason="stateful op",
            ),),
            original_risk=0.7, residual_risk=0.35,
        )
        assert MitigationPlan.model_validate_json(mp.model_dump_json()) == mp


# ---------------------------------------------------------------------------
# NOD-6: Execution model round-trips
# ---------------------------------------------------------------------------


class TestExecutionModels:
    def test_execution_record_round_trip(self) -> None:
        er = ExecutionRecord(
            node_id="a", operation="read_file",
            outcome=ExecutionOutcome.SUCCESS, duration_ms=12.5,
        )
        assert ExecutionRecord.model_validate_json(er.model_dump_json()) == er

    def test_execution_record_failure(self) -> None:
        er = ExecutionRecord(
            node_id="b", operation="invoke_api",
            outcome=ExecutionOutcome.FAILURE, error="timeout",
        )
        assert er.error == "timeout"
        assert ExecutionRecord.model_validate_json(er.model_dump_json()) == er

    def test_workflow_execution_round_trip(self) -> None:
        dag = WorkflowDAG(
            name="test",
            nodes=(DAGNode(id="a", operation="read_file"),),
            edges=(),
        )
        we = WorkflowExecution(
            id="exec-1", workflow_name="test", dag=dag,
            records=(ExecutionRecord(
                node_id="a", operation="read_file",
                outcome=ExecutionOutcome.SUCCESS,
            ),),
            predicted_risk=0.42, actual_outcome=ExecutionOutcome.SUCCESS,
        )
        assert WorkflowExecution.model_validate_json(we.model_dump_json()) == we

    def test_workflow_execution_nullable_fields(self) -> None:
        dag = WorkflowDAG(name="bare", nodes=(), edges=())
        we = WorkflowExecution(
            id="exec-2", workflow_name="bare", dag=dag, records=(),
        )
        assert we.predicted_risk is None
        assert we.actual_outcome is None
        assert WorkflowExecution.model_validate_json(we.model_dump_json()) == we


# ---------------------------------------------------------------------------
# NOD-6: extra="forbid" on all models
# ---------------------------------------------------------------------------


class TestExtraForbid:
    def test_dag_node_rejects_extra(self) -> None:
        with pytest.raises(ValidationError):
            DAGNode(id="a", operation="read_file", extra_field="bad")  # type: ignore[call-arg]

    def test_dag_edge_rejects_extra(self) -> None:
        with pytest.raises(ValidationError):
            DAGEdge(source_id="a", target_id="b", extra_field="bad")  # type: ignore[call-arg]

    def test_workflow_dag_rejects_extra(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowDAG(name="test", nodes=(), edges=(), extra_field="bad")  # type: ignore[call-arg]

    def test_sub_score_rejects_extra(self) -> None:
        with pytest.raises(ValidationError):
            SubScore(name="x", score=0.5, weight=0.1, extra_field="bad")  # type: ignore[call-arg]

    def test_risk_profile_rejects_extra(self) -> None:
        with pytest.raises(ValidationError):
            RiskProfile(
                workflow_name="test", aggregate_score=0.5,
                risk_level=RiskLevel.MEDIUM, sub_scores=(),
                node_count=1, edge_count=0, extra_field="bad",  # type: ignore[call-arg]
            )

    def test_mitigation_rejects_extra(self) -> None:
        with pytest.raises(ValidationError):
            Mitigation(
                action=MitigationAction.ADD_CONFIRMATION,
                priority=MitigationPriority.REQUIRED,
                target_node_ids=(), reason="test", extra_field="bad",  # type: ignore[call-arg]
            )

    def test_mitigation_plan_rejects_extra(self) -> None:
        with pytest.raises(ValidationError):
            MitigationPlan(
                mitigations=(), original_risk=0.5,
                residual_risk=0.3, extra_field="bad",  # type: ignore[call-arg]
            )

    def test_execution_record_rejects_extra(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionRecord(
                node_id="a", operation="read_file",
                outcome=ExecutionOutcome.SUCCESS, extra_field="bad",  # type: ignore[call-arg]
            )

    def test_workflow_execution_rejects_extra(self) -> None:
        dag = WorkflowDAG(name="test", nodes=(), edges=())
        with pytest.raises(ValidationError):
            WorkflowExecution(
                id="x", workflow_name="test", dag=dag,
                records=(), extra_field="bad",  # type: ignore[call-arg]
            )
