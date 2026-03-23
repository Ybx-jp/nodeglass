"""Tests for ontology module: enums, registry, defaults, YAML round-trip."""

import tempfile
from pathlib import Path

import pytest
import yaml

from workflow_eval.ontology.effect_types import EffectTarget, EffectType
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.ontology.defaults import DEFAULT_OPERATIONS, load_defaults
from workflow_eval.types import OperationDefinition


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

    def test_comparison(self) -> None:
        assert EffectType("pure") == EffectType.PURE
        assert EffectType.STATEFUL != EffectType.EXTERNAL

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

    def test_comparison(self) -> None:
        assert EffectTarget("network") == EffectTarget.NETWORK


# ---------------------------------------------------------------------------
# NOD-6: Core Pydantic models (OperationDefinition)
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
        assert op.base_risk_weight == 0.5

    def test_frozen(self) -> None:
        op = OperationDefinition(
            name="test_op",
            category="test",
            base_risk_weight=0.5,
            effect_type=EffectType.PURE,
            effect_targets=frozenset(),
        )
        with pytest.raises(Exception):
            op.name = "changed"  # type: ignore[misc]

    def test_rejects_invalid_risk_weight(self) -> None:
        with pytest.raises(Exception):
            OperationDefinition(
                name="bad",
                category="test",
                base_risk_weight=1.5,
                effect_type=EffectType.PURE,
                effect_targets=frozenset(),
            )
        with pytest.raises(Exception):
            OperationDefinition(
                name="bad",
                category="test",
                base_risk_weight=-0.1,
                effect_type=EffectType.PURE,
                effect_targets=frozenset(),
            )

    def test_json_round_trip(self) -> None:
        op = OperationDefinition(
            name="round_trip",
            category="test",
            base_risk_weight=0.42,
            effect_type=EffectType.EXTERNAL,
            effect_targets=frozenset({EffectTarget.NETWORK, EffectTarget.MEMORY}),
        )
        json_str = op.model_dump_json()
        restored = OperationDefinition.model_validate_json(json_str)
        assert restored.name == op.name
        assert restored.base_risk_weight == op.base_risk_weight
        assert restored.effect_type == op.effect_type
        assert restored.effect_targets == op.effect_targets


# ---------------------------------------------------------------------------
# NOD-8: OperationRegistry
# ---------------------------------------------------------------------------


class TestOperationRegistry:
    def _make_op(self, name: str = "test_op") -> OperationDefinition:
        return OperationDefinition(
            name=name,
            category="test",
            base_risk_weight=0.5,
            effect_type=EffectType.PURE,
            effect_targets=frozenset(),
        )

    def test_register_and_get(self) -> None:
        reg = OperationRegistry()
        op = self._make_op()
        reg.register(op)
        assert reg.get("test_op") is op

    def test_all(self) -> None:
        reg = OperationRegistry()
        reg.register(self._make_op("a"))
        reg.register(self._make_op("b"))
        assert len(reg.all()) == 2

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


# ---------------------------------------------------------------------------
# NOD-8: from_yaml
# ---------------------------------------------------------------------------


class TestRegistryFromYaml:
    def test_load_from_yaml(self, tmp_path: Path) -> None:
        data = {
            "operations": [
                {
                    "name": "yaml_op_1",
                    "category": "test",
                    "base_risk_weight": 0.1,
                    "effect_type": "pure",
                    "effect_targets": ["filesystem"],
                },
                {
                    "name": "yaml_op_2",
                    "category": "network",
                    "base_risk_weight": 0.7,
                    "effect_type": "external",
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
        assert EffectTarget.NETWORK in op2.effect_targets

    def test_round_trip_yaml(self, tmp_path: Path) -> None:
        """Write ops to YAML, reload, verify identical."""
        ops = [
            {
                "name": "rt_op",
                "category": "test",
                "base_risk_weight": 0.33,
                "effect_type": "stateful",
                "effect_targets": ["database"],
            }
        ]
        yaml_path = tmp_path / "rt.yaml"
        yaml_path.write_text(yaml.dump({"operations": ops}))

        reg = OperationRegistry.from_yaml(yaml_path)
        op = reg.get("rt_op")
        assert op.effect_type == EffectType.STATEFUL
        assert op.effect_targets == frozenset({EffectTarget.DATABASE})


# ---------------------------------------------------------------------------
# NOD-9: Default ontology completeness
# ---------------------------------------------------------------------------


class TestDefaultOntology:
    EXPECTED_NAMES = {
        "read_file", "write_file", "delete_file",
        "read_database", "write_database", "delete_record",
        "invoke_api", "send_webhook",
        "mutate_state", "read_state",
        "branch", "loop",
        "execute_code",
        "authenticate", "authorize", "read_credentials",
        "send_email", "send_notification",
        "create_resource", "destroy_resource",
    }

    def test_default_count(self) -> None:
        assert len(DEFAULT_OPERATIONS) == 20

    def test_all_expected_names_present(self) -> None:
        actual_names = {op.name for op in DEFAULT_OPERATIONS}
        assert actual_names == self.EXPECTED_NAMES

    def test_load_defaults_populates_registry(self) -> None:
        reg = load_defaults()
        assert len(reg) == 20
        for name in self.EXPECTED_NAMES:
            op = reg.get(name)
            assert 0.0 <= op.base_risk_weight <= 1.0

    def test_load_defaults_into_existing_registry(self) -> None:
        reg = OperationRegistry()
        custom = OperationDefinition(
            name="custom_op",
            category="custom",
            base_risk_weight=0.99,
            effect_type=EffectType.IRREVERSIBLE,
            effect_targets=frozenset({EffectTarget.SYSTEM_CONFIG}),
        )
        reg.register(custom)
        load_defaults(reg)
        assert len(reg) == 21
        assert "custom_op" in reg

    def test_risk_weights_match_spec(self) -> None:
        reg = load_defaults()
        assert reg.get("read_file").base_risk_weight == 0.05
        assert reg.get("delete_file").base_risk_weight == 0.80
        assert reg.get("delete_record").base_risk_weight == 0.85
        assert reg.get("invoke_api").base_risk_weight == 0.30
        assert reg.get("execute_code").base_risk_weight == 0.60
        assert reg.get("read_credentials").base_risk_weight == 0.45
        assert reg.get("send_email").base_risk_weight == 0.50
        assert reg.get("destroy_resource").base_risk_weight == 0.90

    def test_effect_types_match_spec(self) -> None:
        reg = load_defaults()
        assert reg.get("read_file").effect_type == EffectType.PURE
        assert reg.get("write_file").effect_type == EffectType.STATEFUL
        assert reg.get("delete_file").effect_type == EffectType.IRREVERSIBLE
        assert reg.get("invoke_api").effect_type == EffectType.EXTERNAL
        assert reg.get("send_email").effect_type == EffectType.IRREVERSIBLE
        assert reg.get("destroy_resource").effect_type == EffectType.IRREVERSIBLE

    def test_effect_targets_match_spec(self) -> None:
        reg = load_defaults()
        assert reg.get("read_file").effect_targets == frozenset({EffectTarget.FILESYSTEM})
        assert reg.get("execute_code").effect_targets == frozenset(
            {EffectTarget.MEMORY, EffectTarget.FILESYSTEM, EffectTarget.NETWORK}
        )
        assert reg.get("send_email").effect_targets == frozenset(
            {EffectTarget.NETWORK, EffectTarget.USER_FACING}
        )
        assert reg.get("destroy_resource").effect_targets == frozenset(
            {EffectTarget.NETWORK, EffectTarget.SYSTEM_CONFIG}
        )
