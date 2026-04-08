"""Tests for default ontology YAML config (NOD-44).

AC:
- [x] OperationRegistry.from_yaml("config/default_ontology.yaml") produces
      identical registry to importing defaults.py
- [x] YAML is human-readable and editable
- [x] All 20 operations present with correct risk weights, effect types, and targets
"""

import pytest

from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry


@pytest.fixture()
def yaml_registry() -> OperationRegistry:
    return OperationRegistry.from_yaml("config/default_ontology.yaml")


@pytest.fixture()
def py_registry() -> OperationRegistry:
    return get_default_registry()


class TestYAMLMatchesPython:
    def test_same_count(
        self, yaml_registry: OperationRegistry, py_registry: OperationRegistry,
    ) -> None:
        assert len(yaml_registry) == len(py_registry) == 20

    def test_same_names(
        self, yaml_registry: OperationRegistry, py_registry: OperationRegistry,
    ) -> None:
        py_names = sorted(op.name for op in py_registry)
        yaml_names = sorted(op.name for op in yaml_registry)
        assert yaml_names == py_names

    def test_each_operation_identical(
        self, yaml_registry: OperationRegistry, py_registry: OperationRegistry,
    ) -> None:
        for py_op in py_registry.all():
            yaml_op = yaml_registry.get(py_op.name)
            assert yaml_op.category == py_op.category
            assert yaml_op.base_risk_weight == pytest.approx(py_op.base_risk_weight)
            assert yaml_op.effect_type == py_op.effect_type
            assert yaml_op.effect_targets == py_op.effect_targets


class TestAllOperationsPresent:
    EXPECTED_NAMES = [
        "read_file", "write_file", "delete_file",
        "read_database", "write_database", "delete_record",
        "invoke_api", "send_webhook",
        "mutate_state", "read_state",
        "branch", "loop",
        "execute_code",
        "authenticate", "authorize", "read_credentials",
        "send_email", "send_notification",
        "create_resource", "destroy_resource",
    ]

    def test_all_20_present(self, yaml_registry: OperationRegistry) -> None:
        for name in self.EXPECTED_NAMES:
            assert name in yaml_registry, f"Missing operation: {name}"

    def test_no_extras(self, yaml_registry: OperationRegistry) -> None:
        yaml_names = {op.name for op in yaml_registry}
        assert yaml_names == set(self.EXPECTED_NAMES)


class TestHumanReadable:
    def test_yaml_file_is_readable(self) -> None:
        from pathlib import Path
        content = Path("config/default_ontology.yaml").read_text()
        # Has comments
        assert content.startswith("#")
        # Has category comments
        assert "# --- I/O ---" in content
        assert "# --- Database ---" in content
        # Uses flow style for targets (compact, readable)
        assert "[filesystem]" in content
        assert "[database]" in content
