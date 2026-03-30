"""Tests for YAML/JSON schema loading (NOD-13).

AC:
- [x] All 3 example workflow files load successfully
- [x] Missing node id raises descriptive ValidationError
- [x] Invalid edge_type value raises error
- [x] Unsupported file extension raises ValueError
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from workflow_eval.dag.schema import load_workflow
from workflow_eval.types import EdgeType, WorkflowDAG

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "sample_workflows"


# ---------------------------------------------------------------------------
# AC: All 3 example workflow files load successfully
# ---------------------------------------------------------------------------


class TestExampleFiles:
    def test_safe_read_pipeline(self) -> None:
        dag = load_workflow(EXAMPLES_DIR / "safe_read_pipeline.yaml")
        assert isinstance(dag, WorkflowDAG)
        assert dag.name == "safe-read-pipeline"
        assert len(dag.nodes) == 3
        assert len(dag.edges) == 2

    def test_risky_delete_cascade(self) -> None:
        dag = load_workflow(EXAMPLES_DIR / "risky_delete_cascade.yaml")
        assert isinstance(dag, WorkflowDAG)
        assert dag.name == "risky-delete-cascade"
        assert len(dag.nodes) == 7
        assert len(dag.edges) == 8

    def test_moderate_api_chain_json(self) -> None:
        dag = load_workflow(EXAMPLES_DIR / "moderate_api_chain.json")
        assert isinstance(dag, WorkflowDAG)
        assert dag.name == "moderate-api-chain"
        assert len(dag.nodes) == 4
        assert len(dag.edges) == 3

    def test_edge_types_preserved(self) -> None:
        dag = load_workflow(EXAMPLES_DIR / "safe_read_pipeline.yaml")
        for edge in dag.edges:
            assert edge.edge_type == EdgeType.DATA_FLOW

    def test_params_preserved(self) -> None:
        dag = load_workflow(EXAMPLES_DIR / "risky_delete_cascade.yaml")
        nodes = {n.id: n for n in dag.nodes}
        assert nodes["call_api"].params == {"endpoint": "/users"}
        assert nodes["del_records"].params == {"table": "users"}

    def test_metadata_preserved(self) -> None:
        dag = load_workflow(EXAMPLES_DIR / "safe_read_pipeline.yaml")
        assert dag.metadata == {
            "description": "Pure read operations only. Expected risk: low.",
            "version": 1,
        }


# ---------------------------------------------------------------------------
# AC: Missing node id raises descriptive ValidationError
# ---------------------------------------------------------------------------


class TestMissingFields:
    def test_missing_node_id(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(
            "name: bad\n"
            "nodes:\n"
            "  - operation: read_file\n"
            "edges: []\n"
        )
        with pytest.raises(ValidationError, match="id"):
            load_workflow(f)

    def test_missing_node_operation(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(
            "name: bad\n"
            "nodes:\n"
            "  - id: a\n"
            "edges: []\n"
        )
        with pytest.raises(ValidationError, match="operation"):
            load_workflow(f)

    def test_missing_name(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text('{"nodes": [], "edges": []}')
        with pytest.raises(ValidationError, match="name"):
            load_workflow(f)


# ---------------------------------------------------------------------------
# AC: Invalid edge_type value raises error
# ---------------------------------------------------------------------------


class TestInvalidEdgeType:
    def test_invalid_edge_type(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(
            "name: bad\n"
            "nodes:\n"
            "  - id: a\n"
            "    operation: read_file\n"
            "  - id: b\n"
            "    operation: write_file\n"
            "edges:\n"
            "  - source_id: a\n"
            "    target_id: b\n"
            "    edge_type: bogus\n"
        )
        with pytest.raises(ValidationError, match="edge_type"):
            load_workflow(f)


# ---------------------------------------------------------------------------
# AC: Unsupported file extension raises ValueError
# ---------------------------------------------------------------------------


class TestUnsupportedExtension:
    def test_txt_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "workflow.txt"
        f.write_text("name: x")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_workflow(f)

    def test_xml_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "workflow.xml"
        f.write_text("<dag/>")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_workflow(f)

    def test_no_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "workflow"
        f.write_text("name: x")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_workflow(f)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_yml_extension_works(self, tmp_path: Path) -> None:
        f = tmp_path / "w.yml"
        f.write_text(
            "name: test\n"
            "nodes:\n"
            "  - id: a\n"
            "    operation: read_file\n"
            "edges: []\n"
        )
        dag = load_workflow(f)
        assert dag.name == "test"

    def test_empty_yaml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.yaml"
        f.write_text("")
        with pytest.raises(ValueError, match="mapping"):
            load_workflow(f)

    def test_yaml_list_at_top_level_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "list.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="mapping"):
            load_workflow(f)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_workflow("/nonexistent/path.yaml")

    def test_string_path_works(self) -> None:
        dag = load_workflow(str(EXAMPLES_DIR / "safe_read_pipeline.yaml"))
        assert isinstance(dag, WorkflowDAG)

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        """Syntactically invalid JSON should raise json.JSONDecodeError."""
        import json as _json

        f = tmp_path / "bad.json"
        f.write_text("{invalid json")
        with pytest.raises(_json.JSONDecodeError):
            load_workflow(f)

    def test_json_list_at_top_level_raises(self, tmp_path: Path) -> None:
        """Non-dict JSON should raise ValidationError via Pydantic."""
        f = tmp_path / "list.json"
        f.write_text('[1, 2, 3]')
        with pytest.raises(ValidationError):
            load_workflow(f)

    def test_duplicate_node_ids_via_load(self, tmp_path: Path) -> None:
        """Duplicate IDs through the load_workflow() entrypoint."""
        f = tmp_path / "dupes.yaml"
        f.write_text(
            "name: dupes\n"
            "nodes:\n"
            "  - id: a\n"
            "    operation: read_file\n"
            "  - id: a\n"
            "    operation: write_file\n"
            "edges: []\n"
        )
        with pytest.raises(ValidationError, match="Duplicate node IDs"):
            load_workflow(f)

    def test_extra_fields_rejected(self, tmp_path: Path) -> None:
        """WorkflowDAG has extra='forbid', so unknown top-level keys should fail."""
        f = tmp_path / "extra.yaml"
        f.write_text(
            "name: test\n"
            "nodes: []\n"
            "edges: []\n"
            "bogus_field: true\n"
        )
        with pytest.raises(ValidationError, match="bogus_field"):
            load_workflow(f)
