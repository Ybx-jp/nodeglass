"""Tests for DAG inspector script (NOD-45).

AC:
- [x] --list shows all /tmp/wfeval-*.json files with summary stats
- [x] Inspect mode prints node table, edge list, operation histogram, and full risk profile
- [x] --mermaid outputs valid Mermaid flowchart syntax
- [x] Script handles missing/corrupt state files gracefully
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Import the inspector functions directly
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from inspect_session import (
    _load_state,
    _mermaid_node_label,
    _mermaid_node_shape,
    _print_mermaid,
    _print_node_table,
    _print_edge_list,
    _print_histogram,
    _print_risk_profile,
    _resolve_session,
    cmd_list,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_state(
    tmp_path: Path,
    session_id: str = "test-session",
    nodes: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
) -> Path:
    """Create a state file in tmp_path and return its path."""
    if nodes is None:
        nodes = [
            {"id": "step_0", "operation": "read_file", "params": {"tool": "Read", "file_path": "/a.py"}},
            {"id": "step_1", "operation": "write_file", "params": {"tool": "Edit", "file_path": "/a.py"}},
            {"id": "step_2", "operation": "delete_file", "params": {"tool": "Bash", "command": "rm -rf build/"}},
        ]
    if edges is None:
        edges = [
            {"source_id": "step_0", "target_id": "step_1", "edge_type": "control_flow"},
            {"source_id": "step_1", "target_id": "step_2", "edge_type": "control_flow"},
        ]
    state = {
        "session_id": session_id,
        "step_counter": len(nodes),
        "nodes": nodes,
        "edges": edges,
    }
    path = tmp_path / f"wfeval-{session_id}.json"
    path.write_text(json.dumps(state))
    return path


# ---------------------------------------------------------------------------
# State loading
# ---------------------------------------------------------------------------

class TestLoadState:
    def test_valid_state(self, tmp_path: Path) -> None:
        path = _make_state(tmp_path)
        state = _load_state(path)
        assert state is not None
        assert len(state["nodes"]) == 3
        assert len(state["edges"]) == 2

    def test_corrupt_json(self, tmp_path: Path) -> None:
        path = tmp_path / "wfeval-corrupt.json"
        path.write_text("not json at all{{{")
        assert _load_state(path) is None

    def test_missing_nodes_key(self, tmp_path: Path) -> None:
        path = tmp_path / "wfeval-nokeys.json"
        path.write_text(json.dumps({"session_id": "x"}))
        assert _load_state(path) is None

    def test_empty_but_valid(self, tmp_path: Path) -> None:
        path = _make_state(tmp_path, session_id="empty", nodes=[], edges=[])
        state = _load_state(path)
        assert state is not None
        assert state["nodes"] == []


# ---------------------------------------------------------------------------
# List mode
# ---------------------------------------------------------------------------

class TestListMode:
    def test_list_shows_files(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        _make_state(tmp_path, "sess-a")
        _make_state(tmp_path, "sess-b", nodes=[
            {"id": "step_0", "operation": "invoke_api", "params": {"tool": "WebFetch"}},
        ], edges=[])

        # Patch the state dir/glob to use tmp_path
        import inspect_session
        monkeypatch.setattr(inspect_session, "_STATE_DIR", tmp_path)

        cmd_list()
        output = capsys.readouterr().out
        assert "sess-a" in output
        assert "sess-b" in output
        assert "3" in output  # sess-a has 3 nodes
        assert "1" in output  # sess-b has 1 node

    def test_list_empty(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        import inspect_session
        monkeypatch.setattr(inspect_session, "_STATE_DIR", tmp_path)

        cmd_list()
        output = capsys.readouterr().out
        assert "No session DAGs" in output

    def test_list_shows_corrupt(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "wfeval-bad.json").write_text("garbage")

        import inspect_session
        monkeypatch.setattr(inspect_session, "_STATE_DIR", tmp_path)

        cmd_list()
        output = capsys.readouterr().out
        assert "corrupt" in output


# ---------------------------------------------------------------------------
# Inspect mode — node table, edge list, histogram
# ---------------------------------------------------------------------------

class TestInspectOutput:
    def test_node_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        nodes = [
            {"id": "step_0", "operation": "read_file", "params": {"tool": "Read", "file_path": "/foo"}},
            {"id": "step_1", "operation": "write_file", "params": {"tool": "Edit", "file_path": "/bar"}},
        ]
        _print_node_table(nodes)
        output = capsys.readouterr().out
        assert "step_0" in output
        assert "read_file" in output
        assert "Read" in output
        assert "step_1" in output
        assert "write_file" in output

    def test_edge_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        edges = [
            {"source_id": "step_0", "target_id": "step_1", "edge_type": "control_flow"},
        ]
        _print_edge_list(edges)
        output = capsys.readouterr().out
        assert "step_0 -> step_1" in output
        assert "control_flow" in output

    def test_edge_list_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_edge_list([])
        output = capsys.readouterr().out
        assert "(none)" in output

    def test_histogram(self, capsys: pytest.CaptureFixture[str]) -> None:
        nodes = [
            {"id": "s0", "operation": "read_file", "params": {}},
            {"id": "s1", "operation": "read_file", "params": {}},
            {"id": "s2", "operation": "write_file", "params": {}},
            {"id": "s3", "operation": "delete_file", "params": {}},
        ]
        _print_histogram(nodes)
        output = capsys.readouterr().out
        assert "read_file" in output
        assert "2" in output
        assert "##" in output

    def test_risk_profile_scores(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = {
            "session_id": "test",
            "nodes": [
                {"id": "step_0", "operation": "read_file", "params": {}},
                {"id": "step_1", "operation": "delete_file", "params": {}},
            ],
            "edges": [
                {"source_id": "step_0", "target_id": "step_1", "edge_type": "control_flow"},
            ],
        }
        _print_risk_profile(state)
        output = capsys.readouterr().out
        assert "Aggregate" in output
        assert "fan_out" in output
        assert "irreversibility" in output

    def test_risk_profile_empty_dag(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = {"session_id": "empty", "nodes": [], "edges": []}
        _print_risk_profile(state)
        output = capsys.readouterr().out
        assert "empty DAG" in output


# ---------------------------------------------------------------------------
# Mermaid export
# ---------------------------------------------------------------------------

class TestMermaid:
    def test_mermaid_header(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_mermaid("test-sess", [], [])
        output = capsys.readouterr().out
        assert output.startswith("flowchart TD")
        assert "test-sess" in output

    def test_mermaid_nodes(self, capsys: pytest.CaptureFixture[str]) -> None:
        nodes = [
            {"id": "step_0", "operation": "read_file", "params": {"tool": "Read"}},
            {"id": "step_1", "operation": "delete_file", "params": {"tool": "Bash"}},
        ]
        _print_mermaid("s", nodes, [])
        output = capsys.readouterr().out
        # read_file → rounded
        assert 'step_0("step_0' in output
        # delete_file → hexagon
        assert 'step_1{{"step_1' in output

    def test_mermaid_edges(self, capsys: pytest.CaptureFixture[str]) -> None:
        edges = [
            {"source_id": "step_0", "target_id": "step_1", "edge_type": "control_flow"},
            {"source_id": "step_1", "target_id": "step_2", "edge_type": "data_flow"},
        ]
        _print_mermaid("s", [], edges)
        output = capsys.readouterr().out
        assert "step_0 --> step_1" in output
        assert "step_1 -.-> step_2" in output

    def test_mermaid_conditional_edge(self, capsys: pytest.CaptureFixture[str]) -> None:
        edges = [{"source_id": "a", "target_id": "b", "edge_type": "conditional", "condition": "if ok"}]
        _print_mermaid("s", [], edges)
        output = capsys.readouterr().out
        assert "a -->|if ok| b" in output

    def test_node_label(self) -> None:
        node = {"id": "step_0", "operation": "read_file", "params": {"tool": "Read"}}
        label = _mermaid_node_label(node)
        assert "step_0" in label
        assert "read_file" in label
        assert "Read" in label


class TestMermaidShapes:
    def test_pure_read_rounded(self) -> None:
        node = {"id": "s", "operation": "read_file", "params": {}}
        assert _mermaid_node_shape(node) == ("(", ")")

    def test_stateful_rectangle(self) -> None:
        node = {"id": "s", "operation": "write_file", "params": {}}
        assert _mermaid_node_shape(node) == ("[", "]")

    def test_external_subroutine(self) -> None:
        node = {"id": "s", "operation": "invoke_api", "params": {}}
        assert _mermaid_node_shape(node) == ("[[", "]]")

    def test_irreversible_hexagon(self) -> None:
        node = {"id": "s", "operation": "delete_file", "params": {}}
        assert _mermaid_node_shape(node) == ("{{", "}}")

    def test_destroy_resource_hexagon(self) -> None:
        node = {"id": "s", "operation": "destroy_resource", "params": {}}
        assert _mermaid_node_shape(node) == ("{{", "}}")

    def test_execute_code_subroutine(self) -> None:
        node = {"id": "s", "operation": "execute_code", "params": {}}
        assert _mermaid_node_shape(node) == ("[[", "]]")

    def test_unknown_op_rounded(self) -> None:
        node = {"id": "s", "operation": "something_new", "params": {}}
        assert _mermaid_node_shape(node) == ("(", ")")
