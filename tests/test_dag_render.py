"""Tests for shared DAG rendering module (NOD-51).

AC:
- [x] Compact trail view renders to stderr on every non-read permission prompt
- [x] Current operation highlighted with ◀ NOW marker
- [x] Top 3 sub-score bars rendered with Unicode block characters
- [x] Consecutive same-type operations collapsed (e.g., read_file ×6)
- [x] Full graph mode available via WFEVAL_VISUAL=full env var
- [x] ANSI colors with NO_COLOR / non-TTY fallback
- [x] Shared rendering module between this and NOD-45 inspector
- [x] Visual appears before the permission prompt, not after
"""

from __future__ import annotations

import io
import os
from typing import Any

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dag_render import (
    _Colors,
    _collapse_trail,
    mermaid_node_label,
    mermaid_node_shape,
    op_category,
    render_compact_trail,
    render_dag_visual,
    render_full_graph,
    render_mermaid,
    render_score_bars,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_state(n_reads: int = 3, extra_ops: list[str] | None = None) -> dict[str, Any]:
    """Build a sample state with n reads followed by extra ops."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    ops = ["read_file"] * n_reads + (extra_ops or [])
    tools = ["Read"] * n_reads + ["Edit"] * len(extra_ops or [])

    for i, (op, tool) in enumerate(zip(ops, tools)):
        nodes.append({"id": f"step_{i}", "operation": op, "params": {"tool": tool}})
        if i > 0:
            edges.append({"source_id": f"step_{i-1}", "target_id": f"step_{i}", "edge_type": "control_flow"})

    return {"session_id": "test-sess", "step_counter": len(nodes), "nodes": nodes, "edges": edges}


def _sample_result(aggregate: float = 0.35, level: str = "medium") -> dict[str, Any]:
    return {
        "aggregate": aggregate,
        "risk_level": level,
        "node_count": 5,
        "breakdown": "chain_depth: 0.50, spectral: 0.40",
        "sub_scores": [
            {"name": "fan_out", "score": 0.10, "weight": 0.15},
            {"name": "chain_depth", "score": 0.50, "weight": 0.20},
            {"name": "irreversibility", "score": 0.80, "weight": 0.25},
            {"name": "centrality", "score": 0.05, "weight": 0.15},
            {"name": "spectral", "score": 0.40, "weight": 0.10},
            {"name": "compositional", "score": 0.20, "weight": 0.15},
        ],
    }


# ---------------------------------------------------------------------------
# op_category
# ---------------------------------------------------------------------------

class TestOpCategory:
    def test_pure(self) -> None:
        assert op_category("read_file") == "pure"
        assert op_category("branch") == "pure"

    def test_stateful(self) -> None:
        assert op_category("write_file") == "stateful"
        assert op_category("mutate_state") == "stateful"

    def test_external(self) -> None:
        assert op_category("invoke_api") == "external"
        assert op_category("execute_code") == "external"

    def test_irreversible(self) -> None:
        assert op_category("delete_file") == "irreversible"
        assert op_category("destroy_resource") == "irreversible"


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------

class TestColors:
    def test_tty_has_colors(self) -> None:
        # Create a fake TTY-like stream
        class FakeTTY(io.StringIO):
            def isatty(self) -> bool:
                return True

        c = _Colors(FakeTTY())
        assert c.red != ""
        assert c.reset != ""

    def test_non_tty_no_colors(self) -> None:
        buf = io.StringIO()
        c = _Colors(buf)
        assert c.red == ""
        assert c.reset == ""

    def test_no_color_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeTTY(io.StringIO):
            def isatty(self) -> bool:
                return True

        monkeypatch.setenv("NO_COLOR", "1")
        c = _Colors(FakeTTY())
        assert c.red == ""

    def test_for_op_colors(self) -> None:
        class FakeTTY(io.StringIO):
            def isatty(self) -> bool:
                return True

        c = _Colors(FakeTTY())
        assert c.for_op("read_file") == c.green
        assert c.for_op("write_file") == c.yellow
        assert c.for_op("delete_file") == c.red
        assert c.for_op("invoke_api") == c.blue


# ---------------------------------------------------------------------------
# Collapse trail
# ---------------------------------------------------------------------------

class TestCollapseTrail:
    def test_consecutive_same_ops(self) -> None:
        nodes = [
            {"id": "s0", "operation": "read_file"},
            {"id": "s1", "operation": "read_file"},
            {"id": "s2", "operation": "read_file"},
        ]
        groups = _collapse_trail(nodes)
        assert groups == [("read_file", 3)]

    def test_mixed_ops(self) -> None:
        nodes = [
            {"id": "s0", "operation": "read_file"},
            {"id": "s1", "operation": "read_file"},
            {"id": "s2", "operation": "write_file"},
            {"id": "s3", "operation": "delete_file"},
        ]
        groups = _collapse_trail(nodes)
        assert groups == [("read_file", 2), ("write_file", 1), ("delete_file", 1)]

    def test_empty(self) -> None:
        assert _collapse_trail([]) == []

    def test_single(self) -> None:
        nodes = [{"id": "s0", "operation": "execute_code"}]
        assert _collapse_trail(nodes) == [("execute_code", 1)]


# ---------------------------------------------------------------------------
# Score bars
# ---------------------------------------------------------------------------

class TestScoreBars:
    def test_renders_top_3(self) -> None:
        buf = io.StringIO()
        result = _sample_result()
        render_score_bars(result["sub_scores"], file=buf)
        output = buf.getvalue()
        lines = [l for l in output.strip().split("\n") if l.strip()]
        assert len(lines) == 3

    def test_contains_block_chars(self) -> None:
        buf = io.StringIO()
        render_score_bars([{"name": "test", "score": 0.50, "weight": 0.20}], file=buf)
        output = buf.getvalue()
        assert "█" in output
        assert "░" in output

    def test_score_value_shown(self) -> None:
        buf = io.StringIO()
        render_score_bars([{"name": "chain_depth", "score": 0.75, "weight": 0.20}], file=buf)
        output = buf.getvalue()
        assert "0.75" in output
        assert "chain_depth" in output


# ---------------------------------------------------------------------------
# Compact trail
# ---------------------------------------------------------------------------

class TestCompactTrail:
    def test_now_marker(self) -> None:
        buf = io.StringIO()
        state = _sample_state(3, ["write_file"])
        render_compact_trail(state, "step_3", _sample_result(), file=buf)
        output = buf.getvalue()
        assert "◀ NOW" in output

    def test_collapse_shown(self) -> None:
        buf = io.StringIO()
        state = _sample_state(6, ["write_file"])
        render_compact_trail(state, "step_6", _sample_result(), file=buf)
        output = buf.getvalue()
        assert "read_file ×6" in output

    def test_header_shows_score(self) -> None:
        buf = io.StringIO()
        state = _sample_state(1, ["write_file"])
        render_compact_trail(state, "step_1", _sample_result(0.42, "medium"), file=buf)
        output = buf.getvalue()
        assert "0.42" in output
        assert "MEDIUM" in output

    def test_workflow_eval_label(self) -> None:
        buf = io.StringIO()
        state = _sample_state(1, ["write_file"])
        render_compact_trail(state, "step_1", _sample_result(), file=buf)
        output = buf.getvalue()
        assert "[WORKFLOW-EVAL]" in output

    def test_score_bars_included(self) -> None:
        buf = io.StringIO()
        state = _sample_state(1, ["write_file"])
        render_compact_trail(state, "step_1", _sample_result(), file=buf)
        output = buf.getvalue()
        assert "█" in output

    def test_empty_state(self) -> None:
        buf = io.StringIO()
        state = {"session_id": "x", "nodes": [], "edges": []}
        render_compact_trail(state, "step_0", _sample_result(), file=buf)
        assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Full graph
# ---------------------------------------------------------------------------

class TestFullGraph:
    def test_now_marker_double_border(self) -> None:
        buf = io.StringIO()
        state = _sample_state(1, ["write_file"])
        render_full_graph(state, "step_1", _sample_result(), file=buf)
        output = buf.getvalue()
        assert "◀ NOW" in output
        assert "═" in output  # double border
        assert "╔" in output

    def test_normal_nodes_single_border(self) -> None:
        buf = io.StringIO()
        state = _sample_state(2, ["write_file"])
        render_full_graph(state, "step_2", _sample_result(), file=buf)
        output = buf.getvalue()
        assert "┌" in output  # single border for non-current
        assert "─" in output

    def test_arrows_between_nodes(self) -> None:
        buf = io.StringIO()
        state = _sample_state(2, ["write_file"])
        render_full_graph(state, "step_2", _sample_result(), file=buf)
        output = buf.getvalue()
        assert "▼" in output

    def test_header_shows_score(self) -> None:
        buf = io.StringIO()
        state = _sample_state(1, ["write_file"])
        render_full_graph(state, "step_1", _sample_result(0.72, "high"), file=buf)
        output = buf.getvalue()
        assert "0.72" in output
        assert "HIGH" in output


# ---------------------------------------------------------------------------
# Mermaid (shared functions)
# ---------------------------------------------------------------------------

class TestMermaidShared:
    def test_node_label(self) -> None:
        node = {"id": "step_0", "operation": "read_file", "params": {"tool": "Read"}}
        label = mermaid_node_label(node)
        assert "step_0" in label
        assert "read_file" in label
        assert "Read" in label

    def test_node_shape_pure(self) -> None:
        assert mermaid_node_shape({"id": "s", "operation": "read_file", "params": {}}) == ("(", ")")

    def test_node_shape_stateful(self) -> None:
        assert mermaid_node_shape({"id": "s", "operation": "write_file", "params": {}}) == ("[", "]")

    def test_node_shape_external(self) -> None:
        assert mermaid_node_shape({"id": "s", "operation": "invoke_api", "params": {}}) == ("[[", "]]")

    def test_node_shape_irreversible(self) -> None:
        assert mermaid_node_shape({"id": "s", "operation": "delete_file", "params": {}}) == ("{{", "}}")

    def test_render_mermaid_output(self) -> None:
        buf = io.StringIO()
        nodes = [{"id": "step_0", "operation": "read_file", "params": {"tool": "Read"}}]
        edges: list[dict[str, Any]] = []
        render_mermaid("sess-1", nodes, edges, file=buf)
        output = buf.getvalue()
        assert "flowchart TD" in output
        assert "sess-1" in output

    def test_render_mermaid_edges(self) -> None:
        buf = io.StringIO()
        nodes: list[dict[str, Any]] = []
        edges = [
            {"source_id": "a", "target_id": "b", "edge_type": "control_flow"},
            {"source_id": "b", "target_id": "c", "edge_type": "data_flow"},
        ]
        render_mermaid("s", nodes, edges, file=buf)
        output = buf.getvalue()
        assert "a --> b" in output
        assert "b -.-> c" in output


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TestRenderDagVisual:
    def test_default_compact(self) -> None:
        buf = io.StringIO()
        state = _sample_state(2, ["write_file"])
        render_dag_visual(state, "step_2", _sample_result(), file=buf)
        output = buf.getvalue()
        assert "◀ NOW" in output
        assert "█" in output  # score bars

    def test_full_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WFEVAL_VISUAL", "full")
        buf = io.StringIO()
        state = _sample_state(1, ["write_file"])
        render_dag_visual(state, "step_1", _sample_result(), file=buf)
        output = buf.getvalue()
        assert "╔" in output  # full graph double border

    def test_mermaid_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WFEVAL_VISUAL", "mermaid")
        buf = io.StringIO()
        state = _sample_state(1, ["write_file"])
        render_dag_visual(state, "step_1", _sample_result(), file=buf)
        output = buf.getvalue()
        assert "flowchart TD" in output

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WFEVAL_VISUAL", "FULL")
        buf = io.StringIO()
        state = _sample_state(1, ["write_file"])
        render_dag_visual(state, "step_1", _sample_result(), file=buf)
        output = buf.getvalue()
        assert "╔" in output
