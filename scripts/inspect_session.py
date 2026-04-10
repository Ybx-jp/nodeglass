#!/usr/bin/env python3
"""DAG inspector — inspect accumulated session DAGs from /tmp (NOD-45).

NOD-45 spec (Linear):
- scripts/inspect_session.py
- List mode: --list shows all /tmp/wfeval-*.json files with summary stats
- Inspect mode: <session_id> prints node table, edge list, histogram, risk profile
- Mermaid export: --mermaid outputs valid Mermaid flowchart syntax
- Handles missing/corrupt state files gracefully

AC:
- [x] --list shows all /tmp/wfeval-*.json files with summary stats
- [x] Inspect mode prints node table, edge list, operation histogram, and full risk profile
- [x] --mermaid outputs valid Mermaid flowchart syntax
- [x] Script handles missing/corrupt state files gracefully
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STATE_DIR = Path("/tmp")
_STATE_GLOB = "wfeval-*.json"


# ---------------------------------------------------------------------------
# State file loading
# ---------------------------------------------------------------------------

def _find_state_files() -> list[Path]:
    """Find all wfeval session state files in /tmp, sorted by mtime desc."""
    files = list(_STATE_DIR.glob(_STATE_GLOB))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _load_state(path: Path) -> dict[str, Any] | None:
    """Load and validate a state file. Returns None on corruption."""
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "nodes" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _resolve_session(session_id: str) -> Path | None:
    """Resolve a session ID (full or prefix) to a state file path."""
    # Exact match
    exact = _STATE_DIR / f"wfeval-{session_id}.json"
    if exact.exists():
        return exact

    # Prefix match
    matches = [p for p in _STATE_DIR.glob(f"wfeval-{session_id}*.json")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous prefix '{session_id}' matches {len(matches)} files:", file=sys.stderr)
        for m in matches:
            print(f"  {m.name}", file=sys.stderr)
        return None
    return None


# ---------------------------------------------------------------------------
# List mode
# ---------------------------------------------------------------------------

def cmd_list() -> None:
    """Show all session files with summary stats."""
    files = _find_state_files()
    if not files:
        print("No session DAGs found in /tmp.")
        return

    print(f"{'SESSION ID':<40} {'NODES':>5} {'EDGES':>5} {'LAST MODIFIED':<20}")
    print("-" * 75)

    for path in files:
        state = _load_state(path)
        sid = path.stem.removeprefix("wfeval-")
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S")

        if state is None:
            print(f"{sid:<40} {'ERR':>5} {'ERR':>5} {mtime_str:<20}  (corrupt)")
        else:
            n_nodes = len(state.get("nodes", []))
            n_edges = len(state.get("edges", []))
            print(f"{sid:<40} {n_nodes:>5} {n_edges:>5} {mtime_str:<20}")


# ---------------------------------------------------------------------------
# Inspect mode
# ---------------------------------------------------------------------------

def cmd_inspect(session_id: str, mermaid: bool) -> None:
    """Print full DAG summary for a session."""
    path = _resolve_session(session_id)
    if path is None:
        print(f"Session not found: {session_id}", file=sys.stderr)
        print("Use --list to see available sessions.", file=sys.stderr)
        sys.exit(1)

    state = _load_state(path)
    if state is None:
        print(f"Corrupt state file: {path}", file=sys.stderr)
        sys.exit(1)

    nodes = state.get("nodes", [])
    edges = state.get("edges", [])
    sid = state.get("session_id", session_id)

    if mermaid:
        _print_mermaid(sid, nodes, edges)
        return

    _print_node_table(nodes)
    _print_edge_list(edges)
    _print_histogram(nodes)
    _print_risk_profile(state)


def _print_node_table(nodes: list[dict[str, Any]]) -> None:
    """Print a formatted table of all nodes."""
    print("\n=== Nodes ===")
    print(f"{'STEP':<10} {'OPERATION':<20} {'TOOL':<12} {'PARAMS'}")
    print("-" * 75)
    for node in nodes:
        step_id = node["id"]
        operation = node["operation"]
        params = node.get("params", {})
        tool = params.get("tool", "?")
        # Build truncated param string (exclude 'tool' key)
        param_parts = []
        for k, v in params.items():
            if k == "tool":
                continue
            param_parts.append(f"{k}={str(v)[:40]}")
        param_str = ", ".join(param_parts) if param_parts else "-"
        print(f"{step_id:<10} {operation:<20} {tool:<12} {param_str}")


def _print_edge_list(edges: list[dict[str, Any]]) -> None:
    """Print all edges."""
    print(f"\n=== Edges ({len(edges)}) ===")
    if not edges:
        print("  (none)")
        return
    for edge in edges:
        etype = edge.get("edge_type", "control_flow")
        print(f"  {edge['source_id']} -> {edge['target_id']}  [{etype}]")


def _print_histogram(nodes: list[dict[str, Any]]) -> None:
    """Print operation frequency histogram."""
    print("\n=== Operation Histogram ===")
    counts: Counter[str] = Counter(n["operation"] for n in nodes)
    for op, count in counts.most_common():
        bar = "#" * count
        print(f"  {op:<20} {count:>3}  {bar}")


def _print_risk_profile(state: dict[str, Any]) -> None:
    """Score the DAG and print the full risk profile."""
    print("\n=== Risk Profile ===")

    nodes = state.get("nodes", [])
    if not nodes:
        print("  (empty DAG — nothing to score)")
        return

    try:
        from workflow_eval.dag.models import to_networkx
        from workflow_eval.ontology.defaults import get_default_registry
        from workflow_eval.scoring.engine import RiskScoringEngine
        from workflow_eval.types import ScoringConfig, WorkflowDAG

        sid = state.get("session_id", "unknown")
        dag = WorkflowDAG.model_validate({
            "name": f"session-{sid[:8]}",
            "nodes": state["nodes"],
            "edges": state["edges"],
        })
        nx_dag = to_networkx(dag)
        registry = get_default_registry()
        engine = RiskScoringEngine(ScoringConfig(), registry)
        profile = engine.score(nx_dag)

        print(f"  Aggregate: {profile.aggregate_score:.3f}  ({profile.risk_level.value.upper()})")
        print(f"  Nodes: {profile.node_count}  Edges: {profile.edge_count}")
        print()
        print(f"  {'SCORER':<20} {'SCORE':>6} {'WEIGHT':>7} {'WEIGHTED':>8}")
        print(f"  {'-' * 45}")
        for sub in profile.sub_scores:
            weighted = sub.score * sub.weight
            print(f"  {sub.name:<20} {sub.score:>6.3f} {sub.weight:>7.2f} {weighted:>8.3f}")

        if profile.critical_paths:
            print(f"\n  Critical path: {' -> '.join(profile.critical_paths[0])}")
        if profile.chokepoints:
            print(f"  Chokepoints: {', '.join(profile.chokepoints)}")

    except Exception as exc:
        print(f"  Scoring failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Mermaid export
# ---------------------------------------------------------------------------

def _mermaid_node_label(node: dict[str, Any]) -> str:
    """Build a display label for a Mermaid node."""
    op = node["operation"]
    tool = node.get("params", {}).get("tool", "")
    if tool and tool != op:
        return f"{node['id']}\\n{op}\\n({tool})"
    return f"{node['id']}\\n{op}"


_IRREVERSIBLE_OPS = frozenset({
    "delete_file", "delete_record", "destroy_resource", "send_email",
})
_EXTERNAL_OPS = frozenset({
    "invoke_api", "send_webhook", "execute_code", "send_notification",
    "authenticate", "create_resource",
})
_STATEFUL_OPS = frozenset({
    "write_file", "write_database", "mutate_state",
})


def _mermaid_node_shape(node: dict[str, Any]) -> tuple[str, str]:
    """Return (open_bracket, close_bracket) for Mermaid node shape.

    Shapes by effect type:
      pure         → rounded stadium  ("...")
      stateful     → rectangle         [...]
      external     → subroutine        [[...]]
      irreversible → hexagon            {{...}}
    """
    op = node["operation"]
    if op in _IRREVERSIBLE_OPS:
        return "{{", "}}"
    if op in _EXTERNAL_OPS:
        return "[[", "]]"
    if op in _STATEFUL_OPS:
        return "[", "]"
    # pure (reads, control flow)
    return "(", ")"


def _print_mermaid(session_id: str, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    """Output a valid Mermaid flowchart."""
    print(f"flowchart TD")
    print(f"  %% Session: {session_id}")

    for node in nodes:
        nid = node["id"]
        label = _mermaid_node_label(node)
        open_b, close_b = _mermaid_node_shape(node)
        print(f"  {nid}{open_b}\"{label}\"{close_b}")

    for edge in edges:
        etype = edge.get("edge_type", "control_flow")
        if etype == "data_flow":
            print(f"  {edge['source_id']} -.-> {edge['target_id']}")
        elif etype == "conditional":
            cond = edge.get("condition", "?")
            print(f"  {edge['source_id']} -->|{cond}| {edge['target_id']}")
        else:
            print(f"  {edge['source_id']} --> {edge['target_id']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="inspect_session",
        description="Inspect workflow-eval session DAGs accumulated by Claude Code hooks.",
    )
    parser.add_argument(
        "session_id", nargs="?", default=None,
        help="Session ID (or prefix) to inspect. Omit to use --list.",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_mode",
        help="List all session files with summary stats.",
    )
    parser.add_argument(
        "--mermaid", action="store_true",
        help="Output Mermaid flowchart instead of text summary.",
    )

    args = parser.parse_args()

    if args.list_mode or args.session_id is None:
        cmd_list()
    else:
        cmd_inspect(args.session_id, args.mermaid)


if __name__ == "__main__":
    main()
