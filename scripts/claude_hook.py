#!/usr/bin/env python3
"""Claude Code hook — maps tool calls to workflow-eval DAG and scores risk.

Reads PreToolUse events from stdin, maps Claude Code tools to ontology
operations, accumulates a session DAG in /tmp, scores it, and outputs
a permissionDecision for human review.

Usage (in .claude/settings.local.json):
  "hooks": {
    "PreToolUse": [{
      "matcher": ".*",
      "hooks": [{"type": "command", "command": ".venv/bin/python3 scripts/claude_hook.py"}]
    }]
  }
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tool → ontology operation mapping
# ---------------------------------------------------------------------------

_DIRECT_MAP: dict[str, str] = {
    "Read": "read_file",
    "Glob": "read_file",
    "Grep": "read_file",
    "Write": "write_file",
    "Edit": "write_file",
    "Agent": "execute_code",
    "WebFetch": "invoke_api",
    "WebSearch": "invoke_api",
}

# Pure-read operations skip scoring entirely (fast path).
_SKIP_SCORING_OPS = frozenset({"read_file", "read_state", "read_database"})

# Bash command → operation, first match wins.
_BASH_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\b"), "delete_file"),
    (re.compile(r"\bgit\s+push\b"), "send_webhook"),
    (re.compile(r"\bgit\s+(commit|add|reset|checkout|rebase|merge|stash)\b"), "mutate_state"),
    (re.compile(r"\b(curl|wget|gh|npm\s+publish)\b"), "invoke_api"),
    (re.compile(r"\b(pip|pip3)\s+install\b"), "create_resource"),
    (re.compile(r"\b(npm|yarn|pnpm)\s+install\b"), "create_resource"),
    (re.compile(r"\bapt(-get)?\s+(install|remove)\b"), "create_resource"),
    (re.compile(r"\bdocker\s+rm\b"), "destroy_resource"),
    (re.compile(r"\bkubectl\s+delete\b"), "destroy_resource"),
    (re.compile(r"\b(cat|head|tail|less|ls|find|wc|file|stat)\b"), "read_file"),
    (re.compile(r"\bgit\s+(log|status|diff|show|branch|tag)\b"), "read_file"),
]


def classify_bash(command: str) -> str:
    """Map a Bash command string to a workflow-eval operation."""
    for pattern, operation in _BASH_PATTERNS:
        if pattern.search(command):
            return operation
    return "execute_code"


def map_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Map a Claude Code tool invocation to a workflow-eval operation."""
    # MCP tools → invoke_api
    if tool_name.startswith("mcp__"):
        return "invoke_api"

    # Direct-mapped tools
    if tool_name in _DIRECT_MAP:
        return _DIRECT_MAP[tool_name]

    # Bash: classify by command content
    if tool_name == "Bash":
        return classify_bash(tool_input.get("command", ""))

    # Unknown tool → execute_code (conservative)
    return "execute_code"


# ---------------------------------------------------------------------------
# Session state (persisted to /tmp)
# ---------------------------------------------------------------------------

def _state_path(session_id: str) -> Path:
    return Path(f"/tmp/wfeval-{session_id}.json")


def load_state(session_id: str) -> dict[str, Any]:
    path = _state_path(session_id)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            print("workflow-eval: corrupt state file, starting fresh", file=sys.stderr)
    return {"session_id": session_id, "step_counter": 0, "nodes": [], "edges": []}


def save_state(state: dict[str, Any]) -> None:
    path = _state_path(state["session_id"])
    path.write_text(json.dumps(state, indent=2))


def append_node(
    state: dict[str, Any], operation: str, tool_name: str, tool_input: dict[str, Any],
) -> dict[str, Any]:
    """Add a node (and edge from previous) to the accumulated DAG."""
    step_id = f"step_{state['step_counter']}"
    state["nodes"].append({
        "id": step_id,
        "operation": operation,
        "params": {"tool": tool_name, **{k: str(v)[:80] for k, v in tool_input.items()}},
    })
    # Edge from previous node
    if state["step_counter"] > 0:
        prev_id = f"step_{state['step_counter'] - 1}"
        state["edges"].append({
            "source_id": prev_id,
            "target_id": step_id,
            "edge_type": "control_flow",
        })
    state["step_counter"] += 1
    return state


# ---------------------------------------------------------------------------
# Scoring (lazy import to keep fast-path cheap)
# ---------------------------------------------------------------------------

def score_dag(state: dict[str, Any]) -> dict[str, Any]:
    """Score the accumulated DAG and return a summary dict."""
    from workflow_eval.dag.models import to_networkx
    from workflow_eval.ontology.defaults import get_default_registry
    from workflow_eval.scoring.engine import RiskScoringEngine
    from workflow_eval.types import ScoringConfig, WorkflowDAG

    dag = WorkflowDAG.model_validate({
        "name": f"session-{state['session_id'][:8]}",
        "nodes": state["nodes"],
        "edges": state["edges"],
    })
    nx_dag = to_networkx(dag)
    registry = get_default_registry()
    engine = RiskScoringEngine(ScoringConfig(), registry)
    profile = engine.score(nx_dag)

    top_scores = sorted(profile.sub_scores, key=lambda s: s.score, reverse=True)[:3]
    breakdown = ", ".join(f"{s.name}: {s.score:.2f}" for s in top_scores)

    return {
        "aggregate": profile.aggregate_score,
        "risk_level": profile.risk_level.value,
        "breakdown": breakdown,
        "node_count": profile.node_count,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.exit(0)

    event = json.loads(raw)
    hook_event = event.get("hook_event_name", "")

    if hook_event != "PreToolUse":
        # Not our event — pass through
        sys.exit(0)

    tool_name = event.get("tool_name", "")
    tool_input = event.get("tool_input", {})
    session_id = event.get("session_id", "unknown")

    # Map tool → operation
    operation = map_tool(tool_name, tool_input)

    # Always accumulate into session DAG (even reads, for complete context)
    state = load_state(session_id)
    state = append_node(state, operation, tool_name, tool_input)
    save_state(state)

    # Fast path: pure reads — tracked but not scored/prompted
    if operation in _SKIP_SCORING_OPS:
        sys.exit(0)

    # Score the full accumulated DAG
    try:
        result = score_dag(state)
    except Exception as exc:
        print(f"workflow-eval: scoring error: {exc}", file=sys.stderr)
        sys.exit(0)

    # Build reason string
    step_id = f"step_{state['step_counter'] - 1}"
    tool_desc = tool_name
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        tool_desc = f"Bash({cmd[:60]})"

    reason = (
        f"[WORKFLOW-EVAL] {result['aggregate']:.2f} {result['risk_level'].upper()} "
        f"({result['node_count']} ops) | {result['breakdown']} | "
        f"{step_id}: {operation} via {tool_desc}"
    )

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": reason,
        }
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
