"""Shared DAG rendering for hook visualization and inspector (NOD-51).

NOD-51 spec (Linear):
- Compact trail view with ◀ NOW marker, collapsed consecutive ops, score bars
- Full graph mode via WFEVAL_VISUAL=full env var
- Mermaid export via WFEVAL_VISUAL=mermaid
- ANSI colors with NO_COLOR / non-TTY fallback
- Shared module between hook and inspector

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

import os
import sys
from typing import Any, TextIO

# ---------------------------------------------------------------------------
# Operation classification (shared with inspector)
# ---------------------------------------------------------------------------

IRREVERSIBLE_OPS = frozenset({
    "delete_file", "delete_record", "destroy_resource", "send_email",
})
EXTERNAL_OPS = frozenset({
    "invoke_api", "send_webhook", "execute_code", "send_notification",
    "authenticate", "create_resource",
})
STATEFUL_OPS = frozenset({
    "write_file", "write_database", "mutate_state",
})


def op_category(operation: str) -> str:
    """Classify an operation into a risk category."""
    if operation in IRREVERSIBLE_OPS:
        return "irreversible"
    if operation in EXTERNAL_OPS:
        return "external"
    if operation in STATEFUL_OPS:
        return "stateful"
    return "pure"


# ---------------------------------------------------------------------------
# ANSI color support
# ---------------------------------------------------------------------------

class _Colors:
    """ANSI escape codes with NO_COLOR / non-TTY fallback."""

    def __init__(self, file: TextIO) -> None:
        use_color = (
            file.isatty()
            and "NO_COLOR" not in os.environ
        )
        if use_color:
            self.reset = "\033[0m"
            self.bold = "\033[1m"
            self.dim = "\033[2m"
            self.green = "\033[32m"
            self.yellow = "\033[33m"
            self.red = "\033[31m"
            self.blue = "\033[34m"
            self.cyan = "\033[36m"
            self.bg_red = "\033[41m"
            self.white = "\033[97m"
        else:
            self.reset = ""
            self.bold = ""
            self.dim = ""
            self.green = ""
            self.yellow = ""
            self.red = ""
            self.blue = ""
            self.cyan = ""
            self.bg_red = ""
            self.white = ""

    def for_op(self, operation: str) -> str:
        """Return the color code for an operation's risk category."""
        cat = op_category(operation)
        if cat == "irreversible":
            return self.red
        if cat == "external":
            return self.blue
        if cat == "stateful":
            return self.yellow
        return self.green


# ---------------------------------------------------------------------------
# Score bars
# ---------------------------------------------------------------------------

_BAR_WIDTH = 20
_FULL = "█"
_PARTIAL = "▒"
_EMPTY = "░"


def render_score_bars(
    sub_scores: list[dict[str, Any]],
    file: TextIO = sys.stderr,
) -> None:
    """Render top 3 sub-score bars with Unicode block characters."""
    c = _Colors(file)
    top3 = sorted(sub_scores, key=lambda s: s["score"], reverse=True)[:3]

    for s in top3:
        score = s["score"]
        name = s["name"]
        filled = int(score * _BAR_WIDTH)
        partial = 1 if (score * _BAR_WIDTH) % 1 >= 0.25 and filled < _BAR_WIDTH else 0
        empty = _BAR_WIDTH - filled - partial

        # Color the bar by score severity
        if score >= 0.75:
            bar_color = c.red
        elif score >= 0.25:
            bar_color = c.yellow
        else:
            bar_color = c.green

        bar = (
            f"{bar_color}{_FULL * filled}{_PARTIAL * partial}{c.dim}{_EMPTY * empty}{c.reset}"
        )
        file.write(f"    {name:<18} {bar} {score:.2f}\n")


# ---------------------------------------------------------------------------
# Compact trail
# ---------------------------------------------------------------------------

def _collapse_trail(nodes: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Collapse consecutive same-operation nodes into (operation, count) tuples."""
    if not nodes:
        return []
    groups: list[tuple[str, int]] = []
    current_op = nodes[0]["operation"]
    count = 1
    for node in nodes[1:]:
        if node["operation"] == current_op:
            count += 1
        else:
            groups.append((current_op, count))
            current_op = node["operation"]
            count = 1
    groups.append((current_op, count))
    return groups


def render_compact_trail(
    state: dict[str, Any],
    current_step: str,
    result: dict[str, Any],
    file: TextIO = sys.stderr,
) -> None:
    """Render the compact trail view to file.

    Shows: header line, collapsed operation trail with ◀ NOW marker,
    current tool description, and top 3 sub-score bars.
    """
    c = _Colors(file)
    nodes = state.get("nodes", [])
    if not nodes:
        return

    # Header
    agg = result["aggregate"]
    level = result["risk_level"].upper()
    n_ops = result["node_count"]

    if level == "CRITICAL":
        level_color = c.bg_red + c.white
    elif level == "HIGH":
        level_color = c.red
    elif level == "MEDIUM":
        level_color = c.yellow
    else:
        level_color = c.green

    file.write(
        f"{c.bold}[WORKFLOW-EVAL]{c.reset} "
        f"{agg:.2f} {level_color}{level}{c.reset} "
        f"({n_ops} ops)\n"
    )

    # Trail: collapse consecutive same-type ops, highlight current
    groups = _collapse_trail(nodes)
    trail_parts: list[str] = []

    # Track which group the current step falls in
    node_idx = 0
    for i, (op, count) in enumerate(groups):
        color = c.for_op(op)
        is_last_group = (i == len(groups) - 1)

        # Check if current_step is in this group
        group_end_idx = node_idx + count - 1
        contains_current = any(
            nodes[j]["id"] == current_step
            for j in range(node_idx, min(node_idx + count, len(nodes)))
        )

        if count > 1:
            label = f"{op} ×{count}"
        else:
            label = op

        if contains_current:
            trail_parts.append(
                f"{c.bold}[{color}{label}{c.reset}{c.bold}] ◀ NOW{c.reset}"
            )
        else:
            trail_parts.append(f"{color}{label}{c.reset}")

        node_idx += count

    trail = f" {c.dim}→{c.reset} ".join(trail_parts)
    file.write(f"─── {trail}\n")

    # Current tool description
    current_node = next((n for n in nodes if n["id"] == current_step), None)
    if current_node:
        params = current_node.get("params", {})
        tool = params.get("tool", "?")
        # Build compact description
        desc_parts = []
        for k, v in params.items():
            if k == "tool":
                continue
            desc_parts.append(f"{str(v)[:50]}")
        desc = ", ".join(desc_parts) if desc_parts else ""
        if desc:
            file.write(f"    {c.dim}{tool}({desc}){c.reset}\n")

    # Score bars
    if "sub_scores" in result:
        render_score_bars(result["sub_scores"], file)

    file.write("\n")


# ---------------------------------------------------------------------------
# Full graph (ASCII box-drawing)
# ---------------------------------------------------------------------------

def render_full_graph(
    state: dict[str, Any],
    current_step: str,
    result: dict[str, Any],
    file: TextIO = sys.stderr,
) -> None:
    """Render a full ASCII box-drawing DAG to file."""
    c = _Colors(file)
    nodes = state.get("nodes", [])
    edges = state.get("edges", [])
    if not nodes:
        return

    # Header
    agg = result["aggregate"]
    level = result["risk_level"].upper()
    n_ops = result["node_count"]

    if level in ("CRITICAL", "HIGH"):
        level_color = c.red
    elif level == "MEDIUM":
        level_color = c.yellow
    else:
        level_color = c.green

    file.write(
        f"{c.bold}[WORKFLOW-EVAL]{c.reset} "
        f"{agg:.2f} {level_color}{level}{c.reset} "
        f"({n_ops} ops)\n\n"
    )

    # Build adjacency for layout
    children: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for edge in edges:
        src = edge["source_id"]
        tgt = edge["target_id"]
        if src in children:
            children[src].append(tgt)

    node_map = {n["id"]: n for n in nodes}

    # Render each node as a box in sequence
    for node in nodes:
        nid = node["id"]
        op = node["operation"]
        color = c.for_op(op)
        is_current = (nid == current_step)

        tool = node.get("params", {}).get("tool", "")
        tool_suffix = f"  ({tool})" if tool else ""

        label = f"{op}{tool_suffix}"
        box_width = max(len(label), len(nid)) + 4

        border_char = "═" if is_current else "─"
        corner_tl = "╔" if is_current else "┌"
        corner_tr = "╗" if is_current else "┐"
        corner_bl = "╚" if is_current else "└"
        corner_br = "╝" if is_current else "┘"
        side = "║" if is_current else "│"

        now_marker = f" {c.bold}◀ NOW{c.reset}" if is_current else ""

        file.write(f"  {color}{corner_tl}{border_char * box_width}{corner_tr}{c.reset}{now_marker}\n")
        file.write(f"  {color}{side}{c.reset} {c.bold if is_current else ''}{label:<{box_width - 2}}{c.reset if is_current else ''} {color}{side}{c.reset}\n")
        file.write(f"  {color}{side}{c.reset} {c.dim}{nid:<{box_width - 2}}{c.reset} {color}{side}{c.reset}\n")
        file.write(f"  {color}{corner_bl}{border_char * box_width}{corner_br}{c.reset}\n")

        # Arrow to next
        if children.get(nid):
            file.write(f"  {c.dim}  │{c.reset}\n")
            file.write(f"  {c.dim}  ▼{c.reset}\n")

    # Score bars
    file.write("\n")
    if "sub_scores" in result:
        render_score_bars(result["sub_scores"], file)

    file.write("\n")


# ---------------------------------------------------------------------------
# Mermaid (reusable from inspector)
# ---------------------------------------------------------------------------

def mermaid_node_label(node: dict[str, Any]) -> str:
    """Build a display label for a Mermaid node."""
    op = node["operation"]
    tool = node.get("params", {}).get("tool", "")
    if tool and tool != op:
        return f"{node['id']}\\n{op}\\n({tool})"
    return f"{node['id']}\\n{op}"


def mermaid_node_shape(node: dict[str, Any]) -> tuple[str, str]:
    """Return (open_bracket, close_bracket) for Mermaid node shape.

    Shapes by effect type:
      pure         → rounded stadium  ("...")
      stateful     → rectangle         [...]
      external     → subroutine        [[...]]
      irreversible → hexagon            {{...}}
    """
    cat = op_category(node["operation"])
    if cat == "irreversible":
        return "{{", "}}"
    if cat == "external":
        return "[[", "]]"
    if cat == "stateful":
        return "[", "]"
    return "(", ")"


def render_mermaid(
    session_id: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    file: TextIO = sys.stdout,
) -> None:
    """Output a valid Mermaid flowchart."""
    file.write("flowchart TD\n")
    file.write(f"  %% Session: {session_id}\n")

    for node in nodes:
        nid = node["id"]
        label = mermaid_node_label(node)
        open_b, close_b = mermaid_node_shape(node)
        file.write(f"  {nid}{open_b}\"{label}\"{close_b}\n")

    for edge in edges:
        etype = edge.get("edge_type", "control_flow")
        if etype == "data_flow":
            file.write(f"  {edge['source_id']} -.-> {edge['target_id']}\n")
        elif etype == "conditional":
            cond = edge.get("condition", "?")
            file.write(f"  {edge['source_id']} -->|{cond}| {edge['target_id']}\n")
        else:
            file.write(f"  {edge['source_id']} --> {edge['target_id']}\n")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def render_dag_visual(
    state: dict[str, Any],
    current_step: str,
    result: dict[str, Any],
    file: TextIO = sys.stderr,
) -> None:
    """Render DAG visualization based on WFEVAL_VISUAL env var.

    Modes:
      (unset/compact) → compact trail (default)
      full            → ASCII box-drawing graph
      mermaid         → Mermaid flowchart
    """
    mode = os.environ.get("WFEVAL_VISUAL", "compact").lower()

    if mode == "mermaid":
        sid = state.get("session_id", "unknown")
        render_mermaid(sid, state.get("nodes", []), state.get("edges", []), file=file)
    elif mode == "full":
        render_full_graph(state, current_step, result, file=file)
    else:
        render_compact_trail(state, current_step, result, file=file)
