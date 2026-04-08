"""MCP server — stdio transport for agent integration."""

from workflow_eval.mcp_server.server import mcp
from workflow_eval.mcp_server.tools import (
    analyze_workflow,
    check_step_risk,
    find_similar_workflows,
    get_risk_report,
    record_outcome,
)

__all__ = [
    "analyze_workflow",
    "check_step_risk",
    "find_similar_workflows",
    "get_risk_report",
    "mcp",
    "record_outcome",
]
