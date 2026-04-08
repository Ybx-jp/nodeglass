"""MCP server implementation (NOD-36).

NOD-36 spec (Linear):
- mcp_server/server.py + mcp_server/tools.py
- 5 tools: analyze_workflow, check_step_risk, get_risk_report,
  record_outcome, find_similar_workflows
- Each tool validates input, calls core library, returns JSON

AC:
- [ ] MCP server starts without error
- [ ] Each tool callable via MCP protocol and returns correctly typed response
- [ ] Input validation rejects malformed requests with descriptive errors

Behavioral constraints:
- Validate input before calling core library
- Return JSON responses
- Descriptive error messages for malformed input
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from workflow_eval.mcp_server.tools import (
    analyze_workflow,
    check_step_risk,
    find_similar_workflows,
    get_risk_report,
    record_outcome,
)

mcp = FastMCP("workflow-eval", instructions="Risk scoring for AI agent workflows.")


@mcp.tool(name="analyze_workflow", description="Analyze a workflow DAG and return its risk profile with mitigations.")
def tool_analyze_workflow(workflow: dict[str, Any]) -> str:
    """Analyze a workflow DAG and return its risk profile.

    Args:
        workflow: A workflow definition with 'name', 'nodes', and 'edges'.
    """
    try:
        result = analyze_workflow(workflow)
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool(name="check_step_risk", description="Check the risk of a single operation, optionally in context of preceding operations.")
def tool_check_step_risk(
    operation: str,
    existing_operations: list[str] | None = None,
) -> str:
    """Check the risk of a single operation.

    Args:
        operation: Operation name to check (e.g., 'delete_record').
        existing_operations: Optional list of preceding operations for context scoring.
    """
    try:
        result = check_step_risk(operation, existing_operations)
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool(name="get_risk_report", description="Retrieve the risk report for a previously stored workflow.")
def tool_get_risk_report(workflow_id: str) -> str:
    """Retrieve the risk report for a stored workflow.

    Args:
        workflow_id: The ID of a previously analyzed and stored workflow.
    """
    try:
        result = get_risk_report(workflow_id)
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool(name="record_outcome", description="Record the outcome of a workflow execution.")
def tool_record_outcome(
    workflow_id: str,
    execution_id: str,
    records: list[dict[str, Any]],
    predicted_risk: float | None = None,
    actual_outcome: str | None = None,
) -> str:
    """Record the outcome of a workflow execution.

    Args:
        workflow_id: The ID of the workflow this execution belongs to.
        execution_id: Unique ID for this execution.
        records: List of operation outcome records.
        predicted_risk: Optional predicted risk score.
        actual_outcome: Optional overall outcome ('success', 'failure', 'skipped').
    """
    try:
        result = record_outcome(
            workflow_id, execution_id, records, predicted_risk, actual_outcome,
        )
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool(name="find_similar_workflows", description="Find structurally similar workflows (placeholder for Layer 3 embeddings).")
def tool_find_similar_workflows(
    workflow: dict[str, Any],
    top_k: int = 5,
) -> str:
    """Find structurally similar workflows.

    Args:
        workflow: A workflow definition to compare against stored workflows.
        top_k: Number of similar workflows to return.
    """
    try:
        result = find_similar_workflows(workflow, top_k)
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
