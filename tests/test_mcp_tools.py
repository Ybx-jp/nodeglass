"""Tests for MCP server tools (NOD-36).

AC:
- [x] MCP server starts without error
- [x] Each tool callable via MCP protocol and returns correctly typed response
- [x] Input validation rejects malformed requests with descriptive errors

Behavioral constraints:
- Validate input before calling core library
- Return JSON responses
- Descriptive error messages for malformed input
"""

import json

import pytest

from workflow_eval.mcp_server.server import mcp
from workflow_eval.mcp_server.tools import (
    analyze_workflow,
    check_step_risk,
    find_similar_workflows,
    get_risk_report,
    record_outcome,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

SIMPLE_WORKFLOW = {
    "name": "test-wf",
    "nodes": [
        {"id": "n1", "operation": "read_file", "params": {}},
        {"id": "n2", "operation": "invoke_api", "params": {}},
    ],
    "edges": [
        {"source_id": "n1", "target_id": "n2", "edge_type": "control_flow"},
    ],
}

RISKY_WORKFLOW = {
    "name": "risky-wf",
    "nodes": [
        {"id": "n1", "operation": "authenticate", "params": {}},
        {"id": "n2", "operation": "read_database", "params": {}},
        {"id": "n3", "operation": "delete_record", "params": {}},
    ],
    "edges": [
        {"source_id": "n1", "target_id": "n2", "edge_type": "control_flow"},
        {"source_id": "n2", "target_id": "n3", "edge_type": "control_flow"},
    ],
}


async def _call(tool_name: str, args: dict) -> dict:
    """Call a tool via MCP protocol and parse the JSON result."""
    content_blocks, _ = await mcp.call_tool(tool_name, args)
    text = content_blocks[0].text
    return json.loads(text)


# ---------------------------------------------------------------------------
# AC: MCP server starts without error
# ---------------------------------------------------------------------------


class TestServerStartup:
    def test_server_imports_without_error(self) -> None:
        assert mcp is not None
        assert mcp.name == "workflow-eval"

    @pytest.mark.asyncio()
    async def test_list_tools_returns_five(self) -> None:
        tools = await mcp.list_tools()
        names = sorted(t.name for t in tools)
        assert names == [
            "analyze_workflow",
            "check_step_risk",
            "find_similar_workflows",
            "get_risk_report",
            "record_outcome",
        ]

    @pytest.mark.asyncio()
    async def test_each_tool_has_description(self) -> None:
        tools = await mcp.list_tools()
        for tool in tools:
            assert tool.description, f"Tool {tool.name} missing description"


# ---------------------------------------------------------------------------
# AC: Each tool callable via MCP protocol and returns correctly typed response
# ---------------------------------------------------------------------------


class TestAnalyzeWorkflow:
    @pytest.mark.asyncio()
    async def test_via_mcp_protocol(self) -> None:
        result = await _call("analyze_workflow", {"workflow": SIMPLE_WORKFLOW})
        assert result["workflow_name"] == "test-wf"
        assert isinstance(result["aggregate_score"], float)
        assert result["risk_level"] in ("low", "medium", "high", "critical")
        assert isinstance(result["sub_scores"], list)
        assert len(result["sub_scores"]) == 6
        assert isinstance(result["critical_paths"], list)
        assert isinstance(result["chokepoints"], list)
        assert "mitigation_plan" in result

    def test_direct_call(self) -> None:
        result = analyze_workflow(SIMPLE_WORKFLOW)
        assert result["workflow_name"] == "test-wf"
        assert "aggregate_score" in result
        assert "mitigation_plan" in result

    @pytest.mark.asyncio()
    async def test_risky_workflow_higher_score(self) -> None:
        simple = await _call("analyze_workflow", {"workflow": SIMPLE_WORKFLOW})
        risky = await _call("analyze_workflow", {"workflow": RISKY_WORKFLOW})
        assert risky["aggregate_score"] >= simple["aggregate_score"]

    @pytest.mark.asyncio()
    async def test_mitigation_plan_structure(self) -> None:
        result = await _call("analyze_workflow", {"workflow": RISKY_WORKFLOW})
        plan = result["mitigation_plan"]
        assert "original_risk" in plan
        assert "residual_risk" in plan
        assert isinstance(plan["mitigations"], list)

    @pytest.mark.asyncio()
    async def test_sub_scores_have_required_fields(self) -> None:
        result = await _call("analyze_workflow", {"workflow": SIMPLE_WORKFLOW})
        for sub in result["sub_scores"]:
            assert "name" in sub
            assert "score" in sub
            assert "weight" in sub


class TestCheckStepRisk:
    @pytest.mark.asyncio()
    async def test_known_operation(self) -> None:
        result = await _call("check_step_risk", {"operation": "delete_record"})
        assert result["operation"] == "delete_record"
        assert isinstance(result["base_risk_weight"], (int, float))
        assert "effect_type" in result
        assert "effect_targets" in result

    @pytest.mark.asyncio()
    async def test_with_context(self) -> None:
        result = await _call("check_step_risk", {
            "operation": "delete_record",
            "existing_operations": ["authenticate", "read_database"],
        })
        assert "contextual_risk" in result
        assert "aggregate_score" in result["contextual_risk"]
        assert "risk_level" in result["contextual_risk"]

    def test_direct_call_no_context(self) -> None:
        result = check_step_risk("read_file")
        assert result["operation"] == "read_file"
        assert "contextual_risk" not in result

    def test_direct_call_with_context(self) -> None:
        result = check_step_risk("delete_record", ["authenticate"])
        assert "contextual_risk" in result


class TestGetRiskReport:
    @pytest.mark.asyncio()
    async def test_nonexistent_workflow_returns_error(self) -> None:
        result = await _call("get_risk_report", {"workflow_id": "nonexistent"})
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_direct_call_nonexistent(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_risk_report("nonexistent")


class TestRecordOutcome:
    @pytest.mark.asyncio()
    async def test_nonexistent_workflow_returns_error(self) -> None:
        result = await _call("record_outcome", {
            "workflow_id": "nonexistent",
            "execution_id": "exec-1",
            "records": [
                {"node_id": "n1", "operation": "read_file", "outcome": "success"},
            ],
        })
        assert "error" in result

    @pytest.mark.asyncio()
    async def test_empty_records_returns_error(self) -> None:
        result = await _call("record_outcome", {
            "workflow_id": "any",
            "execution_id": "exec-1",
            "records": [],
        })
        assert "error" in result
        assert "empty" in result["error"].lower()

    def test_direct_call_empty_records(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            record_outcome("wf-1", "exec-1", [])

    @pytest.mark.asyncio()
    async def test_invalid_outcome_returns_error(self) -> None:
        result = await _call("record_outcome", {
            "workflow_id": "any",
            "execution_id": "exec-1",
            "records": [
                {"node_id": "n1", "operation": "read_file", "outcome": "success"},
            ],
            "actual_outcome": "bogus",
        })
        assert "error" in result
        assert "invalid outcome" in result["error"].lower()

    def test_direct_call_invalid_outcome(self) -> None:
        records = [{"node_id": "n1", "operation": "read_file", "outcome": "success"}]
        with pytest.raises(ValueError, match="Invalid outcome"):
            record_outcome("wf-1", "exec-1", records, actual_outcome="bogus")


class TestFindSimilarWorkflows:
    @pytest.mark.asyncio()
    async def test_via_mcp_protocol(self) -> None:
        result = await _call("find_similar_workflows", {"workflow": SIMPLE_WORKFLOW})
        assert result["query_workflow"] == "test-wf"
        assert result["similar_workflows"] == []
        assert "placeholder" in result["note"].lower() or "Layer 3" in result["note"]

    @pytest.mark.asyncio()
    async def test_custom_top_k(self) -> None:
        result = await _call("find_similar_workflows", {
            "workflow": SIMPLE_WORKFLOW,
            "top_k": 10,
        })
        assert result["top_k"] == 10

    def test_direct_call(self) -> None:
        result = find_similar_workflows(SIMPLE_WORKFLOW)
        assert result["node_count"] == 2
        assert result["edge_count"] == 1


# ---------------------------------------------------------------------------
# AC: Input validation rejects malformed requests with descriptive errors
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.asyncio()
    async def test_analyze_invalid_workflow(self) -> None:
        result = await _call("analyze_workflow", {"workflow": {"bad": "data"}})
        assert "error" in result
        assert "invalid workflow" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_analyze_empty_workflow(self) -> None:
        result = await _call("analyze_workflow", {"workflow": {}})
        assert "error" in result

    @pytest.mark.asyncio()
    async def test_check_step_unknown_operation(self) -> None:
        result = await _call("check_step_risk", {"operation": "nonexistent_op"})
        assert "error" in result
        assert "unknown operation" in result["error"].lower()

    @pytest.mark.asyncio()
    async def test_find_similar_invalid_workflow(self) -> None:
        result = await _call("find_similar_workflows", {"workflow": {"not": "valid"}})
        assert "error" in result

    @pytest.mark.asyncio()
    async def test_record_outcome_invalid_record_format(self) -> None:
        result = await _call("record_outcome", {
            "workflow_id": "wf-1",
            "execution_id": "exec-1",
            "records": [{"missing_required": True}],
        })
        assert "error" in result

    def test_direct_analyze_invalid_workflow(self) -> None:
        with pytest.raises(ValueError, match="Invalid workflow"):
            analyze_workflow({"bad": "data"})

    def test_direct_check_step_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown operation"):
            check_step_risk("totally_fake_op")

    def test_direct_find_similar_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid workflow"):
            find_similar_workflows({"bad": "data"})
