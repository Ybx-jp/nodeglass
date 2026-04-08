"""HTTP route handlers (NOD-37).

NOD-37 spec (Linear):
- api/app.py + api/routes.py
- 7 endpoints:
  POST /api/v1/workflows/analyze
  POST /api/v1/workflows/check-step
  GET  /api/v1/workflows/{id}/report
  POST /api/v1/executions/record
  POST /api/v1/workflows/find-similar
  GET  /api/v1/ontology
  GET  /api/v1/health
- Auto-generated OpenAPI docs at /docs

AC:
- [ ] All endpoints return correct status codes and response shapes
- [ ] /docs renders OpenAPI spec
- [ ] /health returns {"status": "ok", ...}

Behavioral constraints:
- Reuse MCP tool handlers for core logic
- Return proper HTTP status codes (200, 422)
- Descriptive error messages for invalid input
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from workflow_eval.mcp_server.tools import (
    analyze_workflow,
    check_step_risk,
    find_similar_workflows,
    get_risk_report,
    record_outcome,
)
from workflow_eval.ontology.defaults import get_default_registry


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    workflow: dict[str, Any]


class CheckStepRequest(BaseModel):
    operation: str
    existing_operations: list[str] | None = None


class RecordOutcomeRequest(BaseModel):
    workflow_id: str
    execution_id: str
    records: list[dict[str, Any]]
    predicted_risk: float | None = None
    actual_outcome: str | None = None


class FindSimilarRequest(BaseModel):
    workflow: dict[str, Any]
    top_k: int = 5


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/workflows/analyze")
def api_analyze_workflow(body: AnalyzeRequest) -> dict[str, Any]:
    try:
        return analyze_workflow(body.workflow)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/workflows/check-step")
def api_check_step_risk(body: CheckStepRequest) -> dict[str, Any]:
    try:
        return check_step_risk(body.operation, body.existing_operations)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/workflows/{workflow_id}/report")
def api_get_risk_report(workflow_id: str) -> dict[str, Any]:
    try:
        return get_risk_report(workflow_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/executions/record")
def api_record_outcome(body: RecordOutcomeRequest) -> dict[str, Any]:
    try:
        return record_outcome(
            body.workflow_id,
            body.execution_id,
            body.records,
            body.predicted_risk,
            body.actual_outcome,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/workflows/find-similar")
def api_find_similar_workflows(body: FindSimilarRequest) -> dict[str, Any]:
    try:
        return find_similar_workflows(body.workflow, body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/ontology")
def api_ontology() -> dict[str, Any]:
    registry = get_default_registry()
    return {
        "operations": [
            {
                "name": op.name,
                "category": op.category,
                "base_risk_weight": op.base_risk_weight,
                "effect_type": op.effect_type.value,
                "effect_targets": sorted(t.value for t in op.effect_targets),
            }
            for op in registry.all()
        ],
        "count": len(registry),
    }
