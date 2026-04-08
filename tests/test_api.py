"""Tests for FastAPI HTTP endpoints (NOD-37).

AC:
- [x] All endpoints return correct status codes and response shapes
- [x] /docs renders OpenAPI spec
- [x] /health returns {"status": "ok", ...}
"""

import pytest
from fastapi.testclient import TestClient

from workflow_eval.api.app import app

client = TestClient(app)


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


# ---------------------------------------------------------------------------
# AC: /health returns {"status": "ok", ...}
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_json_content_type(self) -> None:
        resp = client.get("/api/v1/health")
        assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# AC: /docs renders OpenAPI spec
# ---------------------------------------------------------------------------


class TestOpenAPI:
    def test_docs_endpoint(self) -> None:
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_openapi_json(self) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        spec = resp.json()
        assert "paths" in spec
        assert "/api/v1/health" in spec["paths"]
        assert "/api/v1/workflows/analyze" in spec["paths"]
        assert "/api/v1/workflows/check-step" in spec["paths"]
        assert "/api/v1/workflows/{workflow_id}/report" in spec["paths"]
        assert "/api/v1/executions/record" in spec["paths"]
        assert "/api/v1/workflows/find-similar" in spec["paths"]
        assert "/api/v1/ontology" in spec["paths"]


# ---------------------------------------------------------------------------
# AC: All endpoints return correct status codes and response shapes
# ---------------------------------------------------------------------------


class TestAnalyzeWorkflow:
    def test_success(self) -> None:
        resp = client.post("/api/v1/workflows/analyze", json={"workflow": SIMPLE_WORKFLOW})
        assert resp.status_code == 200
        data = resp.json()
        assert data["workflow_name"] == "test-wf"
        assert isinstance(data["aggregate_score"], float)
        assert data["risk_level"] in ("low", "medium", "high", "critical")
        assert len(data["sub_scores"]) == 6
        assert "mitigation_plan" in data

    def test_invalid_workflow(self) -> None:
        resp = client.post("/api/v1/workflows/analyze", json={"workflow": {"bad": "data"}})
        assert resp.status_code == 422

    def test_risky_workflow(self) -> None:
        resp = client.post("/api/v1/workflows/analyze", json={"workflow": RISKY_WORKFLOW})
        assert resp.status_code == 200
        data = resp.json()
        assert data["aggregate_score"] > 0.0
        assert len(data["mitigation_plan"]["mitigations"]) > 0


class TestCheckStep:
    def test_known_operation(self) -> None:
        resp = client.post("/api/v1/workflows/check-step", json={"operation": "delete_record"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["operation"] == "delete_record"
        assert "base_risk_weight" in data
        assert "effect_type" in data

    def test_with_context(self) -> None:
        resp = client.post("/api/v1/workflows/check-step", json={
            "operation": "delete_record",
            "existing_operations": ["authenticate", "read_database"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "contextual_risk" in data

    def test_unknown_operation(self) -> None:
        resp = client.post("/api/v1/workflows/check-step", json={"operation": "fake_op"})
        assert resp.status_code == 422
        assert "unknown operation" in resp.json()["detail"].lower()


class TestGetReport:
    def test_nonexistent_workflow(self) -> None:
        resp = client.get("/api/v1/workflows/nonexistent-id/report")
        assert resp.status_code == 422
        assert "not found" in resp.json()["detail"].lower()


class TestRecordOutcome:
    def test_empty_records(self) -> None:
        resp = client.post("/api/v1/executions/record", json={
            "workflow_id": "wf-1",
            "execution_id": "exec-1",
            "records": [],
        })
        assert resp.status_code == 422
        assert "empty" in resp.json()["detail"].lower()

    def test_invalid_outcome(self) -> None:
        resp = client.post("/api/v1/executions/record", json={
            "workflow_id": "wf-1",
            "execution_id": "exec-1",
            "records": [{"node_id": "n1", "operation": "read_file", "outcome": "success"}],
            "actual_outcome": "bogus",
        })
        assert resp.status_code == 422
        assert "invalid outcome" in resp.json()["detail"].lower()

    def test_nonexistent_workflow(self) -> None:
        resp = client.post("/api/v1/executions/record", json={
            "workflow_id": "nonexistent",
            "execution_id": "exec-1",
            "records": [{"node_id": "n1", "operation": "read_file", "outcome": "success"}],
        })
        assert resp.status_code == 422


class TestFindSimilar:
    def test_success(self) -> None:
        resp = client.post("/api/v1/workflows/find-similar", json={"workflow": SIMPLE_WORKFLOW})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query_workflow"] == "test-wf"
        assert data["similar_workflows"] == []
        assert "placeholder" in data["note"].lower() or "Layer 3" in data["note"]

    def test_custom_top_k(self) -> None:
        resp = client.post("/api/v1/workflows/find-similar", json={
            "workflow": SIMPLE_WORKFLOW,
            "top_k": 10,
        })
        assert resp.status_code == 200
        assert resp.json()["top_k"] == 10

    def test_invalid_workflow(self) -> None:
        resp = client.post("/api/v1/workflows/find-similar", json={"workflow": {"bad": True}})
        assert resp.status_code == 422


class TestOntology:
    def test_returns_operations(self) -> None:
        resp = client.get("/api/v1/ontology")
        assert resp.status_code == 200
        data = resp.json()
        assert "operations" in data
        assert "count" in data
        assert data["count"] == 20

    def test_operation_shape(self) -> None:
        resp = client.get("/api/v1/ontology")
        op = resp.json()["operations"][0]
        assert "name" in op
        assert "category" in op
        assert "base_risk_weight" in op
        assert "effect_type" in op
        assert "effect_targets" in op
