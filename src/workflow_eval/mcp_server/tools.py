"""MCP tool handlers (NOD-36).

NOD-36 spec (Linear):
- mcp_server/tools.py
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

from workflow_eval.dag.models import to_networkx
from workflow_eval.mitigation.engine import MitigationEngine
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.engine import RiskScoringEngine
from workflow_eval.similarity.structural import structural_similarity
from workflow_eval.storage.repository import SQLiteWorkflowRepository
from workflow_eval.types import (
    ExecutionOutcome,
    ExecutionRecord,
    ScoringConfig,
    WorkflowDAG,
    WorkflowExecution,
)


def _get_registry() -> OperationRegistry:
    return get_default_registry()


def _get_scoring_engine() -> RiskScoringEngine:
    return RiskScoringEngine(ScoringConfig(), _get_registry())


def _get_repository() -> SQLiteWorkflowRepository:
    return SQLiteWorkflowRepository("workflow_eval.db")


def _parse_workflow(workflow: dict[str, Any]) -> WorkflowDAG:
    """Parse a dict into a WorkflowDAG, raising ValueError on invalid input."""
    try:
        return WorkflowDAG.model_validate(workflow)
    except Exception as exc:
        raise ValueError(f"Invalid workflow definition: {exc}") from exc


def analyze_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """Analyze a workflow DAG and return its risk profile.

    Args:
        workflow: A workflow definition with 'name', 'nodes', and 'edges'.

    Returns:
        Risk profile with aggregate score, risk level, sub-scores,
        critical paths, chokepoints, and mitigation plan.
    """
    dag = _parse_workflow(workflow)
    nx_dag = to_networkx(dag)
    registry = _get_registry()
    engine = _get_scoring_engine()
    profile = engine.score(nx_dag)

    mitigation_engine = MitigationEngine()
    plan = mitigation_engine.generate_plan(profile, nx_dag, registry)

    return {
        "workflow_name": profile.workflow_name,
        "aggregate_score": profile.aggregate_score,
        "risk_level": profile.risk_level.value,
        "sub_scores": [
            {"name": s.name, "score": s.score, "weight": s.weight}
            for s in profile.sub_scores
        ],
        "node_count": profile.node_count,
        "edge_count": profile.edge_count,
        "critical_paths": [list(p) for p in profile.critical_paths],
        "chokepoints": list(profile.chokepoints),
        "mitigation_plan": {
            "original_risk": plan.original_risk,
            "residual_risk": plan.residual_risk,
            "mitigations": [
                {
                    "action": m.action.value,
                    "priority": m.priority.value,
                    "target_node_ids": list(m.target_node_ids),
                    "reason": m.reason,
                }
                for m in plan.mitigations
            ],
        },
    }


def check_step_risk(
    operation: str,
    existing_operations: list[str] | None = None,
) -> dict[str, Any]:
    """Check the risk of a single operation, optionally in context of existing ops.

    Args:
        operation: Operation name to check (e.g., 'delete_record').
        existing_operations: Optional list of preceding operations for context scoring.

    Returns:
        Operation risk details and contextual risk if existing_operations provided.
    """
    registry = _get_registry()
    try:
        op_def = registry.get(operation)
    except KeyError:
        raise ValueError(
            f"Unknown operation: {operation!r}. "
            f"Available: {sorted(op.name for op in registry.all())}"
        )

    result: dict[str, Any] = {
        "operation": op_def.name,
        "category": op_def.category,
        "base_risk_weight": op_def.base_risk_weight,
        "effect_type": op_def.effect_type.value,
        "effect_targets": sorted(t.value for t in op_def.effect_targets),
    }

    if existing_operations:
        from workflow_eval.dag.builder import DAGBuilder

        builder = DAGBuilder("step-check")
        for i, op_name in enumerate(existing_operations):
            if i == 0:
                builder.add_step(f"{op_name}_{i}", op_name)
            else:
                builder.then(f"{op_name}_{i}", op_name)
        builder.then(f"{operation}_{len(existing_operations)}", operation)
        dag = builder.build()
        nx_dag = to_networkx(dag)
        profile = _get_scoring_engine().score(nx_dag)
        result["contextual_risk"] = {
            "aggregate_score": profile.aggregate_score,
            "risk_level": profile.risk_level.value,
        }

    return result


def get_risk_report(workflow_id: str) -> dict[str, Any]:
    """Retrieve the risk report for a stored workflow.

    Args:
        workflow_id: The ID of a previously analyzed and stored workflow.

    Returns:
        Workflow DAG summary and risk profile.
    """
    repo = _get_repository()
    try:
        dag, risk_profile = repo.get_workflow(workflow_id)
    except KeyError:
        raise ValueError(f"Workflow not found: {workflow_id!r}")
    finally:
        repo.close()

    return {
        "workflow_id": workflow_id,
        "workflow_name": dag.name,
        "node_count": len(dag.nodes),
        "edge_count": len(dag.edges),
        "risk_profile": {
            "aggregate_score": risk_profile.aggregate_score,
            "risk_level": risk_profile.risk_level.value,
            "sub_scores": [
                {"name": s.name, "score": s.score, "weight": s.weight}
                for s in risk_profile.sub_scores
            ],
            "critical_paths": [list(p) for p in risk_profile.critical_paths],
            "chokepoints": list(risk_profile.chokepoints),
        },
    }


def record_outcome(
    workflow_id: str,
    execution_id: str,
    records: list[dict[str, Any]],
    predicted_risk: float | None = None,
    actual_outcome: str | None = None,
) -> dict[str, Any]:
    """Record the outcome of a workflow execution.

    Args:
        workflow_id: The ID of the workflow this execution belongs to.
        execution_id: Unique ID for this execution.
        records: List of operation outcome records, each with
            'node_id', 'operation', 'outcome', and optional 'error'/'duration_ms'.
        predicted_risk: Optional predicted risk score.
        actual_outcome: Optional overall outcome ('success', 'failure', 'skipped').

    Returns:
        Confirmation with execution details.
    """
    if not records:
        raise ValueError("records must not be empty")

    try:
        parsed_records = tuple(
            ExecutionRecord.model_validate(r) for r in records
        )
    except Exception as exc:
        raise ValueError(f"Invalid execution record: {exc}") from exc

    parsed_outcome = None
    if actual_outcome is not None:
        try:
            parsed_outcome = ExecutionOutcome(actual_outcome)
        except ValueError:
            raise ValueError(
                f"Invalid outcome: {actual_outcome!r}. "
                f"Must be one of: {[e.value for e in ExecutionOutcome]}"
            )

    repo = _get_repository()
    try:
        # Verify workflow exists
        dag, _ = repo.get_workflow(workflow_id)
    except KeyError:
        repo.close()
        raise ValueError(f"Workflow not found: {workflow_id!r}")

    execution = WorkflowExecution(
        id=execution_id,
        workflow_name=dag.name,
        dag=dag,
        records=parsed_records,
        predicted_risk=predicted_risk,
        actual_outcome=parsed_outcome,
    )
    repo.store_execution(workflow_id, execution)
    repo.close()

    return {
        "execution_id": execution_id,
        "workflow_id": workflow_id,
        "record_count": len(parsed_records),
        "actual_outcome": actual_outcome,
        "status": "recorded",
    }


def find_similar_workflows(
    workflow: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    """Find structurally similar workflows (MVP: structural similarity).

    Args:
        workflow: A workflow definition to compare against stored workflows.
        top_k: Number of similar workflows to return.

    Returns:
        Top-k similar workflows with similarity scores, plus Layer 3 note.
    """
    dag = _parse_workflow(workflow)
    repo = _get_repository()
    try:
        stored = repo.list_workflows()
    finally:
        repo.close()

    scored = []
    for wf_id, stored_dag in stored:
        score = structural_similarity(dag, stored_dag)
        scored.append((wf_id, stored_dag.name, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    top = scored[:top_k]

    return {
        "query_workflow": dag.name,
        "node_count": len(dag.nodes),
        "edge_count": len(dag.edges),
        "top_k": top_k,
        "similar_workflows": [
            {"workflow_id": wf_id, "name": name, "similarity": round(score, 4)}
            for wf_id, name, score in top
        ],
        "note": "MVP structural similarity (node count, edge count, operation histogram). "
        "Layer 3 will replace this with learned workflow embeddings.",
    }
