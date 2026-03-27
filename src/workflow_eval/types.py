"""Core Pydantic models shared across the framework."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from workflow_eval.ontology.effect_types import EffectTarget, EffectType


# ---------------------------------------------------------------------------
# Layer 1: Operation ontology models
# ---------------------------------------------------------------------------


class OperationDefinition(BaseModel):
    """A single agent primitive in the ontology."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    category: str
    base_risk_weight: float = Field(ge=0.0, le=1.0)
    effect_type: EffectType
    effect_targets: frozenset[EffectTarget]
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Layer 2: DAG models
# ---------------------------------------------------------------------------


class EdgeType(StrEnum):
    """Relationship between two DAG nodes."""

    CONTROL_FLOW = "control_flow"
    DATA_FLOW = "data_flow"
    CONDITIONAL = "conditional"


class DAGNode(BaseModel):
    """A single step in a workflow DAG."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    operation: str  # References OperationDefinition.name
    params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DAGEdge(BaseModel):
    """Directed edge between two DAG nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.CONTROL_FLOW
    condition: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowDAG(BaseModel):
    """Complete workflow graph — the primary input to scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    nodes: tuple[DAGNode, ...]
    edges: tuple[DAGEdge, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_node_ids(self) -> WorkflowDAG:
        ids = [n.id for n in self.nodes]
        dupes = {x for x in ids if ids.count(x) > 1}
        if dupes:
            raise ValueError(f"Duplicate node IDs: {dupes}")
        return self


# ---------------------------------------------------------------------------
# Scoring models
# ---------------------------------------------------------------------------


class SubScore(BaseModel):
    """Result from a single scorer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)
    flagged_nodes: tuple[str, ...] = ()


class RiskLevel(StrEnum):
    """Discrete risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScoringConfig(BaseModel):
    """Weights for the six scorers. Must sum to 1.0."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fan_out: float = 0.15
    chain_depth: float = 0.20
    irreversibility: float = 0.25
    centrality: float = 0.15
    spectral: float = 0.10
    compositional: float = 0.15

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ScoringConfig:
        total = (
            self.fan_out + self.chain_depth + self.irreversibility
            + self.centrality + self.spectral + self.compositional
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scorer weights must sum to 1.0, got {total:.6f}")
        return self


class RiskProfile(BaseModel):
    """Complete risk assessment for a workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_name: str
    aggregate_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    sub_scores: tuple[SubScore, ...]
    node_count: int
    edge_count: int


# ---------------------------------------------------------------------------
# Mitigation models
# ---------------------------------------------------------------------------


class MitigationAction(StrEnum):
    """Available mitigation actions."""

    ADD_CONFIRMATION = "add_confirmation"
    ADD_ROLLBACK = "add_rollback"
    SANDBOX_EXTERNAL = "sandbox_external"
    REDUCE_PARALLELISM = "reduce_parallelism"
    ADD_AUDIT_LOG = "add_audit_log"
    REQUIRE_AUTHENTICATION = "require_authentication"


class MitigationPriority(StrEnum):
    """How urgent a mitigation recommendation is."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class Mitigation(BaseModel):
    """A single mitigation recommendation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: MitigationAction
    priority: MitigationPriority
    target_node_ids: tuple[str, ...]
    reason: str


class MitigationPlan(BaseModel):
    """Collection of mitigations with residual risk estimate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mitigations: tuple[Mitigation, ...]
    original_risk: float = Field(ge=0.0, le=1.0)
    residual_risk: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Execution / storage models
# ---------------------------------------------------------------------------


class ExecutionOutcome(StrEnum):
    """How a single operation completed."""

    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class ExecutionRecord(BaseModel):
    """Observed outcome for one operation execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    operation: str
    outcome: ExecutionOutcome
    error: str | None = None
    duration_ms: float | None = None


class WorkflowExecution(BaseModel):
    """Full execution trace — stored for middle loop learning."""

    model_config = ConfigDict(extra="forbid")

    id: str
    workflow_name: str
    dag: WorkflowDAG
    records: tuple[ExecutionRecord, ...]
    predicted_risk: float | None = None
    actual_outcome: ExecutionOutcome | None = None
