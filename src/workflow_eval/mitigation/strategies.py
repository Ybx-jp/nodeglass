"""Concrete mitigation strategy rules (NOD-26).

NOD-26 spec (Linear):
- src/workflow_eval/mitigation/models.py + strategies.py
- Strategy rules mapping risk patterns to concrete actions
- Each strategy is a function: (dag, registry, risk_profile) -> list[Mitigation]

AC:
- [ ] Each MitigationAction enum value is produced by at least one strategy
- [ ] Strategies are independent and composable (order doesn't matter)
- [ ] Each Mitigation has action, target_node_ids, reason, and priority
- [ ] Priority levels: required > recommended > optional

Behavioral constraints from description:
- Irreversible ops -> add_confirmation (required) + add_rollback (recommended)
- External ops -> sandbox_external (recommended)
- High fan-out nodes (>3 downstream) -> reduce_parallelism (optional)
- Credential access -> add_audit_log (required)
- User-facing ops -> require_authentication (recommended)
- High-risk external calls -> add_rate_limit (optional)
- Any failed/uncertain predecessor of irreversible -> add_retry (recommended)
"""

from __future__ import annotations

import networkx as nx

from workflow_eval.ontology.effect_types import EffectTarget, EffectType
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import (
    Mitigation,
    MitigationAction,
    MitigationPriority,
    RiskProfile,
)

# Threshold for "high-risk" external calls (strategy 6).
_HIGH_RISK_EXTERNAL_THRESHOLD = 0.5

# Fan-out degree above which reduce_parallelism is recommended (strategy 3).
_FAN_OUT_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Strategy 1: Irreversible ops → add_confirmation + add_rollback
# ---------------------------------------------------------------------------


class MitigateIrreversibleOps:
    """Irreversible operations require confirmation and should have rollback plans."""

    name = "mitigate_irreversible_ops"

    def __call__(
        self,
        dag: nx.DiGraph,
        registry: OperationRegistry,
        risk_profile: RiskProfile,
    ) -> list[Mitigation]:
        mitigations: list[Mitigation] = []
        for nid in dag.nodes:
            op = registry.get(dag.nodes[nid]["operation"])
            if op.effect_type == EffectType.IRREVERSIBLE:
                mitigations.append(Mitigation(
                    action=MitigationAction.ADD_CONFIRMATION,
                    priority=MitigationPriority.REQUIRED,
                    target_node_ids=(nid,),
                    reason=f"Node '{nid}' performs irreversible operation '{op.name}'; "
                           f"require explicit confirmation before execution",
                ))
                mitigations.append(Mitigation(
                    action=MitigationAction.ADD_ROLLBACK,
                    priority=MitigationPriority.RECOMMENDED,
                    target_node_ids=(nid,),
                    reason=f"Node '{nid}' performs irreversible operation '{op.name}'; "
                           f"add rollback plan for recovery",
                ))
        return mitigations


# ---------------------------------------------------------------------------
# Strategy 2: External ops → sandbox_external
# ---------------------------------------------------------------------------


class MitigateExternalOps:
    """External operations should be sandboxed to limit blast radius."""

    name = "mitigate_external_ops"

    def __call__(
        self,
        dag: nx.DiGraph,
        registry: OperationRegistry,
        risk_profile: RiskProfile,
    ) -> list[Mitigation]:
        mitigations: list[Mitigation] = []
        for nid in dag.nodes:
            op = registry.get(dag.nodes[nid]["operation"])
            if op.effect_type == EffectType.EXTERNAL:
                mitigations.append(Mitigation(
                    action=MitigationAction.SANDBOX_EXTERNAL,
                    priority=MitigationPriority.RECOMMENDED,
                    target_node_ids=(nid,),
                    reason=f"Node '{nid}' makes external call '{op.name}'; "
                           f"sandbox to contain side effects",
                ))
        return mitigations


# ---------------------------------------------------------------------------
# Strategy 3: High fan-out → reduce_parallelism
# ---------------------------------------------------------------------------


class MitigateHighFanOut:
    """Nodes with high fan-out (>3 downstream) risk cascading failures."""

    name = "mitigate_high_fan_out"

    def __call__(
        self,
        dag: nx.DiGraph,
        registry: OperationRegistry,
        risk_profile: RiskProfile,
    ) -> list[Mitigation]:
        mitigations: list[Mitigation] = []
        for nid in dag.nodes:
            out_degree = dag.out_degree(nid)
            if out_degree > _FAN_OUT_THRESHOLD:
                mitigations.append(Mitigation(
                    action=MitigationAction.REDUCE_PARALLELISM,
                    priority=MitigationPriority.OPTIONAL,
                    target_node_ids=(nid,),
                    reason=f"Node '{nid}' fans out to {out_degree} downstream nodes; "
                           f"consider sequential execution to limit blast radius",
                ))
        return mitigations


# ---------------------------------------------------------------------------
# Strategy 4: Credential access → add_audit_log
# ---------------------------------------------------------------------------


class MitigateCredentialAccess:
    """Operations touching credentials must be audit-logged."""

    name = "mitigate_credential_access"

    def __call__(
        self,
        dag: nx.DiGraph,
        registry: OperationRegistry,
        risk_profile: RiskProfile,
    ) -> list[Mitigation]:
        mitigations: list[Mitigation] = []
        for nid in dag.nodes:
            op = registry.get(dag.nodes[nid]["operation"])
            if EffectTarget.CREDENTIALS in op.effect_targets:
                mitigations.append(Mitigation(
                    action=MitigationAction.ADD_AUDIT_LOG,
                    priority=MitigationPriority.REQUIRED,
                    target_node_ids=(nid,),
                    reason=f"Node '{nid}' accesses credentials via '{op.name}'; "
                           f"audit log required for security compliance",
                ))
        return mitigations


# ---------------------------------------------------------------------------
# Strategy 5: User-facing ops → require_authentication
# ---------------------------------------------------------------------------


class MitigateUserFacingOps:
    """User-facing operations should require authentication."""

    name = "mitigate_user_facing_ops"

    def __call__(
        self,
        dag: nx.DiGraph,
        registry: OperationRegistry,
        risk_profile: RiskProfile,
    ) -> list[Mitigation]:
        mitigations: list[Mitigation] = []
        for nid in dag.nodes:
            op = registry.get(dag.nodes[nid]["operation"])
            if EffectTarget.USER_FACING in op.effect_targets:
                mitigations.append(Mitigation(
                    action=MitigationAction.REQUIRE_AUTHENTICATION,
                    priority=MitigationPriority.RECOMMENDED,
                    target_node_ids=(nid,),
                    reason=f"Node '{nid}' is user-facing via '{op.name}'; "
                           f"require authentication to prevent unauthorized access",
                ))
        return mitigations


# ---------------------------------------------------------------------------
# Strategy 6: High-risk external calls → add_rate_limit
# ---------------------------------------------------------------------------


class MitigateHighRiskExternal:
    """High-risk external calls should be rate-limited to prevent abuse."""

    name = "mitigate_high_risk_external"

    def __call__(
        self,
        dag: nx.DiGraph,
        registry: OperationRegistry,
        risk_profile: RiskProfile,
    ) -> list[Mitigation]:
        mitigations: list[Mitigation] = []
        for nid in dag.nodes:
            op = registry.get(dag.nodes[nid]["operation"])
            if (op.effect_type == EffectType.EXTERNAL
                    and op.base_risk_weight >= _HIGH_RISK_EXTERNAL_THRESHOLD):
                mitigations.append(Mitigation(
                    action=MitigationAction.ADD_RATE_LIMIT,
                    priority=MitigationPriority.OPTIONAL,
                    target_node_ids=(nid,),
                    reason=f"Node '{nid}' makes high-risk external call '{op.name}' "
                           f"(risk={op.base_risk_weight:.2f}); rate-limit to prevent abuse",
                ))
        return mitigations


# ---------------------------------------------------------------------------
# Strategy 7: Uncertain predecessors of irreversible → add_retry
# ---------------------------------------------------------------------------


class MitigateUncertainPredecessors:
    """Uncertain/failed predecessors of irreversible ops should have retry logic."""

    name = "mitigate_uncertain_predecessors"

    _UNCERTAIN_TYPES = frozenset({EffectType.EXTERNAL, EffectType.STATEFUL})

    def __call__(
        self,
        dag: nx.DiGraph,
        registry: OperationRegistry,
        risk_profile: RiskProfile,
    ) -> list[Mitigation]:
        mitigations: list[Mitigation] = []
        for nid in dag.nodes:
            op = registry.get(dag.nodes[nid]["operation"])
            if op.effect_type != EffectType.IRREVERSIBLE:
                continue
            uncertain_preds = tuple(
                pid for pid in dag.predecessors(nid)
                if registry.get(dag.nodes[pid]["operation"]).effect_type
                in self._UNCERTAIN_TYPES
            )
            if uncertain_preds:
                mitigations.append(Mitigation(
                    action=MitigationAction.ADD_RETRY,
                    priority=MitigationPriority.RECOMMENDED,
                    target_node_ids=uncertain_preds,
                    reason=f"Uncertain predecessors of irreversible node '{nid}'; "
                           f"add retry logic to prevent partial execution",
                ))
        return mitigations


# ---------------------------------------------------------------------------
# Default strategy set
# ---------------------------------------------------------------------------


def get_default_strategies() -> tuple[
    MitigateIrreversibleOps,
    MitigateExternalOps,
    MitigateHighFanOut,
    MitigateCredentialAccess,
    MitigateUserFacingOps,
    MitigateHighRiskExternal,
    MitigateUncertainPredecessors,
]:
    """Return all 7 default mitigation strategies."""
    return (
        MitigateIrreversibleOps(),
        MitigateExternalOps(),
        MitigateHighFanOut(),
        MitigateCredentialAccess(),
        MitigateUserFacingOps(),
        MitigateHighRiskExternal(),
        MitigateUncertainPredecessors(),
    )
