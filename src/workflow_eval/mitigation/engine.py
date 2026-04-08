"""MitigationEngine — generates mitigation plans from risk profiles (NOD-27).

NOD-27 spec (Linear):
- src/workflow_eval/mitigation/engine.py
- MitigationEngine class with generate_plan(risk_profile, dag, registry) -> MitigationPlan
- Runs all registered strategies
- Deduplicates mitigations (same action + same target nodes)
- Sorts by priority (required first)
- Estimates residual risk: aggregate * (0.5 ^ count_of_required_mitigations)

AC:
- [ ] Low-risk DAG -> empty mitigations or optional-only
- [ ] risky_delete_cascade -> multiple required mitigations (add_confirmation on deletes, add_audit_log on auth)
- [ ] Residual risk < original aggregate risk
- [ ] No duplicate mitigations in output
- [ ] Mitigations sorted: required first, then recommended, then optional
"""

from __future__ import annotations

import networkx as nx

from workflow_eval.mitigation.models import MitigationStrategy
from workflow_eval.mitigation.strategies import get_default_strategies
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import (
    Mitigation,
    MitigationPlan,
    MitigationPriority,
    RiskProfile,
)

_PRIORITY_ORDER = {
    MitigationPriority.REQUIRED: 0,
    MitigationPriority.RECOMMENDED: 1,
    MitigationPriority.OPTIONAL: 2,
}

# Each required mitigation halves the residual risk (heuristic).
_REQUIRED_DECAY = 0.5


class MitigationEngine:
    """Runs all mitigation strategies and produces a deduplicated, sorted plan."""

    def __init__(
        self,
        strategies: tuple[MitigationStrategy, ...] | None = None,
    ) -> None:
        self._strategies = strategies if strategies is not None else get_default_strategies()

    def generate_plan(
        self,
        risk_profile: RiskProfile,
        dag: nx.DiGraph[str],
        registry: OperationRegistry,
    ) -> MitigationPlan:
        """Run all strategies, deduplicate, sort, and estimate residual risk."""
        # 1. Collect mitigations from all strategies
        all_mitigations: list[Mitigation] = []
        for strategy in self._strategies:
            all_mitigations.extend(strategy(dag, registry, risk_profile))

        # 2. Deduplicate: same action + same set of target nodes
        seen: set[tuple[str, frozenset[str]]] = set()
        unique: list[Mitigation] = []
        for m in all_mitigations:
            key = (m.action, frozenset(m.target_node_ids))
            if key not in seen:
                seen.add(key)
                unique.append(m)

        # 3. Sort by priority: required first, then recommended, then optional
        unique.sort(key=lambda m: _PRIORITY_ORDER[m.priority])

        # 4. Estimate residual risk
        required_count = sum(
            1 for m in unique if m.priority == MitigationPriority.REQUIRED
        )
        residual = risk_profile.aggregate_score * (_REQUIRED_DECAY ** required_count)

        return MitigationPlan(
            mitigations=tuple(unique),
            original_risk=risk_profile.aggregate_score,
            residual_risk=min(residual, 1.0),
        )
