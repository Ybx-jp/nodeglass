"""Mitigation data models and strategy protocol (NOD-26).

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

from typing import Protocol, runtime_checkable

import networkx as nx

from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import Mitigation, RiskProfile


@runtime_checkable
class MitigationStrategy(Protocol):
    """Interface for a single mitigation strategy rule.

    Each strategy independently analyzes a scored workflow and returns
    zero or more Mitigation recommendations. Strategies are composable:
    the engine runs all strategies and merges results.
    """

    name: str

    def __call__(
        self,
        dag: nx.DiGraph[str],
        registry: OperationRegistry,
        risk_profile: RiskProfile,
    ) -> list[Mitigation]:
        """Analyze the DAG and return mitigation recommendations.

        Args:
            dag: networkx DiGraph with node/edge attributes.
            registry: operation registry for looking up effect types and risk weights.
            risk_profile: scored risk profile from the scoring engine.

        Returns:
            List of Mitigation recommendations (may be empty).
        """
        ...
