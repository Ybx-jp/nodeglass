"""Irreversibility depth scorer (NOD-20).

NOD-20 spec (Linear):
- src/workflow_eval/scoring/irreversibility.py
- For each irreversible node i:
    ancestors(i) = all nodes on any path from a root to i
    uncertain_ancestors(i) = {a in ancestors(i) : effect_type(a) in {external, stateful, irreversible}}
    irrev_risk(i) = (|uncertain_ancestors(i)| / max(|ancestors(i)|, 1)) * (depth(i) / max_dag_depth)
    SCORE = max(irrev_risk(i) for i in irreversible_nodes) or 0.0
- depth_ratio normalizes by max depth of irreversible nodes (not max DAG depth)
  to prevent trailing pure ops from diluting the score.
- Clamped to [0, 1]. No irreversible ops scores 0.

AC:
- [ ] No irreversible ops -> 0.0
- [ ] invoke_api -> delete_record -> high score (external ancestor before irreversible)
- [ ] read_state -> delete_record -> lower score (pure ancestor, not uncertain)
- [ ] read_file -> read_db -> invoke_api -> branch -> delete_record at depth 4 -> highest score
- [ ] Flagged nodes include all irreversible nodes with irrev_risk > 0.3
"""

from __future__ import annotations

import networkx as nx

from workflow_eval.ontology.effect_types import EffectType
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import SubScore

_UNCERTAIN_TYPES = frozenset({EffectType.EXTERNAL, EffectType.STATEFUL, EffectType.IRREVERSIBLE})
_FLAGGED_THRESHOLD = 0.3


def _compute_node_depths(dag: nx.DiGraph) -> dict[str, int]:
    """Compute the depth of each node in the DAG (edge count from nearest root).

    Depth = length of the longest path from any source node (in-degree 0)
    to this node, measured in edges. Root nodes have depth 0.

    Uses topological-sort DP: since topo order guarantees all predecessors are
    visited before their successors, each node's depth is simply
    max(predecessor depths) + 1. This runs in O(V+E).
    """
    # Topo order ensures every predecessor of n is already in `depths` when we reach n.
    depths: dict[str, int] = {}
    for n in nx.topological_sort(dag):
        # A node with no predecessors (root) gets depth 0.
        # Otherwise, it's one edge deeper than its deepest parent.
        pred_depths = [depths[p] for p in dag.predecessors(n)]
        depths[n] = (max(pred_depths) + 1) if pred_depths else 0
    return depths


class IrreversibilityScorer:
    """Scores risk from irreversible ops downstream of uncertain ancestors."""

    name: str = "irreversibility"

    def score(self, dag: nx.DiGraph, registry: OperationRegistry) -> SubScore:
        n = dag.number_of_nodes()
        if n <= 1:
            return SubScore(name=self.name, score=0.0, weight=0.0, details={"irrev_risks": {}})

        # Identify irreversible nodes
        irrev_nodes = [
            nid for nid in dag.nodes
            if registry.get(dag.nodes[nid]["operation"]).effect_type == EffectType.IRREVERSIBLE
        ]

        if not irrev_nodes:
            return SubScore(name=self.name, score=0.0, weight=0.0, details={"irrev_risks": {}})

        node_depths = _compute_node_depths(dag)

        # Normalize by deepest irreversible node, not deepest node overall.
        # This prevents trailing pure ops from diluting the score.
        max_irrev_depth = max(node_depths[nid] for nid in irrev_nodes)

        if max_irrev_depth == 0:
            return SubScore(name=self.name, score=0.0, weight=0.0, details={"irrev_risks": {}})

        # Per-node irrev_risk
        irrev_risks: dict[str, float] = {}
        for nid in irrev_nodes:
            ancestors = nx.ancestors(dag, nid)
            uncertain = {
                a for a in ancestors
                if registry.get(dag.nodes[a]["operation"]).effect_type in _UNCERTAIN_TYPES
            }
            uncertainty_ratio = len(uncertain) / max(len(ancestors), 1)
            depth_ratio = node_depths[nid] / max_irrev_depth
            irrev_risks[nid] = uncertainty_ratio * depth_ratio

        score_val = max(irrev_risks.values()) if irrev_risks else 0.0
        clamped = min(score_val, 1.0)

        flagged = tuple(nid for nid, risk in irrev_risks.items() if risk > _FLAGGED_THRESHOLD)

        return SubScore(
            name=self.name,
            score=clamped,
            weight=0.0,
            details={"irrev_risks": irrev_risks},
            flagged_nodes=flagged,
        )
