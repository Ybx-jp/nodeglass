"""Fan-out (blast radius) scorer (NOD-18).

NOD-18 spec (Linear):
- src/workflow_eval/scoring/fan_out.py
- Blast radius scorer. For each node v:
    fan_out(v) = |reachable(v)| / |V|
    weighted_fan_out(v) = fan_out(v) * risk_weight(op(v))
    SCORE = max(weighted_fan_out(v) for all v)
- Reports top-5 nodes by weighted fan-out in `details`
- Flags nodes with weighted_fan_out > 0.5 in `flagged_nodes`

AC:
- [ ] Single-node DAG -> score 0.0
- [ ] Linear A->B->C with root risk_weight 0.5 -> (2/3)*0.5 = ~0.33
- [ ] Star topology: root (risk 0.8) fanning to 5 leaves -> high score
- [ ] `details` contains top-5 nodes ranked by weighted fan-out
- [ ] `flagged_nodes` lists nodes exceeding threshold
"""

from __future__ import annotations

import networkx as nx

from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import SubScore

FLAGGED_THRESHOLD = 0.5


class FanOutScorer:
    """Blast radius scorer — measures how much of the DAG each node can reach."""

    name: str = "fan_out"

    def score(self, dag: nx.DiGraph, registry: OperationRegistry) -> SubScore:
        n = dag.number_of_nodes()
        if n == 0:
            return SubScore(name=self.name, score=0.0, weight=0.0)

        node_scores: dict[str, float] = {}
        for node_id in dag.nodes:
            descendants = nx.descendants(dag, node_id)
            fan_out = len(descendants) / n
            op_name = dag.nodes[node_id]["operation"]
            risk_weight = registry.get(op_name).base_risk_weight
            node_scores[node_id] = fan_out * risk_weight

        ranked = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        top_5 = ranked[:5]
        max_score = ranked[0][1] if ranked else 0.0
        flagged = tuple(nid for nid, val in ranked if val > FLAGGED_THRESHOLD)

        return SubScore(
            name=self.name,
            score=min(max_score, 1.0),
            weight=0.0,
            details={
                "top_nodes": [
                    {"node_id": nid, "weighted_fan_out": round(val, 6)}
                    for nid, val in top_5
                ],
            },
            flagged_nodes=flagged,
        )
