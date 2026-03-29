"""Betweenness centrality-weighted risk scorer (NOD-21).

NOD-21 spec (Linear):
- src/workflow_eval/scoring/centrality.py
- bc(v) = betweenness_centrality(v)  # networkx, normalized
- centrality_risk(v) = bc(v) * risk_weight(op(v))
- SCORE = mean(centrality_risk) * (1 + std(centrality_risk))
- Clamped to [0, 1]. Penalizes DAGs with concentrated risk at chokepoints.

AC:
- [ ] Uniform low-risk DAG -> low score
- [ ] DAG with single high-risk chokepoint (high betweenness + high risk weight) -> significantly higher
- [ ] Linear chain -> moderate (middle nodes have highest betweenness)
- [ ] Single-node DAG -> 0.0 (no betweenness)
- [ ] Flagged nodes = those with centrality_risk > mean + 1*std
"""

from __future__ import annotations

import networkx as nx

from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import SubScore


class CentralityScorer:
    """Scores risk concentration at DAG chokepoints via betweenness centrality."""

    name: str = "centrality"

    def score(self, dag: nx.DiGraph, registry: OperationRegistry) -> SubScore:
        n = dag.number_of_nodes()
        if n <= 1:
            return SubScore(name=self.name, score=0.0, weight=0.0, details={"centrality_risks": {}})

        # Normalized betweenness centrality for directed graph
        bc = nx.betweenness_centrality(dag, normalized=True)

        # centrality_risk(v) = bc(v) * risk_weight(op(v))
        centrality_risks: dict[str, float] = {}
        for nid in dag.nodes:
            risk_weight = registry.get(dag.nodes[nid]["operation"]).base_risk_weight
            centrality_risks[nid] = bc[nid] * risk_weight

        values = list(centrality_risks.values())
        mean_val = sum(values) / len(values)

        # Population standard deviation
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_val = variance ** 0.5

        raw_score = mean_val * (1 + std_val)
        clamped = min(raw_score, 1.0)

        # Flag nodes with centrality_risk > mean + 1*std
        threshold = mean_val + std_val
        flagged = tuple(nid for nid, risk in centrality_risks.items() if risk > threshold)

        return SubScore(
            name=self.name,
            score=clamped,
            weight=0.0,
            details={"centrality_risks": centrality_risks},
            flagged_nodes=flagged,
        )
