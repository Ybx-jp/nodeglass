"""RiskScoringEngine — orchestrates all scorers (NOD-24).

NOD-24 spec (Linear):
- RiskScoringEngine(config, registry) with score(dag) -> RiskProfile
- Runs all 6 scorers, aggregates, identifies critical paths and chokepoints

AC:
- [ ] Invalid weights (sum != 1.0) raise ValueError
- [ ] Engine returns RiskProfile with all 6 sub-scores + aggregate
- [ ] Risk level thresholds correct: 0.24 -> low, 0.25 -> medium, 0.74 -> high, 0.75 -> critical
- [ ] critical_paths populated (longest risk-weighted paths)
- [ ] chokepoints populated (high centrality + high risk nodes)
"""

from __future__ import annotations

import networkx as nx

from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.aggregator import aggregate, apply_weights, classify_risk
from workflow_eval.scoring.centrality import CentralityScorer
from workflow_eval.scoring.chain_depth import ChainDepthScorer
from workflow_eval.scoring.compositional import CompositionalScorer
from workflow_eval.scoring.fan_out import FanOutScorer
from workflow_eval.scoring.irreversibility import IrreversibilityScorer
from workflow_eval.scoring.protocols import Scorer
from workflow_eval.scoring.spectral import SpectralScorer
from workflow_eval.types import RiskProfile, ScoringConfig, SubScore

# Chokepoint = centrality-flagged node whose operation risk weight >= this threshold.
_CHOKEPOINT_RISK_THRESHOLD = 0.25


class RiskScoringEngine:
    """Runs all six scorers, aggregates, and produces a RiskProfile."""

    def __init__(self, config: ScoringConfig, registry: OperationRegistry) -> None:
        self._config = config
        self._registry = registry
        self._scorers: tuple[Scorer, ...] = (
            FanOutScorer(),
            ChainDepthScorer(),
            IrreversibilityScorer(),
            CentralityScorer(),
            SpectralScorer(),
            CompositionalScorer(),
        )

    def score(self, dag: nx.DiGraph) -> RiskProfile:
        raw_scores = tuple(s.score(dag, self._registry) for s in self._scorers)
        weighted_scores = apply_weights(raw_scores, self._config)
        agg = aggregate(raw_scores, self._config)
        risk_level = classify_risk(agg)

        return RiskProfile(
            workflow_name=dag.graph.get("name", "unnamed"),
            aggregate_score=min(agg, 1.0),
            risk_level=risk_level,
            sub_scores=weighted_scores,
            node_count=dag.number_of_nodes(),
            edge_count=dag.number_of_edges(),
            critical_paths=self._find_critical_paths(dag),
            chokepoints=self._find_chokepoints(dag, raw_scores),
        )

    def _find_critical_paths(
        self, dag: nx.DiGraph,
    ) -> tuple[tuple[str, ...], ...]:
        """Find the longest risk-weighted path from source to sink via topo-sort DP."""
        if dag.number_of_nodes() == 0:
            return ()

        node_risk: dict[str, float] = {}
        for nid in dag.nodes:
            op = dag.nodes[nid]["operation"]
            node_risk[nid] = self._registry.get(op).base_risk_weight

        # DP: best (highest cumulative risk) path ending at each node.
        dp: dict[str, tuple[float, tuple[str, ...]]] = {}
        for node in nx.topological_sort(dag):
            preds = list(dag.predecessors(node))
            if not preds:
                dp[node] = (node_risk[node], (node,))
            else:
                best_pred = max(preds, key=lambda p: dp[p][0])
                prev_risk, prev_path = dp[best_pred]
                dp[node] = (prev_risk + node_risk[node], prev_path + (node,))

        sinks = [n for n in dag.nodes if dag.out_degree(n) == 0]
        if not sinks:
            return ()

        best_sink = max(sinks, key=lambda s: dp[s][0])
        return (dp[best_sink][1],)

    def _find_chokepoints(
        self, dag: nx.DiGraph, sub_scores: tuple[SubScore, ...],
    ) -> tuple[str, ...]:
        """Nodes with high centrality AND high operation risk weight."""
        centrality_result = next(
            (s for s in sub_scores if s.name == "centrality"), None,
        )
        if centrality_result is None:
            return ()

        chokepoints = []
        for nid in centrality_result.flagged_nodes:
            if nid in dag.nodes:
                op = dag.nodes[nid]["operation"]
                risk = self._registry.get(op).base_risk_weight
                if risk >= _CHOKEPOINT_RISK_THRESHOLD:
                    chokepoints.append(nid)
        return tuple(sorted(chokepoints))
