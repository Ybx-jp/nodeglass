"""Side-effect chain depth scorer (NOD-19).

NOD-19 spec (Linear):
- src/workflow_eval/scoring/chain_depth.py
- Longest path through nodes with effect_type in {stateful, external, irreversible}
- SCORE = side_effect_chain_length / max(max_dag_depth, 1)
- Clamped to [0, 1]. Pure-function-only DAG scores 0.

AC:
- [ ] Pure-only DAG (all `pure` effect_type) -> 0.0
- [ ] All-stateful chain of depth 5 in DAG of depth 5 -> 1.0
- [ ] Mixed DAG: 2 side-effect nodes in depth-4 DAG -> ~0.5
- [ ] Single-node DAG -> 0.0
- [ ] Details include the longest side-effect path (node IDs)
"""

from __future__ import annotations

import networkx as nx

from workflow_eval.ontology.effect_types import EffectType
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import SubScore

_SIDE_EFFECT_TYPES = frozenset({EffectType.STATEFUL, EffectType.EXTERNAL, EffectType.IRREVERSIBLE})


class ChainDepthScorer:
    """Scores how deep side-effect chains run relative to total DAG depth."""

    name: str = "chain_depth"

    def score(self, dag: nx.DiGraph[str], registry: OperationRegistry) -> SubScore:
        n = dag.number_of_nodes()
        if n <= 1:
            return SubScore(name=self.name, score=0.0, weight=0.0, details={"longest_side_effect_path": []})

        # Full DAG longest path (node count)
        full_path = nx.dag_longest_path(dag)
        max_dag_depth = len(full_path)

        # Build subgraph of side-effect nodes only
        side_effect_nodes = [
            node_id
            for node_id in dag.nodes
            if registry.get(dag.nodes[node_id]["operation"]).effect_type in _SIDE_EFFECT_TYPES
        ]

        if not side_effect_nodes:
            return SubScore(
                name=self.name,
                score=0.0,
                weight=0.0,
                details={"longest_side_effect_path": []},
            )

        subgraph = dag.subgraph(side_effect_nodes)
        se_path = nx.dag_longest_path(subgraph)
        se_chain_length = len(se_path)

        raw_score = se_chain_length / max(max_dag_depth, 1)
        clamped_score = min(raw_score, 1.0)

        return SubScore(
            name=self.name,
            score=clamped_score,
            weight=0.0,
            details={"longest_side_effect_path": list(se_path)},
            flagged_nodes=tuple(se_path),
        )
