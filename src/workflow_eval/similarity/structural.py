"""Structural similarity for workflow DAGs (NOD-38).

NOD-38 spec (Linear):
- Structural similarity without embeddings
- Compare node count, edge count, operation type histogram (Jaccard similarity)
- Return top-k from stored workflows
- MVP — Layer 3 will replace with learned embeddings

AC:
- [ ] Given two structurally similar DAGs, similarity score > 0.7
- [ ] Structurally different DAGs score < 0.3
- [ ] Returns placeholder note about Layer 3 future enhancement

Behavioral constraints:
- Node count ratio: 1 - |n1-n2| / max(n1, n2)
- Edge count ratio: 1 - |e1-e2| / max(e1, e2)
- Operation histogram: multiset Jaccard (min-intersection / max-union)
- Final score: weighted average of all three components
"""

from __future__ import annotations

from collections import Counter

from workflow_eval.types import WorkflowDAG


def _count_ratio(a: int, b: int) -> float:
    """Similarity ratio for two counts. Returns 1.0 when equal, 0.0 when maximally different."""
    if a == 0 and b == 0:
        return 1.0
    return 1.0 - abs(a - b) / max(a, b)


def _jaccard_multiset(a: Counter, b: Counter) -> float:
    """Multiset Jaccard similarity: sum(min) / sum(max) over all keys."""
    all_keys = set(a) | set(b)
    if not all_keys:
        return 1.0
    intersection = sum(min(a[k], b[k]) for k in all_keys)
    union = sum(max(a[k], b[k]) for k in all_keys)
    return intersection / union if union > 0 else 1.0


def structural_similarity(dag_a: WorkflowDAG, dag_b: WorkflowDAG) -> float:
    """Compute structural similarity between two workflow DAGs.

    Returns a score in [0.0, 1.0] based on:
    - Node count ratio (weight 0.25)
    - Edge count ratio (weight 0.25)
    - Operation type histogram Jaccard similarity (weight 0.50)
    """
    node_sim = _count_ratio(len(dag_a.nodes), len(dag_b.nodes))
    edge_sim = _count_ratio(len(dag_a.edges), len(dag_b.edges))

    ops_a = Counter(n.operation for n in dag_a.nodes)
    ops_b = Counter(n.operation for n in dag_b.nodes)
    ops_sim = _jaccard_multiset(ops_a, ops_b)

    return 0.25 * node_sim + 0.25 * edge_sim + 0.50 * ops_sim
