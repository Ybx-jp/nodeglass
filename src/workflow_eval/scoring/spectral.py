"""Algebraic connectivity (spectral) scorer (NOD-22).

NOD-22 spec (Linear):
- src/workflow_eval/scoring/spectral.py
- L = laplacian_matrix(G.to_undirected())
- fiedler_value = second_smallest_eigenvalue(L)
- SCORE = 1 - min(fiedler_value / (2 * ln(|V|)), 1.0)
- Low algebraic connectivity = loosely connected = fragile.
- Uses scipy sparse eigenvalue computation for efficiency.

AC:
- [ ] Single-node DAG -> score 0.0 (handle edge case)
- [ ] Two-node DAG -> handled without error
- [ ] Loosely connected DAG (two clusters with single bridge) scores higher than tightly connected
- [ ] Complete-ish graph scores lower (more robust)
- [ ] Uses scipy.sparse.linalg for eigenvalue computation
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import scipy.sparse.linalg

from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import SubScore

# eigsh requires k < n, so sparse path needs n >= 3
_SPARSE_THRESHOLD = 3


class SpectralScorer:
    """Scores graph fragility via algebraic connectivity of the undirected Laplacian."""

    name: str = "spectral"

    def score(self, dag: nx.DiGraph[str], registry: OperationRegistry) -> SubScore:
        n = dag.number_of_nodes()

        if n <= 1:
            return SubScore(
                name=self.name,
                score=0.0,
                weight=0.0,
                details={"fiedler_value": 0.0, "normalized": 0.0},
            )

        undirected = dag.to_undirected()

        # Disconnected graph (no edges): Laplacian is all zeros, fiedler = 0
        if undirected.number_of_edges() == 0:
            return SubScore(
                name=self.name,
                score=1.0,
                weight=0.0,
                details={"fiedler_value": 0.0, "normalized": 0.0},
            )

        laplacian = nx.laplacian_matrix(undirected).astype(float)

        if n < _SPARSE_THRESHOLD:
            # Dense fallback for tiny graphs (eigsh requires k < n)
            eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
            eigenvalues.sort()
            fiedler = float(eigenvalues[1])
        else:
            # Sparse eigenvalue computation (AC requirement)
            eigenvalues = scipy.sparse.linalg.eigsh(
                laplacian, k=2, which="SM", return_eigenvectors=False,
            )
            eigenvalues.sort()
            fiedler = float(eigenvalues[1])

        # Guard against floating-point noise producing tiny negatives
        fiedler = max(fiedler, 0.0)

        normalized = fiedler / (2.0 * math.log(n))
        clamped = min(normalized, 1.0)
        score_val = 1.0 - clamped

        return SubScore(
            name=self.name,
            score=score_val,
            weight=0.0,
            details={"fiedler_value": fiedler, "normalized": clamped},
        )
