"""Scorer protocol interface (NOD-17).

NOD-17 spec (Linear):
- src/workflow_eval/scoring/protocols.py
- Scorer protocol (typing.Protocol) with name attribute and score() method
- All six scorers implement this interface
- Engine discovers and runs them polymorphically

AC:
- [ ] Protocol defined with `name` attribute and `score()` method
- [ ] Type-checkable with mypy (`isinstance` runtime check via `runtime_checkable`)
- [ ] A minimal stub scorer implementing the protocol passes type checking
- [ ] Returns `SubScore` with name, value [0,1], details dict, and flagged_nodes list
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import networkx as nx

from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import SubScore


@runtime_checkable
class Scorer(Protocol):
    """Interface that all risk scorers must implement.

    Each scorer analyzes a DAG from a specific risk perspective
    (fan-out, chain depth, irreversibility, etc.) and returns
    a SubScore with a value in [0, 1].
    """

    name: str

    def score(self, dag: nx.DiGraph[str], registry: OperationRegistry) -> SubScore:
        """Score a workflow DAG from this scorer's perspective.

        Args:
            dag: networkx DiGraph with node/edge attributes from to_networkx().
            registry: operation registry for looking up risk weights and effect types.

        Returns:
            SubScore with name, score in [0,1], details dict, and flagged_nodes list.
        """
        ...
