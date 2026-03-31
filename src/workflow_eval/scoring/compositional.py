"""Edge-based compositional risk amplification scorer (NOD-23).

NOD-23 spec (Linear):
- src/workflow_eval/scoring/compositional.py
- C[op_a][op_b] default 1.0, range [1.0, 3.0]
- edge_risk(u,v) = risk_weight(op_u) * risk_weight(op_v) * C[op_u][op_v]
- SCORE = max(edge_risk) / 3.0
- Ship with ~10 non-default composition entries (see _DEFAULT_COMPOSITIONS)
- Matrix is configurable — users can override/extend

AC:
- [ ] read_credentials -> invoke_api (C=2.5) scores high
- [ ] read_file -> read_file (C=1.0, low weights) scores low
- [ ] Custom matrix entries override defaults
- [ ] No edges -> score 0.0
- [ ] Details include the highest-risk edge pair and its multiplier
"""

from __future__ import annotations

import fnmatch
from typing import Any

import networkx as nx

from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import SubScore

# Sparse composition matrix — unlisted pairs default to 1.0.
# Supports shell-style wildcards (e.g. "delete_*" matches "delete_file").
_DEFAULT_COMPOSITIONS: dict[tuple[str, str], float] = {
    ("read_credentials", "invoke_api"): 2.5,
    ("branch", "delete_*"): 2.0,
    ("read_file", "write_file"): 1.2,
    ("invoke_api", "mutate_state"): 1.8,
    ("authenticate", "execute_code"): 2.0,
    ("read_credentials", "execute_code"): 2.5,
    ("invoke_api", "delete_record"): 2.2,
    ("execute_code", "send_email"): 2.0,
    ("read_database", "invoke_api"): 1.5,
    ("authenticate", "destroy_resource"): 2.3,
}

# Theoretical max: risk_weight=1.0 * risk_weight=1.0 * C=3.0
_MAX_EDGE_RISK = 3.0


class CompositionalScorer:
    """Scores risk from dangerous operation sequences along DAG edges."""

    name: str = "compositional"

    def __init__(
        self, compositions: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self._compositions = dict(_DEFAULT_COMPOSITIONS)
        if compositions:
            self._compositions.update(compositions)

    def _get_multiplier(self, op_source: str, op_target: str) -> float:
        """Look up the composition multiplier for an (op_source, op_target) pair.

        Lookup order:
        1. Exact match — O(1) dict lookup, handles the common case where both
           sides are literal operation names (9 of the 10 default entries).
        2. Wildcard fallback — linear scan over entries using fnmatch. This
           covers patterns like ("branch", "delete_*") that match any delete op.
           Wildcards can appear on either side of the pair.
        3. Default 1.0 — neutral multiplier when no entry matches.

        Exact-first matters: ("invoke_api", "delete_record") has its own entry
        at 2.2 and must not fall through to ("branch", "delete_*") at 2.0.
        """
        # 1. Exact match (fast path — most lookups land here)
        exact = self._compositions.get((op_source, op_target))
        if exact is not None:
            return exact

        # 2. Wildcard scan (only reached for pattern entries like delete_*)
        for (src_pat, tgt_pat), mult in self._compositions.items():
            if fnmatch.fnmatch(op_source, src_pat) and fnmatch.fnmatch(op_target, tgt_pat):
                return mult

        # 3. No match — neutral multiplier
        return 1.0

    def score(self, dag: nx.DiGraph, registry: OperationRegistry) -> SubScore:
        if dag.number_of_edges() == 0:
            return SubScore(
                name=self.name,
                score=0.0,
                weight=0.0,
                details={
                    "highest_risk_edge": None,
                    "highest_risk_ops": None,
                    "composition_multiplier": 0.0,
                    "edge_risk": 0.0,
                },
            )

        best: dict[str, Any] | None = None

        for u, v in dag.edges():
            op_u = dag.nodes[u]["operation"]
            op_v = dag.nodes[v]["operation"]
            rw_u = registry.get(op_u).base_risk_weight
            rw_v = registry.get(op_v).base_risk_weight
            multiplier = self._get_multiplier(op_u, op_v)
            edge_risk = rw_u * rw_v * multiplier

            if best is None or edge_risk > best["edge_risk"]:
                best = {
                    "highest_risk_edge": (u, v),
                    "highest_risk_ops": (op_u, op_v),
                    "composition_multiplier": multiplier,
                    "edge_risk": edge_risk,
                }

        score_val = min(best["edge_risk"] / _MAX_EDGE_RISK, 1.0)  # type: ignore[index]

        return SubScore(
            name=self.name,
            score=score_val,
            weight=0.0,
            details=best,  # type: ignore[arg-type]
        )
