"""Outcome derivation for workflow executions (NOD-34).

Derives the overall workflow outcome from individual operation records.
"""

from __future__ import annotations

from workflow_eval.types import ExecutionOutcome, ExecutionRecord


def derive_outcome(records: tuple[ExecutionRecord, ...]) -> ExecutionOutcome | None:
    """Derive overall workflow outcome from individual operation records.

    - Any FAILURE → FAILURE
    - All SUCCESS → SUCCESS
    - All SKIPPED → SKIPPED
    - Empty records → None
    """
    if not records:
        return None
    outcomes = {r.outcome for r in records}
    if ExecutionOutcome.FAILURE in outcomes:
        return ExecutionOutcome.FAILURE
    if outcomes == {ExecutionOutcome.SKIPPED}:
        return ExecutionOutcome.SKIPPED
    return ExecutionOutcome.SUCCESS
