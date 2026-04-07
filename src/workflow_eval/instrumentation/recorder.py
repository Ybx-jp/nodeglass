"""WorkflowRecorder — persists instrumented executions to storage (NOD-34).

NOD-34 spec (Linear):
- instrumentation/outcome.py + recorder.py
- On workflow_context exit, persist WorkflowExecution to WorkflowRepository
- Include predicted risk from pre-execution scoring

AC:
- [x] After context exit, repository.get_execution(id) returns the complete record
- [x] DAG in execution matches the observed operations
- [x] predicted_risk is populated

Behavioral constraints:
- Stores workflow DAG + risk profile, then stores execution with predicted_risk
- Derives actual_outcome from individual operation records
"""

from __future__ import annotations

from workflow_eval.instrumentation.outcome import derive_outcome
from workflow_eval.storage.repository import WorkflowRepository

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from workflow_eval.instrumentation.sdk import WorkflowContext


class WorkflowRecorder:
    """Records completed workflow executions to a WorkflowRepository."""

    def __init__(self, repository: WorkflowRepository) -> None:
        self._repository = repository

    def record(self, context: WorkflowContext) -> str:
        """Persist the workflow and execution from a completed context.

        Returns the execution id.

        Steps:
        1. Score the final DAG to get predicted_risk
        2. Store the workflow (DAG + risk profile)
        3. Derive overall outcome from operation records
        4. Store the execution with predicted_risk and actual_outcome
        """
        # 1. Score for predicted risk
        risk_profile = context.get_current_risk()

        # 2. Store the workflow
        dag = context.get_dag()
        workflow_id = self._repository.store_workflow(dag, risk_profile)

        # 3. Build execution with predicted risk and derived outcome
        execution = context.get_execution()
        records = execution.records
        actual_outcome = derive_outcome(records)
        execution.predicted_risk = risk_profile.aggregate_score
        execution.actual_outcome = actual_outcome

        # 4. Persist
        self._repository.store_execution(workflow_id, execution)
        return execution.id
