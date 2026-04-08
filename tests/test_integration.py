"""Full integration test suite (NOD-42).

NOD-42 spec (Linear):
- tests/test_integration.py
- End-to-end: define YAML workflow → analyze via library → score →
  generate mitigations → instrument execution → record outcome →
  retrieve report
- Exercises the full pipeline in a single test flow

AC:
- [ ] pytest tests/test_integration.py -v passes
- [ ] Single test exercises declarative + runtime + storage + scoring + mitigation
- [ ] Verifies data consistency across all layers
"""

import pytest

from workflow_eval.dag.builder import DAGBuilder
from workflow_eval.dag.models import to_networkx
from workflow_eval.dag.schema import load_workflow
from workflow_eval.instrumentation.outcome import derive_outcome
from workflow_eval.instrumentation.recorder import WorkflowRecorder
from workflow_eval.instrumentation.sdk import track_operation, workflow_context
from workflow_eval.mitigation.engine import MitigationEngine
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.scoring.engine import RiskScoringEngine
from workflow_eval.similarity.structural import structural_similarity
from workflow_eval.storage.repository import SQLiteWorkflowRepository
from workflow_eval.types import ExecutionOutcome, RiskLevel, ScoringConfig


@pytest.fixture()
def repo() -> SQLiteWorkflowRepository:
    r = SQLiteWorkflowRepository(":memory:")
    yield r
    r.close()


@pytest.fixture()
def recorder(repo: SQLiteWorkflowRepository) -> WorkflowRecorder:
    return WorkflowRecorder(repo)


# ---------------------------------------------------------------------------
# AC: Single test exercises declarative + runtime + storage + scoring +
#     mitigation. Verifies data consistency across all layers.
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end: YAML → score → mitigate → instrument → persist → retrieve."""

    @pytest.mark.asyncio()
    async def test_yaml_to_report_round_trip(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        """Full pipeline through all 7 layers."""
        registry = get_default_registry()
        engine = RiskScoringEngine(ScoringConfig(), registry)

        # ── Layer 1: Load YAML workflow ──────────────────────────────
        dag = load_workflow("examples/sample_workflows/risky_delete_cascade.yaml")
        assert dag.name == "risky-delete-cascade"
        assert len(dag.nodes) == 7
        assert len(dag.edges) == 8

        # ── Layer 2: Convert to networkx ─────────────────────────────
        nx_dag = to_networkx(dag)
        assert nx_dag.number_of_nodes() == 7
        assert nx_dag.number_of_edges() == 8

        # ── Layer 3: Score ───────────────────────────────────────────
        profile = engine.score(nx_dag)
        assert profile.workflow_name == "risky-delete-cascade"
        assert profile.aggregate_score > 0.5  # known risky workflow
        assert profile.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert len(profile.sub_scores) == 6
        assert profile.node_count == 7
        assert profile.edge_count == 8

        # ── Layer 4: Mitigate ────────────────────────────────────────
        mitigation_engine = MitigationEngine()
        plan = mitigation_engine.generate_plan(profile, nx_dag, registry)
        assert plan.original_risk == pytest.approx(profile.aggregate_score)
        assert plan.residual_risk < plan.original_risk
        assert len(plan.mitigations) > 0
        # Must have required mitigations for irreversible nodes
        required = [m for m in plan.mitigations if m.priority.value == "required"]
        assert len(required) > 0

        # ── Layer 5: Store declarative workflow ──────────────────────
        workflow_id = repo.store_workflow(dag, profile)
        got_dag, got_profile = repo.get_workflow(workflow_id)
        assert got_dag == dag
        assert got_profile.aggregate_score == pytest.approx(profile.aggregate_score)
        assert got_profile.risk_level == profile.risk_level

        # ── Layer 6: Instrument equivalent runtime execution ─────────
        async with workflow_context("risky-delete-cascade", recorder=recorder) as wf:
            async with wf.operation("invoke_api"):
                pass
            async with wf.operation("mutate_state"):
                pass
            async with wf.operation("execute_code"):
                pass
            async with wf.operation("delete_record"):
                pass
            async with wf.operation("delete_file"):
                pass
            async with wf.operation("destroy_resource"):
                pass
            async with wf.operation("send_notification"):
                pass
            runtime_risk = wf.get_current_risk()

        # Runtime DAG was auto-built
        runtime_dag = wf.get_dag()
        assert len(runtime_dag.nodes) == 7
        assert runtime_dag.name == "risky-delete-cascade"

        # Execution was recorded
        execution = wf.get_execution()
        assert len(execution.records) == 7
        assert all(r.outcome == ExecutionOutcome.SUCCESS for r in execution.records)

        # ── Layer 7: Retrieve from storage ───────────────────────────
        got_exec = repo.get_execution(execution.id)
        assert got_exec.id == execution.id
        assert got_exec.workflow_name == "risky-delete-cascade"
        assert got_exec.predicted_risk == pytest.approx(runtime_risk.aggregate_score)
        assert got_exec.actual_outcome == ExecutionOutcome.SUCCESS
        assert len(got_exec.records) == 7
        assert got_exec.dag.name == "risky-delete-cascade"

    @pytest.mark.asyncio()
    async def test_declarative_vs_runtime_consistency(self) -> None:
        """Declarative DAGBuilder and runtime instrumentation produce
        equivalent scoring for the same operation sequence."""
        ops = [
            ("auth", "authenticate"),
            ("read", "read_database"),
            ("delete", "delete_record"),
            ("notify", "send_email"),
        ]

        # Declarative path
        builder = DAGBuilder("consistency-check")
        builder.add_step(ops[0][0], ops[0][1])
        for nid, op in ops[1:]:
            builder.then(nid, op)
        decl_dag = builder.build()
        decl_profile = RiskScoringEngine(
            ScoringConfig(), get_default_registry(),
        ).score(to_networkx(decl_dag))

        # Runtime path
        async with workflow_context("consistency-check") as wf:
            for _, op in ops:
                async with wf.operation(op):
                    pass
            runtime_profile = wf.get_current_risk()

        # Scores must match
        assert runtime_profile.aggregate_score == pytest.approx(
            decl_profile.aggregate_score,
        )
        assert runtime_profile.risk_level == decl_profile.risk_level
        for r_sub, d_sub in zip(runtime_profile.sub_scores, decl_profile.sub_scores):
            assert r_sub.name == d_sub.name
            assert r_sub.score == pytest.approx(d_sub.score)

    @pytest.mark.asyncio()
    async def test_failure_pipeline(
        self, repo: SQLiteWorkflowRepository, recorder: WorkflowRecorder,
    ) -> None:
        """Failure during execution propagates through storage correctly."""

        @track_operation("invoke_api")
        async def flaky_api():
            raise ConnectionError("service down")

        async with workflow_context("failure-flow", recorder=recorder) as wf:
            async with wf.operation("authenticate"):
                pass
            with pytest.raises(ConnectionError):
                await flaky_api()
            async with wf.operation("send_email"):
                pass

        # Outcome derived as FAILURE
        execution = wf.get_execution()
        assert derive_outcome(execution.records) == ExecutionOutcome.FAILURE

        # Persisted with failure
        got = repo.get_execution(execution.id)
        assert got.actual_outcome == ExecutionOutcome.FAILURE
        assert len(got.records) == 3
        assert got.records[0].outcome == ExecutionOutcome.SUCCESS
        assert got.records[1].outcome == ExecutionOutcome.FAILURE
        assert got.records[1].error == "service down"
        assert got.records[2].outcome == ExecutionOutcome.SUCCESS

    @pytest.mark.asyncio()
    async def test_similarity_across_stored_workflows(
        self, repo: SQLiteWorkflowRepository,
    ) -> None:
        """Structural similarity works against stored workflows."""
        registry = get_default_registry()
        engine = RiskScoringEngine(ScoringConfig(), registry)

        # Store two workflows
        dag_a = (DAGBuilder("wf-a")
            .add_step("n1", "authenticate")
            .then("n2", "read_database")
            .then("n3", "delete_record")
            .build())
        dag_b = (DAGBuilder("wf-b")
            .add_step("n1", "authenticate")
            .then("n2", "read_database")
            .then("n3", "delete_record")
            .then("n4", "send_email")
            .build())
        profile_a = engine.score(to_networkx(dag_a))
        profile_b = engine.score(to_networkx(dag_b))
        repo.store_workflow(dag_a, profile_a)
        repo.store_workflow(dag_b, profile_b)

        # Query with a similar workflow
        query = (DAGBuilder("query")
            .add_step("n1", "authenticate")
            .then("n2", "read_database")
            .then("n3", "delete_record")
            .build())

        stored = repo.list_workflows()
        assert len(stored) == 2

        scores = [
            (name, structural_similarity(query, stored_dag))
            for _, stored_dag in stored
            for name in [stored_dag.name]
        ]
        # wf-a (identical structure) should score higher than wf-b
        score_a = next(s for n, s in scores if n == "wf-a")
        score_b = next(s for n, s in scores if n == "wf-b")
        assert score_a > score_b
        assert score_a == pytest.approx(1.0)
        assert score_b > 0.7  # still similar
