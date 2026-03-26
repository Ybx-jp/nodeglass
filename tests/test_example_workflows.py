"""Tests for example workflow files (NOD-15).

NOD-15 spec (Linear):
- examples/sample_workflows/ contains 3 structurally distinct workflows
- safe_read_pipeline.yaml — 3 read ops, linear chain, expected risk: low
- risky_delete_cascade.yaml — 6 nodes, fan-out deletes, expected risk: high/critical
- moderate_api_chain.json — 4 nodes, linear API chain, expected risk: medium

AC:
- [ ] All three files load via load_workflow() without error
- [ ] All three pass validate_dag() with no errors
- [ ] Structurally distinct: different node counts, edge patterns, operation mixes
- [ ] ~~When scored (Phase 3), produce distinct risk levels as noted~~ Deferred to NOD-25
"""

from pathlib import Path

import pytest

from workflow_eval.dag.schema import load_workflow
from workflow_eval.dag.validation import ValidationLevel, validate_dag
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.types import WorkflowDAG

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "sample_workflows"

REGISTRY = get_default_registry()

EXAMPLE_FILES = [
    "safe_read_pipeline.yaml",
    "risky_delete_cascade.yaml",
    "moderate_api_chain.json",
]


# ---------------------------------------------------------------------------
# AC: All three files load via load_workflow() without error
# ---------------------------------------------------------------------------


class TestExampleFilesLoad:
    @pytest.mark.parametrize("filename", EXAMPLE_FILES)
    def test_loads_without_error(self, filename: str) -> None:
        dag = load_workflow(EXAMPLES_DIR / filename)
        assert isinstance(dag, WorkflowDAG)
        assert dag.name  # non-empty name
        assert len(dag.nodes) > 0
        assert len(dag.edges) > 0


# ---------------------------------------------------------------------------
# AC: All three pass validate_dag() with no errors
# ---------------------------------------------------------------------------


class TestExampleFilesValidate:
    @pytest.mark.parametrize("filename", EXAMPLE_FILES)
    def test_no_validation_errors(self, filename: str) -> None:
        dag = load_workflow(EXAMPLES_DIR / filename)
        issues = validate_dag(dag, REGISTRY)
        errors = [i for i in issues if i.level == ValidationLevel.ERROR]
        assert errors == [], f"{filename} has validation errors: {errors}"

    @pytest.mark.parametrize("filename", EXAMPLE_FILES)
    def test_no_validation_warnings(self, filename: str) -> None:
        """Example files should also be warning-free (no cycles, no orphans)."""
        dag = load_workflow(EXAMPLES_DIR / filename)
        issues = validate_dag(dag, REGISTRY)
        assert issues == [], f"{filename} has validation issues: {issues}"


# ---------------------------------------------------------------------------
# AC: Structurally distinct — different node counts, edge patterns, op mixes
# ---------------------------------------------------------------------------


class TestStructuralDistinctness:
    @pytest.fixture()
    def dags(self) -> dict[str, WorkflowDAG]:
        return {f: load_workflow(EXAMPLES_DIR / f) for f in EXAMPLE_FILES}

    def test_different_node_counts(self, dags: dict[str, WorkflowDAG]) -> None:
        counts = {f: len(d.nodes) for f, d in dags.items()}
        assert len(set(counts.values())) == len(counts), (
            f"Node counts are not all distinct: {counts}"
        )

    def test_different_edge_counts(self, dags: dict[str, WorkflowDAG]) -> None:
        counts = {f: len(d.edges) for f, d in dags.items()}
        assert len(set(counts.values())) == len(counts), (
            f"Edge counts are not all distinct: {counts}"
        )

    def test_different_operation_mixes(self, dags: dict[str, WorkflowDAG]) -> None:
        """Each workflow should use a different set of operations."""
        op_sets = {
            f: frozenset(n.operation for n in d.nodes) for f, d in dags.items()
        }
        # No two workflows share the exact same operation set
        op_values = list(op_sets.values())
        for i in range(len(op_values)):
            for j in range(i + 1, len(op_values)):
                assert op_values[i] != op_values[j], (
                    f"Workflows share identical operation sets: {op_values[i]}"
                )

    def test_different_edge_patterns(self, dags: dict[str, WorkflowDAG]) -> None:
        """Each workflow should have a different topology fingerprint."""
        from collections import Counter

        def topology_fingerprint(dag: WorkflowDAG) -> tuple[int, int, int, int]:
            out_degree: Counter[str] = Counter()
            in_degree: Counter[str] = Counter()
            for edge in dag.edges:
                out_degree[edge.source_id] += 1
                in_degree[edge.target_id] += 1
            node_ids = {n.id for n in dag.nodes}
            max_out = max((out_degree.get(n, 0) for n in node_ids), default=0)
            max_in = max((in_degree.get(n, 0) for n in node_ids), default=0)
            leaves = sum(1 for n in node_ids if out_degree.get(n, 0) == 0)
            return (len(dag.nodes), max_out, max_in, leaves)

        fingerprints = {f: topology_fingerprint(d) for f, d in dags.items()}
        values = list(fingerprints.values())
        assert len(set(values)) == len(values), (
            f"Topology fingerprints are not all distinct: {fingerprints}"
        )


