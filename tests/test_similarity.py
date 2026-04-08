"""Tests for structural similarity (NOD-38).

AC:
- [x] Given two structurally similar DAGs, similarity score > 0.7
- [x] Structurally different DAGs score < 0.3
- [x] Returns placeholder note about Layer 3 future enhancement
"""

import pytest

from workflow_eval.dag.builder import DAGBuilder
from workflow_eval.similarity.structural import (
    _count_ratio,
    _jaccard_multiset,
    structural_similarity,
)
from workflow_eval.mcp_server.tools import find_similar_workflows

from collections import Counter


# ---------------------------------------------------------------------------
# Unit tests: _count_ratio
# ---------------------------------------------------------------------------


class TestCountRatio:
    def test_equal_values(self) -> None:
        assert _count_ratio(5, 5) == 1.0

    def test_zero_and_zero(self) -> None:
        assert _count_ratio(0, 0) == 1.0

    def test_one_zero(self) -> None:
        assert _count_ratio(5, 0) == 0.0

    def test_close_values(self) -> None:
        assert _count_ratio(4, 5) == pytest.approx(0.8)

    def test_distant_values(self) -> None:
        assert _count_ratio(1, 10) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Unit tests: _jaccard_multiset
# ---------------------------------------------------------------------------


class TestJaccardMultiset:
    def test_identical(self) -> None:
        a = Counter({"read_file": 2, "delete_record": 1})
        assert _jaccard_multiset(a, a) == 1.0

    def test_disjoint(self) -> None:
        a = Counter({"read_file": 1})
        b = Counter({"delete_record": 1})
        assert _jaccard_multiset(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        a = Counter({"read_file": 2, "delete_record": 1})
        b = Counter({"read_file": 1, "delete_record": 1, "invoke_api": 1})
        # intersection: min(2,1) + min(1,1) + min(0,1) = 1+1+0 = 2
        # union: max(2,1) + max(1,1) + max(0,1) = 2+1+1 = 4
        assert _jaccard_multiset(a, b) == pytest.approx(0.5)

    def test_both_empty(self) -> None:
        assert _jaccard_multiset(Counter(), Counter()) == 1.0


# ---------------------------------------------------------------------------
# AC: Similar DAGs score > 0.7
# ---------------------------------------------------------------------------


class TestSimilarDAGs:
    def test_identical_dags_score_1(self) -> None:
        dag = (DAGBuilder("wf")
            .add_step("n1", "read_file")
            .then("n2", "invoke_api")
            .then("n3", "send_email")
            .build())
        assert structural_similarity(dag, dag) == pytest.approx(1.0)

    def test_same_structure_different_names(self) -> None:
        """Same ops/edges, different node IDs and workflow names."""
        dag_a = (DAGBuilder("wf-a")
            .add_step("a1", "read_file")
            .then("a2", "invoke_api")
            .then("a3", "send_email")
            .build())
        dag_b = (DAGBuilder("wf-b")
            .add_step("b1", "read_file")
            .then("b2", "invoke_api")
            .then("b3", "send_email")
            .build())
        assert structural_similarity(dag_a, dag_b) == pytest.approx(1.0)

    def test_nearly_identical_high_score(self) -> None:
        """One extra node — should still be > 0.7."""
        dag_a = (DAGBuilder("wf-a")
            .add_step("n1", "authenticate")
            .then("n2", "read_database")
            .then("n3", "send_email")
            .build())
        dag_b = (DAGBuilder("wf-b")
            .add_step("n1", "authenticate")
            .then("n2", "read_database")
            .then("n3", "invoke_api")
            .then("n4", "send_email")
            .build())
        score = structural_similarity(dag_a, dag_b)
        assert score > 0.7


# ---------------------------------------------------------------------------
# AC: Different DAGs score < 0.3
# ---------------------------------------------------------------------------


class TestDifferentDAGs:
    def test_completely_different_ops(self) -> None:
        """No shared operations at all."""
        dag_a = (DAGBuilder("reader")
            .add_step("n1", "read_file")
            .build())
        dag_b = (DAGBuilder("deleter")
            .add_step("n1", "authenticate")
            .then("n2", "delete_record")
            .then("n3", "destroy_resource")
            .then("n4", "send_email")
            .build())
        score = structural_similarity(dag_a, dag_b)
        assert score < 0.3

    def test_very_different_sizes(self) -> None:
        """1 node vs 5 nodes with different ops."""
        dag_a = (DAGBuilder("tiny")
            .add_step("n1", "read_file")
            .build())
        builder_b = DAGBuilder("large")
        builder_b.add_step("n1", "authenticate")
        builder_b.then("n2", "read_database")
        builder_b.then("n3", "invoke_api")
        builder_b.then("n4", "delete_record")
        builder_b.then("n5", "destroy_resource")
        dag_b = builder_b.build()
        score = structural_similarity(dag_a, dag_b)
        assert score < 0.3


# ---------------------------------------------------------------------------
# AC: Returns placeholder note about Layer 3
# ---------------------------------------------------------------------------


class TestFindSimilarTool:
    def test_layer3_note(self) -> None:
        workflow = {
            "name": "test",
            "nodes": [{"id": "n1", "operation": "read_file", "params": {}}],
            "edges": [],
        }
        result = find_similar_workflows(workflow)
        assert "Layer 3" in result["note"]

    def test_empty_store_returns_empty(self) -> None:
        workflow = {
            "name": "test",
            "nodes": [{"id": "n1", "operation": "read_file", "params": {}}],
            "edges": [],
        }
        result = find_similar_workflows(workflow)
        assert result["similar_workflows"] == []
        assert result["query_workflow"] == "test"
        assert result["top_k"] == 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_node_identical(self) -> None:
        dag = DAGBuilder("wf").add_step("n1", "read_file").build()
        assert structural_similarity(dag, dag) == pytest.approx(1.0)

    def test_symmetry(self) -> None:
        dag_a = (DAGBuilder("a")
            .add_step("n1", "read_file")
            .then("n2", "invoke_api")
            .build())
        dag_b = (DAGBuilder("b")
            .add_step("n1", "authenticate")
            .then("n2", "delete_record")
            .then("n3", "send_email")
            .build())
        assert structural_similarity(dag_a, dag_b) == pytest.approx(
            structural_similarity(dag_b, dag_a),
        )
