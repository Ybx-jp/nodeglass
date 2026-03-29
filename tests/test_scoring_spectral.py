"""Tests for spectral (algebraic connectivity) scorer (NOD-22).

AC:
- [x] Single-node DAG -> score 0.0 (handle edge case)
- [x] Two-node DAG -> handled without error
- [x] Loosely connected DAG (two clusters with single bridge) scores higher than tightly connected
- [x] Complete-ish graph scores lower (more robust)
- [x] Uses scipy.sparse.linalg for eigenvalue computation
"""

import networkx as nx
import pytest

from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.protocols import Scorer
from workflow_eval.scoring.spectral import SpectralScorer
from workflow_eval.types import SubScore


@pytest.fixture()
def registry() -> OperationRegistry:
    return get_default_registry()


@pytest.fixture()
def scorer() -> SpectralScorer:
    return SpectralScorer()


def _make_graph(nodes: dict[str, str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Helper: nodes is {id: operation}, edges is [(src, tgt)]."""
    g = nx.DiGraph(name="test", metadata={})
    for nid, op in nodes.items():
        g.add_node(nid, operation=op, params={}, metadata={})
    for src, tgt in edges:
        g.add_edge(src, tgt, edge_type="control_flow")
    return g


# ---------------------------------------------------------------------------
# AC: Single-node DAG -> score 0.0
# ---------------------------------------------------------------------------


class TestSingleNode:
    def test_single_node_score_zero(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file"}, [])
        result = scorer.score(g, registry)
        assert result.score == 0.0
        assert result.details["fiedler_value"] == 0.0

    def test_empty_dag_score_zero(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = scorer.score(g, registry)
        assert result.score == 0.0
        assert result.details["fiedler_value"] == 0.0
        assert result.details["normalized"] == 0.0


# ---------------------------------------------------------------------------
# AC: Two-node DAG -> handled without error
# ---------------------------------------------------------------------------


class TestTwoNode:
    def test_two_node_connected_no_error(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        # Two connected nodes: fiedler = 2.0, normalized = 2/(2*ln(2)) = 1.44 > 1 -> score = 0.0
        g = _make_graph(
            {"a": "read_file", "b": "read_file"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.0, abs=0.01)
        assert result.details["fiedler_value"] == pytest.approx(2.0, abs=0.01)

    def test_two_node_disconnected(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        # Two disconnected nodes: fiedler = 0 -> score = 1.0 (maximally fragile)
        g = _make_graph(
            {"a": "read_file", "b": "read_file"},
            [],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(1.0, abs=0.01)
        assert result.details["fiedler_value"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# AC: Loosely connected > tightly connected
# ---------------------------------------------------------------------------


class TestLooselyVsTightly:
    def test_bridge_scores_higher_than_complete(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        # Bridge: two triangles (a-b-c, d-e-f) connected by single edge c->d
        # Undirected fiedler ≈ 0.44 -> score ≈ 0.88 (fragile)
        bridge = _make_graph(
            {n: "read_file" for n in ["a", "b", "c", "d", "e", "f"]},
            [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d"), ("d", "e"), ("d", "f"), ("e", "f")],
        )
        # Complete DAG: all edges in topological order
        # Undirected fiedler = 6.0 -> score = 0.0 (robust)
        complete = _make_graph(
            {n: "read_file" for n in ["a", "b", "c", "d", "e", "f"]},
            [
                ("a", "b"), ("a", "c"), ("a", "d"), ("a", "e"), ("a", "f"),
                ("b", "c"), ("b", "d"), ("b", "e"), ("b", "f"),
                ("c", "d"), ("c", "e"), ("c", "f"),
                ("d", "e"), ("d", "f"),
                ("e", "f"),
            ],
        )
        bridge_result = scorer.score(bridge, registry)
        complete_result = scorer.score(complete, registry)
        assert bridge_result.score > complete_result.score

    def test_chain_scores_higher_than_dense(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        # 6-node chain: fiedler ≈ 0.268 -> score ≈ 0.925 (fragile)
        chain = _make_graph(
            {f"n{i}": "read_file" for i in range(6)},
            [(f"n{i}", f"n{i+1}") for i in range(5)],
        )
        # Dense DAG with extra cross-edges
        dense = _make_graph(
            {f"n{i}": "read_file" for i in range(6)},
            [
                ("n0", "n1"), ("n0", "n2"), ("n0", "n3"),
                ("n1", "n2"), ("n1", "n3"), ("n1", "n4"),
                ("n2", "n3"), ("n2", "n4"), ("n2", "n5"),
                ("n3", "n4"), ("n3", "n5"),
                ("n4", "n5"),
            ],
        )
        chain_result = scorer.score(chain, registry)
        dense_result = scorer.score(dense, registry)
        assert chain_result.score > dense_result.score

    def test_bridge_exact_value(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        # Verify the bridge score is in the expected range
        bridge = _make_graph(
            {n: "read_file" for n in ["a", "b", "c", "d", "e", "f"]},
            [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d"), ("d", "e"), ("d", "f"), ("e", "f")],
        )
        result = scorer.score(bridge, registry)
        assert result.score == pytest.approx(0.878, abs=0.01)
        assert result.details["fiedler_value"] == pytest.approx(0.438, abs=0.01)


# ---------------------------------------------------------------------------
# AC: Complete-ish graph scores lower (more robust)
# ---------------------------------------------------------------------------


class TestCompleteGraph:
    def test_complete_dag_score_zero(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        # Complete DAG K_6: fiedler = 6, normalized > 1.0 -> score = 0.0
        g = _make_graph(
            {f"n{i}": "read_file" for i in range(6)},
            [
                (f"n{i}", f"n{j}")
                for i in range(6) for j in range(i + 1, 6)
            ],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.0, abs=0.01)

    def test_near_complete_low_score(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        # K_6 minus 2 edges (n0->n5, n1->n4): fiedler=4.0, normalized>1 clamped -> score=0.0
        all_edges = [(f"n{i}", f"n{j}") for i in range(6) for j in range(i + 1, 6)]
        edges = [e for e in all_edges if e not in [("n0", "n5"), ("n1", "n4")]]
        g = _make_graph(
            {f"n{i}": "read_file" for i in range(6)},
            edges,
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.0, abs=0.01)
        assert result.details["fiedler_value"] == pytest.approx(4.0, abs=0.01)

    def test_disconnected_graph_scores_one(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        # Completely disconnected: fiedler = 0 -> score = 1.0
        g = _make_graph(
            {f"n{i}": "read_file" for i in range(5)},
            [],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# AC: Uses scipy.sparse.linalg for eigenvalue computation
# ---------------------------------------------------------------------------


class TestScipySparse:
    def test_uses_sparse_for_large_graph(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        """Verify sparse path is used for n >= 3 (the _SPARSE_THRESHOLD)."""
        from unittest.mock import patch

        import scipy.sparse.linalg

        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        with patch.object(
            scipy.sparse.linalg, "eigsh", wraps=scipy.sparse.linalg.eigsh
        ) as mock_eigsh:
            scorer.score(g, registry)
            mock_eigsh.assert_called_once()

    def test_dense_fallback_for_two_nodes(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        """n=2 uses dense eigensolver (eigsh requires k < n)."""
        from unittest.mock import patch

        import scipy.sparse.linalg

        g = _make_graph(
            {"a": "read_file", "b": "read_file"},
            [("a", "b")],
        )
        with patch.object(
            scipy.sparse.linalg, "eigsh", wraps=scipy.sparse.linalg.eigsh
        ) as mock_eigsh:
            scorer.score(g, registry)
            mock_eigsh.assert_not_called()


# ---------------------------------------------------------------------------
# Details
# ---------------------------------------------------------------------------


class TestDetails:
    def test_details_contain_fiedler_and_normalized(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "read_file", "c": "read_file"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        # 3-node chain: fiedler=1.0, normalized=1/(2*ln(3))≈0.4551
        assert result.details["fiedler_value"] == pytest.approx(1.0, abs=0.001)
        assert result.details["normalized"] == pytest.approx(0.4551, abs=0.001)

    def test_normalized_capped_at_one(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        # Complete graph: fiedler/normalization > 1.0 -> capped
        g = _make_graph(
            {f"n{i}": "read_file" for i in range(6)},
            [(f"n{i}", f"n{j}") for i in range(6) for j in range(i + 1, 6)],
        )
        result = scorer.score(g, registry)
        assert result.details["normalized"] == 1.0


# ---------------------------------------------------------------------------
# Coverage gaps from review
# ---------------------------------------------------------------------------


class TestCoverageGaps:
    def test_disconnected_components_with_internal_edges(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        """Two disconnected triangles — bypasses the 'no edges' early return,
        exercises eigenvalue path where fiedler should be 0.0."""
        g = _make_graph(
            {n: "read_file" for n in ["a", "b", "c", "d", "e", "f"]},
            [("a", "b"), ("a", "c"), ("b", "c"), ("d", "e"), ("d", "f"), ("e", "f")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(1.0, abs=0.01)
        assert result.details["fiedler_value"] == pytest.approx(0.0, abs=0.001)

    def test_operation_agnostic(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        """Spectral scorer never calls registry.get() — identical topologies
        with different operations must produce identical scores."""
        topo_edges = [("a", "b"), ("b", "c"), ("a", "c")]
        g_read = _make_graph(
            {n: "read_file" for n in ["a", "b", "c"]}, topo_edges,
        )
        g_delete = _make_graph(
            {n: "delete_file" for n in ["a", "b", "c"]}, topo_edges,
        )
        r1 = scorer.score(g_read, registry)
        r2 = scorer.score(g_delete, registry)
        assert r1.score == pytest.approx(r2.score, abs=1e-10)
        assert r1.details["fiedler_value"] == pytest.approx(r2.details["fiedler_value"], abs=1e-10)
        assert r1.details["normalized"] == pytest.approx(r2.details["normalized"], abs=1e-10)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_satisfies_scorer_protocol(self) -> None:
        assert isinstance(SpectralScorer(), Scorer)

    def test_name_is_spectral(self) -> None:
        assert SpectralScorer().name == "spectral"

    def test_returns_subscore_with_correct_name(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file"}, [])
        result = scorer.score(g, registry)
        assert isinstance(result, SubScore)
        assert result.name == "spectral"
        assert result.weight == 0.0

    def test_score_in_range(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        # 6-node chain: fiedler=2-2*cos(π/6)≈0.268, score≈0.925
        g = _make_graph(
            {f"n{i}": "read_file" for i in range(6)},
            [(f"n{i}", f"n{i+1}") for i in range(5)],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.9252, abs=0.01)

    def test_flagged_nodes_empty(
        self, scorer: SpectralScorer, registry: OperationRegistry
    ) -> None:
        """Spectral scorer is a global graph property — no per-node flagging."""
        g = _make_graph(
            {f"n{i}": "read_file" for i in range(6)},
            [(f"n{i}", f"n{i+1}") for i in range(5)],
        )
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ()
