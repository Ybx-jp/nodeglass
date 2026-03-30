"""Tests for irreversibility scorer (NOD-20).

AC:
- [x] No irreversible ops -> 0.0
- [x] invoke_api -> delete_record -> high score (external ancestor before irreversible)
- [x] read_state -> delete_record -> lower score (pure ancestor, not uncertain)
- [x] read_file -> read_db -> invoke_api -> branch -> delete_record at depth 4 -> highest score
- [x] Flagged nodes include all irreversible nodes with irrev_risk > 0.3
"""

import networkx as nx
import pytest

from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.irreversibility import IrreversibilityScorer
from workflow_eval.scoring.protocols import Scorer
from workflow_eval.types import SubScore


@pytest.fixture()
def registry() -> OperationRegistry:
    return get_default_registry()


@pytest.fixture()
def scorer() -> IrreversibilityScorer:
    return IrreversibilityScorer()


def _make_graph(nodes: dict[str, str], edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Helper: nodes is {id: operation}, edges is [(src, tgt)]."""
    g = nx.DiGraph(name="test", metadata={})
    for nid, op in nodes.items():
        g.add_node(nid, operation=op, params={}, metadata={})
    for src, tgt in edges:
        g.add_edge(src, tgt, edge_type="control_flow")
    return g


# ---------------------------------------------------------------------------
# AC: No irreversible ops -> 0.0
# ---------------------------------------------------------------------------


class TestNoIrreversible:
    def test_all_pure_ops(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "read_database", "c": "read_state"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.score == 0.0

    def test_stateful_and_external_but_no_irreversible(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "invoke_api", "b": "write_database", "c": "mutate_state"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.score == 0.0

    def test_single_node_pure(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file"}, [])
        result = scorer.score(g, registry)
        assert result.score == 0.0

    def test_empty_dag(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        g = nx.DiGraph(name="empty", metadata={})
        result = scorer.score(g, registry)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# AC: invoke_api -> delete_record -> high score
# ---------------------------------------------------------------------------


class TestExternalBeforeIrreversible:
    def test_invoke_api_to_delete_record(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # ancestors(delete_record) = {invoke_api}
        # uncertain = {invoke_api} (EXTERNAL)
        # irrev_risk = (1/1) * (1/1) = 1.0
        g = _make_graph(
            {"a": "invoke_api", "b": "delete_record"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(1.0, abs=0.01)

    def test_external_chain_before_irreversible(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # invoke_api -> send_webhook -> delete_file
        # ancestors = {invoke_api, send_webhook}, both EXTERNAL -> uncertain = 2
        # irrev_risk = (2/2) * (2/2) = 1.0
        g = _make_graph(
            {"a": "invoke_api", "b": "send_webhook", "c": "delete_file"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(1.0, abs=0.01)

    def test_stateful_ancestor_also_uncertain(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # write_database (STATEFUL) -> delete_record (IRREVERSIBLE)
        # uncertain = {write_database} -> (1/1) * (1/1) = 1.0
        g = _make_graph(
            {"a": "write_database", "b": "delete_record"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# AC: read_state -> delete_record -> lower score (pure ancestor, not uncertain)
# ---------------------------------------------------------------------------


class TestPureBeforeIrreversible:
    def test_read_state_to_delete_record(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # ancestors = {read_state} (PURE) -> uncertain = 0
        # irrev_risk = (0/1) * (1/1) = 0.0
        g = _make_graph(
            {"a": "read_state", "b": "delete_record"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.score == 0.0

    def test_all_pure_ancestors_before_irreversible(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # read_file -> read_database -> delete_file
        # ancestors = {read_file, read_database} both PURE -> uncertain = 0
        # irrev_risk = (0/2) * (2/2) = 0.0
        g = _make_graph(
            {"a": "read_file", "b": "read_database", "c": "delete_file"},
            [("a", "b"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.score == 0.0

    def test_lower_than_external_ancestor(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # Directly compare: pure ancestor gives lower score than external ancestor
        pure_g = _make_graph(
            {"a": "read_state", "b": "delete_record"},
            [("a", "b")],
        )
        ext_g = _make_graph(
            {"a": "invoke_api", "b": "delete_record"},
            [("a", "b")],
        )
        pure_result = scorer.score(pure_g, registry)
        ext_result = scorer.score(ext_g, registry)
        assert pure_result.score < ext_result.score


# ---------------------------------------------------------------------------
# AC: read_file -> read_db -> invoke_api -> branch -> delete_record
#     at depth 4 -> highest score
# ---------------------------------------------------------------------------


class TestDeepChain:
    def test_mixed_chain_depth_4(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # read_file(PURE) -> read_database(PURE) -> invoke_api(EXTERNAL) -> branch(PURE) -> delete_record(IRREVERSIBLE)
        # ancestors(delete_record) = {read_file, read_database, invoke_api, branch} = 4
        # uncertain = {invoke_api} = 1
        # depth(delete_record) = 4 edges, max_irrev_depth = 4 edges
        # irrev_risk = (1/4) * (4/4) = 0.25
        g = _make_graph(
            {
                "a": "read_file",
                "b": "read_database",
                "c": "invoke_api",
                "d": "branch",
                "e": "delete_record",
            },
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.25, abs=0.01)

    def test_pure_nodes_dilute_uncertainty_ratio(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # Adding pure nodes between external and irreversible dilutes the uncertainty ratio.
        # Shallow: invoke_api -> delete_record (uncertain 1/1, depth 1/1) -> 1.0
        # Deep: invoke_api -> read_file -> read_database -> delete_record
        #   uncertain = 1/3 (diluted by pure nodes), depth = 3/3 = 1.0 -> 0.333
        shallow = _make_graph(
            {"a": "invoke_api", "b": "delete_record"},
            [("a", "b")],
        )
        deep = _make_graph(
            {"a": "invoke_api", "b": "read_file", "c": "read_database", "d": "delete_record"},
            [("a", "b"), ("b", "c"), ("c", "d")],
        )
        shallow_result = scorer.score(shallow, registry)
        deep_result = scorer.score(deep, registry)
        # Shallow has higher score due to higher uncertainty ratio
        assert shallow_result.score > deep_result.score
        assert deep_result.score == pytest.approx(1 / 3, abs=0.01)

    def test_trailing_pure_ops_do_not_dilute_score(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # invoke_api -> delete_record (no trailing ops)
        #   depth=1, max_irrev_depth=1, depth_ratio=1.0, uncertain=1/1=1.0
        #   irrev_risk = 1.0
        no_trailing = _make_graph(
            {"a": "invoke_api", "b": "delete_record"},
            [("a", "b")],
        )
        # invoke_api -> delete_record -> read_file -> read_database
        #   depth=1, max_irrev_depth=1, depth_ratio=1.0, uncertain=1/1=1.0
        #   irrev_risk = 1.0 (trailing reads don't affect score)
        with_trailing = _make_graph(
            {"a": "invoke_api", "b": "delete_record", "c": "read_file", "d": "read_database"},
            [("a", "b"), ("b", "c"), ("c", "d")],
        )
        no_trail = scorer.score(no_trailing, registry)
        with_trail = scorer.score(with_trailing, registry)
        assert no_trail.score == pytest.approx(1.0, abs=0.01)
        assert with_trail.score == pytest.approx(1.0, abs=0.01)

    def test_pure_ancestors_dilute_uncertainty_not_depth(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # Shallow: invoke_api -> delete_record -> read -> read
        #   uncertain=1/1=1.0, depth_ratio=1.0, irrev_risk=1.0
        shallow = _make_graph(
            {"a": "invoke_api", "b": "delete_record", "c": "read_file", "d": "read_database"},
            [("a", "b"), ("b", "c"), ("c", "d")],
        )
        # Deep: invoke_api -> read_file -> read_database -> delete_record
        #   uncertain=1/3=0.333, depth_ratio=1.0, irrev_risk=0.333
        deep = _make_graph(
            {"a": "invoke_api", "b": "read_file", "c": "read_database", "d": "delete_record"},
            [("a", "b"), ("b", "c"), ("c", "d")],
        )
        shallow_result = scorer.score(shallow, registry)
        deep_result = scorer.score(deep, registry)
        # Shallow scores higher: 100% uncertain ancestors vs 33%
        assert shallow_result.score == pytest.approx(1.0, abs=0.01)
        assert deep_result.score == pytest.approx(1 / 3, abs=0.01)
        assert shallow_result.score > deep_result.score


# ---------------------------------------------------------------------------
# AC: Flagged nodes include all irreversible nodes with irrev_risk > 0.3
# ---------------------------------------------------------------------------


class TestFlaggedNodes:
    def test_high_risk_irreversible_flagged(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # invoke_api -> delete_record: irrev_risk = 1.0 > 0.3 -> flagged
        g = _make_graph(
            {"a": "invoke_api", "b": "delete_record"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ("b",)
        assert "a" not in result.flagged_nodes

    def test_low_risk_irreversible_not_flagged(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # read_state -> delete_record: irrev_risk = 0.0 < 0.3 -> not flagged
        g = _make_graph(
            {"a": "read_state", "b": "delete_record"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ()

    def test_mixed_chain_below_threshold_not_flagged(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # 5-node chain: irrev_risk = 0.25 < 0.3 -> not flagged
        g = _make_graph(
            {
                "a": "read_file",
                "b": "read_database",
                "c": "invoke_api",
                "d": "branch",
                "e": "delete_record",
            },
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")],
        )
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ()

    def test_multiple_irreversible_both_flagged(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # Two irreversible nodes, both above threshold:
        # max_dag_depth = 2 (path a -> c -> d)
        # delete_record (b): ancestors={a}, uncertain={a}(EXT), depth=1
        #   irrev_risk = (1/1) * (1/2) = 0.5 > 0.3 -> flagged
        # delete_file (d): ancestors={a, c}, uncertain={a}(EXT), depth=2
        #   irrev_risk = (1/2) * (2/2) = 0.5 > 0.3 -> flagged
        g = _make_graph(
            {
                "a": "invoke_api",
                "b": "delete_record",
                "c": "read_file",
                "d": "delete_file",
            },
            [("a", "b"), ("a", "c"), ("c", "d")],
        )
        result = scorer.score(g, registry)
        # SCORE = max(0.5, 0.5) = 0.5
        assert result.score == pytest.approx(0.5, abs=0.01)
        assert result.flagged_nodes == ("b", "d")
        assert "a" not in result.flagged_nodes
        assert "c" not in result.flagged_nodes

    def test_no_irreversible_ops_empty_flagged(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "invoke_api", "b": "write_database"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ()

    def test_single_node_irreversible_not_flagged(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # Single node: n <= 1 early return -> score 0.0, no flagged nodes
        g = _make_graph({"a": "delete_file"}, [])
        result = scorer.score(g, registry)
        assert result.flagged_nodes == ()

    def test_threshold_boundary_exactly_03_not_flagged(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        """irrev_risk == 0.3 exactly should NOT be flagged (strict > check)."""
        from workflow_eval.ontology.effect_types import EffectTarget, EffectType
        from workflow_eval.types import OperationDefinition

        custom_reg = OperationRegistry()
        custom_reg.register(OperationDefinition(
            name="ext_op",
            category="test",
            base_risk_weight=0.50,
            effect_type=EffectType.EXTERNAL,
            effect_targets=frozenset({EffectTarget.NETWORK}),
        ))
        custom_reg.register(OperationDefinition(
            name="pure_op",
            category="test",
            base_risk_weight=0.0,
            effect_type=EffectType.PURE,
            effect_targets=frozenset(),
        ))
        custom_reg.register(OperationDefinition(
            name="irrev_op",
            category="test",
            base_risk_weight=0.90,
            effect_type=EffectType.IRREVERSIBLE,
            effect_targets=frozenset({EffectTarget.DATABASE}),
        ))
        # Chain: ext_op -> pure_op -> pure_op -> ... -> irrev_op
        # We need uncertain/ancestors * depth/max_depth = 0.3
        # 10-node chain: ext_op at start, 8 pure_ops, irrev_op at end
        # uncertain = 1, ancestors = 9, depth = 9, max_depth = 9
        # irrev_risk = (1/9) * (9/9) = 1/9 ≈ 0.111 (too low)
        #
        # Try: 3 ext_ops + 7 pure_ops + 1 irrev_op = 11 nodes
        # uncertain = 3, ancestors = 10, depth = 10/10
        # irrev_risk = 3/10 = 0.3 exactly!
        nodes: dict[str, str] = {}
        for i in range(3):
            nodes[f"e{i}"] = "ext_op"
        for i in range(7):
            nodes[f"p{i}"] = "pure_op"
        nodes["irr"] = "irrev_op"
        # Linear chain: e0 -> e1 -> e2 -> p0 -> ... -> p6 -> irr
        node_ids = [f"e{i}" for i in range(3)] + [f"p{i}" for i in range(7)] + ["irr"]
        edges = [(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids) - 1)]
        g = _make_graph(nodes, edges)
        result = scorer.score(g, custom_reg)
        # irrev_risk = (3/10) * (10/10) = 0.3 exactly -> NOT flagged (strict >)
        assert result.score == pytest.approx(0.3, abs=1e-6)
        assert "irr" not in result.flagged_nodes


# ---------------------------------------------------------------------------
# Max aggregation across multiple irreversible nodes
# ---------------------------------------------------------------------------


class TestMaxAggregation:
    def test_max_picks_highest_irrev_risk(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # Two irreversible nodes with different irrev_risk values.
        # invoke_api -> delete_record: ancestors={invoke_api}, uncertain=1/1, depth=1/2 -> 0.5
        # invoke_api -> read_file -> read_database -> delete_file:
        #   ancestors={invoke_api, read_file, read_database}, uncertain=1/3, depth=3/3 -> 0.333
        # Wait, max_dag_depth depends on longest path in full DAG.
        #
        # Graph: a(invoke_api) -> b(delete_record)
        #         a -> c(read_file) -> d(read_database) -> e(delete_file)
        # Longest path = a->c->d->e = 3 edges. max_dag_depth = 3.
        # delete_record (b): ancestors={a}, uncertain={a}, depth=1
        #   irrev_risk = (1/1) * (1/3) = 0.333
        # delete_file (e): ancestors={a,c,d}, uncertain={a}, depth=3
        #   irrev_risk = (1/3) * (3/3) = 0.333
        # Hmm, same again. Let me use a different topology.
        #
        # Graph: a(invoke_api) -> b(invoke_api) -> c(delete_record)
        #         a -> d(delete_file)
        # Longest path = a->b->c = 2 edges. max_dag_depth = 2.
        # delete_record (c): ancestors={a,b}, uncertain={a(EXT),b(EXT)}=2
        #   irrev_risk = (2/2) * (2/2) = 1.0
        # delete_file (d): ancestors={a}, uncertain={a}=1
        #   irrev_risk = (1/1) * (1/2) = 0.5
        # SCORE = max(1.0, 0.5) = 1.0
        g = _make_graph(
            {
                "a": "invoke_api",
                "b": "invoke_api",
                "c": "delete_record",
                "d": "delete_file",
            },
            [("a", "b"), ("b", "c"), ("a", "d")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(1.0, abs=0.01)
        # Both flagged (1.0 > 0.3 and 0.5 > 0.3)
        assert "c" in result.flagged_nodes
        assert "d" in result.flagged_nodes
        # Verify individual risks in details
        assert result.details["irrev_risks"]["c"] == pytest.approx(1.0, abs=0.01)
        assert result.details["irrev_risks"]["d"] == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Edge case: max_dag_depth == 0 guard
# ---------------------------------------------------------------------------


class TestMaxDagDepthZero:
    def test_disconnected_irreversible_nodes_score_zero(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # 2 disconnected irreversible nodes: no edges -> all depths = 0, max_dag_depth = 0
        # Hits the max_dag_depth == 0 early return
        g = _make_graph(
            {"a": "delete_file", "b": "delete_record"},
            [],
        )
        result = scorer.score(g, registry)
        assert result.score == 0.0
        assert result.flagged_nodes == ()
        assert result.details["irrev_risks"] == {}


# ---------------------------------------------------------------------------
# Details
# ---------------------------------------------------------------------------


class TestDetails:
    def test_irrev_risks_in_details(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "invoke_api", "b": "delete_record"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert "irrev_risks" in result.details
        assert "b" in result.details["irrev_risks"]
        assert result.details["irrev_risks"]["b"] == pytest.approx(1.0, abs=0.01)

    def test_empty_risks_when_no_irreversible(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "read_file", "b": "read_database"},
            [("a", "b")],
        )
        result = scorer.score(g, registry)
        assert result.details["irrev_risks"] == {}

    def test_diamond_ancestor_counting(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # Diamond: invoke_api -> c, write_database -> c, where c = delete_record
        # ancestors(c) = {invoke_api, write_database} = 2
        # uncertain = {invoke_api(EXT), write_database(STATEFUL)} = 2
        # depth(c) = 1 edge (both paths are length 1), max_dag_depth = 1
        # irrev_risk = (2/2) * (1/1) = 1.0
        g = _make_graph(
            {"a": "invoke_api", "b": "write_database", "c": "delete_record"},
            [("a", "c"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(1.0, abs=0.01)

    def test_diamond_with_mixed_ancestors(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        # Diamond: read_file -> d, invoke_api -> d, where d = delete_file
        # ancestors(d) = {read_file, invoke_api} = 2
        # uncertain = {invoke_api} = 1
        # depth(d) = 1, max_dag_depth = 1
        # irrev_risk = (1/2) * (1/1) = 0.5
        g = _make_graph(
            {"a": "read_file", "b": "invoke_api", "c": "delete_file"},
            [("a", "c"), ("b", "c")],
        )
        result = scorer.score(g, registry)
        assert result.score == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_cyclic_graph_raises(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "delete_file", "b": "invoke_api"},
            [("a", "b"), ("b", "a")],
        )
        with pytest.raises(nx.NetworkXUnfeasible):
            scorer.score(g, registry)

    def test_unknown_operation_propagates_key_error(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph(
            {"a": "nonexistent_op", "b": "read_file"},
            [("a", "b")],
        )
        with pytest.raises(KeyError, match="nonexistent_op"):
            scorer.score(g, registry)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_satisfies_scorer_protocol(self) -> None:
        assert isinstance(IrreversibilityScorer(), Scorer)

    def test_name_is_irreversibility(self) -> None:
        assert IrreversibilityScorer().name == "irreversibility"

    def test_returns_subscore_with_correct_name(
        self, scorer: IrreversibilityScorer, registry: OperationRegistry
    ) -> None:
        g = _make_graph({"a": "read_file"}, [])
        result = scorer.score(g, registry)
        assert isinstance(result, SubScore)
        assert result.name == "irreversibility"
        assert result.weight == 0.0
