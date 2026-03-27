"""Tests for Scorer protocol (NOD-17).

AC:
- [x] Protocol defined with `name` attribute and `score()` method
- [x] Type-checkable with mypy (`isinstance` runtime check via `runtime_checkable`)
- [x] A minimal stub scorer implementing the protocol passes type checking
- [x] Returns `SubScore` with name, value [0,1], details dict, and flagged_nodes list
"""

import networkx as nx
import pytest

from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.scoring.protocols import Scorer
from workflow_eval.types import SubScore


# ---------------------------------------------------------------------------
# Stub scorer for testing
# ---------------------------------------------------------------------------


class StubScorer:
    """Minimal scorer that satisfies the Scorer protocol."""

    name: str = "stub"

    def score(self, dag: nx.DiGraph, registry: OperationRegistry) -> SubScore:
        return SubScore(
            name=self.name,
            score=0.5,
            weight=1.0,
            details={"reason": "stub"},
            flagged_nodes=("a",),
        )


class NotAScorer:
    """Does not satisfy the Scorer protocol — missing score() method."""

    name: str = "bad"


# ---------------------------------------------------------------------------
# AC: Protocol defined with `name` attribute and `score()` method
# ---------------------------------------------------------------------------


class TestProtocolDefinition:
    def test_protocol_has_name(self) -> None:
        assert "name" in Scorer.__protocol_attrs__

    def test_protocol_has_score_method(self) -> None:
        assert hasattr(Scorer, "score")
        assert callable(getattr(Scorer, "score", None))

    def test_protocol_is_runtime_checkable(self) -> None:
        assert hasattr(Scorer, "__protocol_attrs__") or hasattr(
            Scorer, "__abstractmethods__"
        )


# ---------------------------------------------------------------------------
# AC: Type-checkable with mypy (isinstance runtime check via runtime_checkable)
# ---------------------------------------------------------------------------


class TestRuntimeCheckable:
    def test_stub_scorer_is_instance(self) -> None:
        scorer = StubScorer()
        assert isinstance(scorer, Scorer)

    def test_non_scorer_is_not_instance(self) -> None:
        bad = NotAScorer()
        assert not isinstance(bad, Scorer)

    def test_plain_object_is_not_instance(self) -> None:
        assert not isinstance("not a scorer", Scorer)

    def test_none_is_not_instance(self) -> None:
        assert not isinstance(None, Scorer)


# ---------------------------------------------------------------------------
# AC: A minimal stub scorer implementing the protocol passes type checking
# ---------------------------------------------------------------------------


class TestStubScorer:
    def test_stub_returns_subscore(self) -> None:
        scorer = StubScorer()
        registry = get_default_registry()
        g = nx.DiGraph(name="test", metadata={})
        g.add_node("a", operation="read_file", params={}, metadata={})
        result = scorer.score(g, registry)
        assert isinstance(result, SubScore)

    def test_stub_score_value_in_range(self) -> None:
        scorer = StubScorer()
        registry = get_default_registry()
        g = nx.DiGraph(name="test", metadata={})
        result = scorer.score(g, registry)
        assert 0.0 <= result.score <= 1.0

    def test_stub_name_matches(self) -> None:
        scorer = StubScorer()
        assert scorer.name == "stub"
        registry = get_default_registry()
        g = nx.DiGraph(name="test", metadata={})
        result = scorer.score(g, registry)
        assert result.name == "stub"


# ---------------------------------------------------------------------------
# AC: Returns SubScore with name, value [0,1], details dict, flagged_nodes list
# ---------------------------------------------------------------------------


class TestSubScoreFields:
    def test_subscore_has_name(self) -> None:
        ss = SubScore(name="test", score=0.5, weight=1.0)
        assert ss.name == "test"

    def test_subscore_has_score_in_range(self) -> None:
        ss = SubScore(name="test", score=0.0, weight=1.0)
        assert ss.score == 0.0
        ss_max = SubScore(name="test", score=1.0, weight=1.0)
        assert ss_max.score == 1.0

    def test_subscore_rejects_out_of_range(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SubScore(name="test", score=1.1, weight=1.0)
        with pytest.raises(ValidationError):
            SubScore(name="test", score=-0.1, weight=1.0)

    def test_subscore_has_details_dict(self) -> None:
        ss = SubScore(name="test", score=0.5, weight=1.0, details={"key": "val"})
        assert ss.details == {"key": "val"}

    def test_subscore_details_defaults_to_empty(self) -> None:
        ss = SubScore(name="test", score=0.5, weight=1.0)
        assert ss.details == {}

    def test_subscore_has_flagged_nodes(self) -> None:
        ss = SubScore(
            name="test", score=0.5, weight=1.0, flagged_nodes=("a", "b")
        )
        assert ss.flagged_nodes == ("a", "b")

    def test_subscore_flagged_nodes_defaults_to_empty(self) -> None:
        ss = SubScore(name="test", score=0.5, weight=1.0)
        assert ss.flagged_nodes == ()

    def test_subscore_is_frozen(self) -> None:
        from pydantic import ValidationError

        ss = SubScore(name="test", score=0.5, weight=1.0)
        with pytest.raises(ValidationError, match="frozen"):
            ss.score = 0.9  # type: ignore[misc]

    def test_subscore_json_round_trip(self) -> None:
        ss = SubScore(
            name="fan_out",
            score=0.75,
            weight=0.15,
            details={"top_nodes": ["a", "b"]},
            flagged_nodes=("a",),
        )
        restored = SubScore.model_validate_json(ss.model_dump_json())
        assert restored == ss
