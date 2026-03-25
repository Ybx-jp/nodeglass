"""Tests for DAGBuilder fluent API (NOD-12).

AC:
- [x] Builder produces valid WorkflowDAG
- [x] then() creates control_flow edge from previous step
- [x] parallel() fans out from current to multiple steps
- [x] join() converges parallel branches
- [x] build() on empty builder raises ValueError
"""

import pytest

from workflow_eval.dag.builder import DAGBuilder
from workflow_eval.types import EdgeType, WorkflowDAG


# ---------------------------------------------------------------------------
# build() basics
# ---------------------------------------------------------------------------


class TestBuild:
    def test_produces_workflow_dag(self) -> None:
        dag = DAGBuilder("test").add_step("a", "read_file").build()
        assert isinstance(dag, WorkflowDAG)

    def test_name_preserved(self) -> None:
        dag = DAGBuilder("my-name").add_step("a", "read_file").build()
        assert dag.name == "my-name"

    def test_metadata_preserved(self) -> None:
        dag = DAGBuilder("t", metadata={"v": 1}).add_step("a", "read_file").build()
        assert dag.metadata == {"v": 1}

    def test_empty_builder_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            DAGBuilder("empty").build()

    def test_single_node_no_edges(self) -> None:
        dag = DAGBuilder("t").add_step("a", "read_file").build()
        assert len(dag.nodes) == 1
        assert len(dag.edges) == 0

    def test_build_with_pending_parallel_heads(self) -> None:
        """build() without join() is valid — produces a DAG with leaf branches."""
        dag = (
            DAGBuilder("t")
            .add_step("root", "read_file")
            .parallel(["a", "b"], operation="send_email")
            .build()
        )
        assert len(dag.nodes) == 3
        assert len(dag.edges) == 2
        edges = {(e.source_id, e.target_id) for e in dag.edges}
        assert edges == {("root", "a"), ("root", "b")}


# ---------------------------------------------------------------------------
# add_step() and then()
# ---------------------------------------------------------------------------


class TestLinearChain:
    def test_then_creates_edge(self) -> None:
        dag = (
            DAGBuilder("t")
            .add_step("a", "read_file")
            .then("b", "write_file")
            .build()
        )
        assert len(dag.nodes) == 2
        assert len(dag.edges) == 1
        assert dag.edges[0].source_id == "a"
        assert dag.edges[0].target_id == "b"
        assert dag.edges[0].edge_type == EdgeType.CONTROL_FLOW

    def test_three_step_chain(self) -> None:
        dag = (
            DAGBuilder("t")
            .add_step("a", "read_file")
            .then("b", "write_database")
            .then("c", "send_email")
            .build()
        )
        assert len(dag.nodes) == 3
        assert len(dag.edges) == 2
        edges = {(e.source_id, e.target_id) for e in dag.edges}
        assert ("a", "b") in edges
        assert ("b", "c") in edges

    def test_then_without_predecessor_raises(self) -> None:
        with pytest.raises(ValueError, match="preceding"):
            DAGBuilder("t").then("a", "read_file")

    def test_add_step_with_cursor_creates_edge(self) -> None:
        """add_step() after another add_step() creates an edge (like then)."""
        dag = (
            DAGBuilder("t")
            .add_step("a", "read_file")
            .add_step("b", "write_file")
            .build()
        )
        assert len(dag.edges) == 1
        assert dag.edges[0].source_id == "a"
        assert dag.edges[0].target_id == "b"

    def test_params_preserved(self) -> None:
        dag = (
            DAGBuilder("t")
            .add_step("a", "read_file", params={"path": "/tmp/x"})
            .then("b", "write_database", params={"table": "t"})
            .build()
        )
        nodes = {n.id: n for n in dag.nodes}
        assert nodes["a"].params == {"path": "/tmp/x"}
        assert nodes["b"].params == {"table": "t"}

    def test_duplicate_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            DAGBuilder("t").add_step("a", "read_file").then("a", "write_file")

    def test_custom_edge_type(self) -> None:
        dag = (
            DAGBuilder("t")
            .add_step("a", "read_file")
            .then("b", "write_file", edge_type=EdgeType.DATA_FLOW)
            .build()
        )
        assert dag.edges[0].edge_type == EdgeType.DATA_FLOW


# ---------------------------------------------------------------------------
# parallel()
# ---------------------------------------------------------------------------


class TestParallel:
    def test_fans_out_from_cursor(self) -> None:
        dag = (
            DAGBuilder("t")
            .add_step("root", "authenticate")
            .parallel(["a", "b"], operation="delete_record")
            .join("end", "send_notification")
            .build()
        )
        edges_from_root = [e for e in dag.edges if e.source_id == "root"]
        assert len(edges_from_root) == 2
        targets = {e.target_id for e in edges_from_root}
        assert targets == {"a", "b"}

    def test_parallel_nodes_share_operation(self) -> None:
        dag = (
            DAGBuilder("t")
            .add_step("root", "read_file")
            .parallel(["a", "b", "c"], operation="write_database", params={"table": "t"})
            .join("end", "send_email")
            .build()
        )
        nodes = {n.id: n for n in dag.nodes}
        for nid in ("a", "b", "c"):
            assert nodes[nid].operation == "write_database"
            assert nodes[nid].params == {"table": "t"}

    def test_chained_parallel_accumulates(self) -> None:
        """Two parallel() calls from the same cursor create simultaneous branches."""
        dag = (
            DAGBuilder("t")
            .add_step("root", "authenticate")
            .parallel(["a", "b"], operation="delete_record")
            .parallel(["c", "d"], operation="write_file")
            .join("end", "send_notification")
            .build()
        )
        # All 4 branches fan from root
        edges_from_root = [e for e in dag.edges if e.source_id == "root"]
        assert len(edges_from_root) == 4
        targets = {e.target_id for e in edges_from_root}
        assert targets == {"a", "b", "c", "d"}

        # join converges all 4
        edges_to_end = [e for e in dag.edges if e.target_id == "end"]
        assert len(edges_to_end) == 4

    def test_parallel_without_cursor_raises(self) -> None:
        with pytest.raises(ValueError, match="preceding"):
            DAGBuilder("t").parallel(["a", "b"], operation="read_file")

    def test_parallel_custom_edge_type(self) -> None:
        dag = (
            DAGBuilder("t")
            .add_step("root", "read_file")
            .parallel(["a", "b"], operation="write_file", edge_type=EdgeType.DATA_FLOW)
            .join("end", "send_email")
            .build()
        )
        edges_from_root = [e for e in dag.edges if e.source_id == "root"]
        for e in edges_from_root:
            assert e.edge_type == EdgeType.DATA_FLOW

    def test_then_after_parallel_raises(self) -> None:
        """then() with pending parallel heads is ambiguous and should error."""
        builder = (
            DAGBuilder("t")
            .add_step("root", "read_file")
            .parallel(["a", "b"], operation="write_file")
        )
        with pytest.raises(ValueError, match="parallel"):
            builder.then("c", "send_email")

    def test_add_step_after_parallel_raises(self) -> None:
        """add_step() with pending parallel heads is also ambiguous."""
        builder = (
            DAGBuilder("t")
            .add_step("root", "read_file")
            .parallel(["a", "b"], operation="write_file")
        )
        with pytest.raises(ValueError, match="parallel"):
            builder.add_step("c", "send_email")

    def test_duplicate_in_parallel_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            (
                DAGBuilder("t")
                .add_step("root", "read_file")
                .parallel(["a", "a"], operation="write_file")
            )

    def test_empty_parallel_raises(self) -> None:
        """parallel([]) with no node IDs is almost certainly a bug."""
        with pytest.raises(ValueError, match="at least one"):
            (
                DAGBuilder("t")
                .add_step("root", "read_file")
                .parallel([], operation="write_file")
            )

    def test_single_element_parallel(self) -> None:
        """parallel() with one ID works but requires join() before then()."""
        dag = (
            DAGBuilder("t")
            .add_step("root", "read_file")
            .parallel(["only"], operation="write_file")
            .join("end", "send_email")
            .build()
        )
        assert len(dag.nodes) == 3
        edges = {(e.source_id, e.target_id) for e in dag.edges}
        assert edges == {("root", "only"), ("only", "end")}


# ---------------------------------------------------------------------------
# join()
# ---------------------------------------------------------------------------


class TestJoin:
    def test_converges_parallel_branches(self) -> None:
        dag = (
            DAGBuilder("t")
            .add_step("root", "authenticate")
            .parallel(["a", "b", "c"], operation="delete_record")
            .join("end", "send_notification")
            .build()
        )
        edges_to_end = [e for e in dag.edges if e.target_id == "end"]
        assert len(edges_to_end) == 3
        sources = {e.source_id for e in edges_to_end}
        assert sources == {"a", "b", "c"}

    def test_join_without_parallel_acts_like_then(self) -> None:
        dag = (
            DAGBuilder("t")
            .add_step("a", "read_file")
            .join("b", "write_file")
            .build()
        )
        assert len(dag.edges) == 1
        assert dag.edges[0].source_id == "a"
        assert dag.edges[0].target_id == "b"

    def test_join_resets_parallel_heads(self) -> None:
        """After join(), then() should work normally."""
        dag = (
            DAGBuilder("t")
            .add_step("root", "read_file")
            .parallel(["a", "b"], operation="write_file")
            .join("mid", "send_email")
            .then("end", "read_database")
            .build()
        )
        assert len(dag.nodes) == 5
        # mid -> end edge exists
        assert any(e.source_id == "mid" and e.target_id == "end" for e in dag.edges)

    def test_join_custom_edge_type(self) -> None:
        dag = (
            DAGBuilder("t")
            .add_step("root", "read_file")
            .parallel(["a", "b"], operation="write_file")
            .join("end", "send_email", edge_type=EdgeType.DATA_FLOW)
            .build()
        )
        edges_to_end = [e for e in dag.edges if e.target_id == "end"]
        for e in edges_to_end:
            assert e.edge_type == EdgeType.DATA_FLOW

    def test_join_without_cursor_raises(self) -> None:
        with pytest.raises(ValueError, match="preceding"):
            DAGBuilder("t").join("a", "read_file")


# ---------------------------------------------------------------------------
# Full pipeline (matches NOD-12 example)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_nod12_example(self) -> None:
        """Reproduce the exact example from the Linear issue."""
        dag = (
            DAGBuilder("my-workflow")
            .add_step("auth", "authenticate", params={"method": "jwt"})
            .then("lookup", "read_database", params={"table": "users"})
            .parallel(["delete_posts", "delete_files"], operation="delete_record")
            .join("notify", "send_notification")
            .build()
        )
        assert dag.name == "my-workflow"
        assert len(dag.nodes) == 5
        nodes = {n.id: n for n in dag.nodes}
        assert nodes["auth"].operation == "authenticate"
        assert nodes["lookup"].operation == "read_database"
        assert nodes["delete_posts"].operation == "delete_record"
        assert nodes["delete_files"].operation == "delete_record"
        assert nodes["notify"].operation == "send_notification"

        edges = {(e.source_id, e.target_id) for e in dag.edges}
        assert ("auth", "lookup") in edges
        assert ("lookup", "delete_posts") in edges
        assert ("lookup", "delete_files") in edges
        assert ("delete_posts", "notify") in edges
        assert ("delete_files", "notify") in edges
        assert len(dag.edges) == 5

    def test_parallel_then_join_then_continue(self) -> None:
        """parallel -> join -> then chain works end to end."""
        dag = (
            DAGBuilder("pipeline")
            .add_step("start", "read_file")
            .parallel(["a", "b"], operation="invoke_api")
            .join("merge", "write_database")
            .then("done", "send_email")
            .build()
        )
        assert len(dag.nodes) == 5
        nodes = {n.id: n for n in dag.nodes}
        assert nodes["start"].operation == "read_file"
        assert nodes["a"].operation == "invoke_api"
        assert nodes["b"].operation == "invoke_api"
        assert nodes["merge"].operation == "write_database"
        assert nodes["done"].operation == "send_email"

        edges = {(e.source_id, e.target_id) for e in dag.edges}
        assert edges == {
            ("start", "a"), ("start", "b"),
            ("a", "merge"), ("b", "merge"),
            ("merge", "done"),
        }
        for e in dag.edges:
            assert e.edge_type == EdgeType.CONTROL_FLOW
