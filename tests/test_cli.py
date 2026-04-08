"""Tests for CLI entry point (NOD-40).

AC:
- [x] workflow-eval analyze examples/sample_workflows/risky_delete_cascade.yaml
      prints human-readable risk profile to stdout
- [x] workflow-eval serve starts uvicorn on configurable port
- [x] workflow-eval ontology lists all 20+ registered operations
"""

import sys
from io import StringIO
from unittest.mock import patch

import pytest

from workflow_eval.cli import main


# ---------------------------------------------------------------------------
# AC: analyze prints human-readable risk profile
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_analyze_risky_delete_cascade(self) -> None:
        with patch("sys.argv", ["workflow-eval", "analyze",
                                "examples/sample_workflows/risky_delete_cascade.yaml"]):
            buf = StringIO()
            with patch("sys.stdout", buf):
                main()
            output = buf.getvalue()

        assert "risky-delete-cascade" in output
        assert "Risk:" in output
        assert "high" in output
        assert "Sub-scores:" in output
        assert "Mitigations" in output

    def test_analyze_safe_pipeline(self) -> None:
        with patch("sys.argv", ["workflow-eval", "analyze",
                                "examples/sample_workflows/safe_read_pipeline.yaml"]):
            buf = StringIO()
            with patch("sys.stdout", buf):
                main()
            output = buf.getvalue()

        assert "Risk:" in output
        assert "Sub-scores:" in output

    def test_analyze_nonexistent_file(self) -> None:
        with patch("sys.argv", ["workflow-eval", "analyze", "nonexistent.yaml"]):
            with pytest.raises(FileNotFoundError):
                main()

    def test_analyze_shows_critical_paths(self) -> None:
        with patch("sys.argv", ["workflow-eval", "analyze",
                                "examples/sample_workflows/risky_delete_cascade.yaml"]):
            buf = StringIO()
            with patch("sys.stdout", buf):
                main()
            output = buf.getvalue()

        assert "Critical paths:" in output
        assert "→" in output

    def test_analyze_shows_chokepoints(self) -> None:
        with patch("sys.argv", ["workflow-eval", "analyze",
                                "examples/sample_workflows/risky_delete_cascade.yaml"]):
            buf = StringIO()
            with patch("sys.stdout", buf):
                main()
            output = buf.getvalue()

        assert "Chokepoints:" in output


# ---------------------------------------------------------------------------
# AC: serve starts uvicorn on configurable port
# ---------------------------------------------------------------------------


class TestServe:
    def test_serve_calls_uvicorn(self) -> None:
        with patch("sys.argv", ["workflow-eval", "serve"]):
            with patch("uvicorn.run") as mock_run:
                main()
        mock_run.assert_called_once_with(
            "workflow_eval.api.app:app", host="127.0.0.1", port=8000,
        )

    def test_serve_custom_port(self) -> None:
        with patch("sys.argv", ["workflow-eval", "serve", "--port", "9000"]):
            with patch("uvicorn.run") as mock_run:
                main()
        mock_run.assert_called_once_with(
            "workflow_eval.api.app:app", host="127.0.0.1", port=9000,
        )

    def test_serve_custom_host(self) -> None:
        with patch("sys.argv", ["workflow-eval", "serve",
                                "--host", "0.0.0.0", "--port", "3000"]):
            with patch("uvicorn.run") as mock_run:
                main()
        mock_run.assert_called_once_with(
            "workflow_eval.api.app:app", host="0.0.0.0", port=3000,
        )


# ---------------------------------------------------------------------------
# AC: ontology lists all 20+ registered operations
# ---------------------------------------------------------------------------


class TestOntology:
    def test_ontology_lists_all_operations(self) -> None:
        with patch("sys.argv", ["workflow-eval", "ontology"]):
            buf = StringIO()
            with patch("sys.stdout", buf):
                main()
            output = buf.getvalue()

        assert "Registered operations (20):" in output

    def test_ontology_shows_categories(self) -> None:
        with patch("sys.argv", ["workflow-eval", "ontology"]):
            buf = StringIO()
            with patch("sys.stdout", buf):
                main()
            output = buf.getvalue()

        for category in ["io", "database", "network", "auth", "execution"]:
            assert f"[{category}]" in output

    def test_ontology_shows_risk_and_effect(self) -> None:
        with patch("sys.argv", ["workflow-eval", "ontology"]):
            buf = StringIO()
            with patch("sys.stdout", buf):
                main()
            output = buf.getvalue()

        assert "risk=" in output
        assert "effect=" in output
        assert "targets=" in output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_command_exits(self) -> None:
        with patch("sys.argv", ["workflow-eval"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
