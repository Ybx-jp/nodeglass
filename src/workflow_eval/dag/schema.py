"""YAML/JSON workflow schema loading (NOD-13).

NOD-13 spec (Linear):
- load_workflow(path: Path | str) -> WorkflowDAG
- Supports .yaml, .yml, and .json files
- Parses YAML/JSON into dict, validates against WorkflowDAG Pydantic model
- Descriptive error messages on validation failure
- Structural validation (dangling edges, orphan nodes, cycles) belongs in
  validate_dag() (NOD-14), not here.

AC:
- [x] All 3 example workflow files load successfully
- [x] Missing node id raises descriptive ValidationError
- [x] Invalid edge_type value raises error
- [x] Unsupported file extension raises ValueError
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from workflow_eval.types import WorkflowDAG

_YAML_EXTENSIONS = {".yaml", ".yml"}
_JSON_EXTENSIONS = {".json"}
_SUPPORTED_EXTENSIONS = _YAML_EXTENSIONS | _JSON_EXTENSIONS


def load_workflow(path: Path | str) -> WorkflowDAG:
    """Load a WorkflowDAG from a YAML or JSON file.

    Parses the file into a dict, then validates against the WorkflowDAG
    Pydantic model. Raises ValueError for unsupported file extensions and
    lets Pydantic ValidationError propagate for schema violations.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension {suffix!r}; "
            f"expected one of: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    raw = path.read_text(encoding="utf-8")
    data = _parse(raw, suffix)

    return WorkflowDAG.model_validate(data)


def _parse(raw: str, suffix: str) -> dict[str, Any]:
    """Parse raw file content into a dict based on file extension."""
    if suffix in _YAML_EXTENSIONS:
        result = yaml.safe_load(raw)
        if not isinstance(result, dict):
            raise ValueError(
                f"Expected a YAML mapping at top level, got {type(result).__name__}"
            )
        return result
    else:
        parsed: dict[str, Any] = json.loads(raw)
        return parsed
