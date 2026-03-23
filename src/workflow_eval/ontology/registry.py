"""OperationRegistry — extensible catalog of agent operation definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import yaml

from workflow_eval.ontology.effect_types import EffectTarget, EffectType
from workflow_eval.types import OperationDefinition


class OperationRegistry:
    """Singleton-like registry of known agent operations."""

    def __init__(self) -> None:
        self._ops: dict[str, OperationDefinition] = {}

    def register(self, op_def: OperationDefinition) -> None:
        """Register an operation. Raises ValueError on duplicate name."""
        if op_def.name in self._ops:
            raise ValueError(f"Duplicate operation: {op_def.name!r}")
        self._ops[op_def.name] = op_def

    def get(self, name: str) -> OperationDefinition:
        """Retrieve by name. Raises KeyError if not found."""
        try:
            return self._ops[name]
        except KeyError:
            raise KeyError(f"Unknown operation: {name!r}") from None

    def all(self) -> list[OperationDefinition]:
        """Return all registered operations."""
        return list(self._ops.values())

    def __len__(self) -> int:
        return len(self._ops)

    def __iter__(self) -> Iterator[OperationDefinition]:
        return iter(self._ops.values())

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    def reset(self) -> None:
        """Clear all registered operations."""
        self._ops.clear()

    @classmethod
    def from_yaml(cls, path: str | Path) -> OperationRegistry:
        """Load a registry from a YAML file.

        Expected format:
            operations:
              - name: read_file
                category: io
                base_risk_weight: 0.05
                effect_type: pure
                effect_targets: [filesystem]
        """
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)

        registry = cls()
        for entry in data["operations"]:
            op = OperationDefinition(
                name=entry["name"],
                category=entry["category"],
                base_risk_weight=entry["base_risk_weight"],
                effect_type=EffectType(entry["effect_type"]),
                effect_targets=frozenset(
                    EffectTarget(t) for t in entry["effect_targets"]
                ),
            )
            registry.register(op)
        return registry
