"""Default ontology — ~20 built-in agent primitives."""

from workflow_eval.ontology.effect_types import EffectTarget, EffectType
from workflow_eval.ontology.registry import OperationRegistry
from workflow_eval.types import OperationDefinition

_T = EffectTarget
_E = EffectType

DEFAULT_OPERATIONS: tuple[OperationDefinition, ...] = (
    # --- I/O ---
    OperationDefinition(
        name="read_file",
        category="io",
        base_risk_weight=0.05,
        effect_type=_E.PURE,
        effect_targets=frozenset({_T.FILESYSTEM}),
    ),
    OperationDefinition(
        name="write_file",
        category="io",
        base_risk_weight=0.35,
        effect_type=_E.STATEFUL,
        effect_targets=frozenset({_T.FILESYSTEM}),
    ),
    OperationDefinition(
        name="delete_file",
        category="io",
        base_risk_weight=0.80,
        effect_type=_E.IRREVERSIBLE,
        effect_targets=frozenset({_T.FILESYSTEM}),
    ),
    # --- Database ---
    OperationDefinition(
        name="read_database",
        category="database",
        base_risk_weight=0.05,
        effect_type=_E.PURE,
        effect_targets=frozenset({_T.DATABASE}),
    ),
    OperationDefinition(
        name="write_database",
        category="database",
        base_risk_weight=0.40,
        effect_type=_E.STATEFUL,
        effect_targets=frozenset({_T.DATABASE}),
    ),
    OperationDefinition(
        name="delete_record",
        category="database",
        base_risk_weight=0.85,
        effect_type=_E.IRREVERSIBLE,
        effect_targets=frozenset({_T.DATABASE}),
    ),
    # --- Network ---
    OperationDefinition(
        name="invoke_api",
        category="network",
        base_risk_weight=0.30,
        effect_type=_E.EXTERNAL,
        effect_targets=frozenset({_T.NETWORK}),
    ),
    OperationDefinition(
        name="send_webhook",
        category="network",
        base_risk_weight=0.35,
        effect_type=_E.EXTERNAL,
        effect_targets=frozenset({_T.NETWORK}),
    ),
    # --- State ---
    OperationDefinition(
        name="mutate_state",
        category="state",
        base_risk_weight=0.25,
        effect_type=_E.STATEFUL,
        effect_targets=frozenset({_T.MEMORY}),
    ),
    OperationDefinition(
        name="read_state",
        category="state",
        base_risk_weight=0.05,
        effect_type=_E.PURE,
        effect_targets=frozenset({_T.MEMORY}),
    ),
    # --- Control flow ---
    OperationDefinition(
        name="branch",
        category="control",
        base_risk_weight=0.10,
        effect_type=_E.PURE,
        effect_targets=frozenset(),
    ),
    OperationDefinition(
        name="loop",
        category="control",
        base_risk_weight=0.15,
        effect_type=_E.PURE,
        effect_targets=frozenset(),
    ),
    # --- Execution ---
    OperationDefinition(
        name="execute_code",
        category="execution",
        base_risk_weight=0.60,
        effect_type=_E.EXTERNAL,
        effect_targets=frozenset({_T.MEMORY, _T.FILESYSTEM, _T.NETWORK}),
    ),
    # --- Auth ---
    OperationDefinition(
        name="authenticate",
        category="auth",
        base_risk_weight=0.20,
        effect_type=_E.EXTERNAL,
        effect_targets=frozenset({_T.CREDENTIALS, _T.NETWORK}),
    ),
    OperationDefinition(
        name="authorize",
        category="auth",
        base_risk_weight=0.15,
        effect_type=_E.PURE,
        effect_targets=frozenset({_T.CREDENTIALS}),
    ),
    OperationDefinition(
        name="read_credentials",
        category="auth",
        base_risk_weight=0.45,
        effect_type=_E.PURE,
        effect_targets=frozenset({_T.CREDENTIALS}),
    ),
    # --- Communication ---
    OperationDefinition(
        name="send_email",
        category="communication",
        base_risk_weight=0.50,
        effect_type=_E.IRREVERSIBLE,
        effect_targets=frozenset({_T.NETWORK, _T.USER_FACING}),
    ),
    OperationDefinition(
        name="send_notification",
        category="communication",
        base_risk_weight=0.30,
        effect_type=_E.EXTERNAL,
        effect_targets=frozenset({_T.NETWORK, _T.USER_FACING}),
    ),
    # --- Resource lifecycle ---
    OperationDefinition(
        name="create_resource",
        category="resource",
        base_risk_weight=0.35,
        effect_type=_E.STATEFUL,
        effect_targets=frozenset({_T.NETWORK, _T.SYSTEM_CONFIG}),
    ),
    OperationDefinition(
        name="destroy_resource",
        category="resource",
        base_risk_weight=0.90,
        effect_type=_E.IRREVERSIBLE,
        effect_targets=frozenset({_T.NETWORK, _T.SYSTEM_CONFIG}),
    ),
)


def load_defaults(registry: OperationRegistry | None = None) -> OperationRegistry:
    """Populate a registry with the 20 default operations."""
    if registry is None:
        registry = OperationRegistry()
    for op in DEFAULT_OPERATIONS:
        registry.register(op)
    return registry
