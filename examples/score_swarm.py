"""Score all sample workflows with an extended custom operation registry.

Demonstrates registry extensibility: adds domain-specific operations
to the default 20, then scores every workflow through the full engine
with custom compositional risk entries.
"""

from pathlib import Path

from workflow_eval.dag import load_workflow, to_networkx
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.ontology.effect_types import EffectTarget, EffectType
from workflow_eval.scoring.compositional import CompositionalScorer
from workflow_eval.scoring.engine import RiskScoringEngine
from workflow_eval.types import OperationDefinition, ScoringConfig

_T = EffectTarget
_E = EffectType

WORKFLOWS_DIR = Path(__file__).parent / "sample_workflows"

# ---------------------------------------------------------------------------
# Domain-specific operations beyond the default 20
# ---------------------------------------------------------------------------

CUSTOM_OPS = (
    # --- Agent / autonomous ---
    OperationDefinition(
        name="ingest_user_input",
        category="io",
        base_risk_weight=0.70,
        effect_type=_E.EXTERNAL,
        effect_targets=frozenset({_T.NETWORK, _T.MEMORY}),
    ),
    OperationDefinition(
        name="spawn_agent",
        category="execution",
        base_risk_weight=0.75,
        effect_type=_E.EXTERNAL,
        effect_targets=frozenset({_T.MEMORY, _T.NETWORK, _T.FILESYSTEM}),
    ),
    OperationDefinition(
        name="web_search",
        category="network",
        base_risk_weight=0.55,
        effect_type=_E.EXTERNAL,
        effect_targets=frozenset({_T.NETWORK}),
    ),
    OperationDefinition(
        name="open_port",
        category="network",
        base_risk_weight=0.85,
        effect_type=_E.EXTERNAL,
        effect_targets=frozenset({_T.NETWORK, _T.SYSTEM_CONFIG}),
    ),
    OperationDefinition(
        name="coordinate_agents",
        category="state",
        base_risk_weight=0.45,
        effect_type=_E.STATEFUL,
        effect_targets=frozenset({_T.MEMORY, _T.NETWORK}),
    ),
    # --- ML / embedding ---
    OperationDefinition(
        name="embed_text",
        category="ml",
        base_risk_weight=0.03,
        effect_type=_E.PURE,
        effect_targets=frozenset({_T.MEMORY}),
    ),
    OperationDefinition(
        name="classify_content",
        category="ml",
        base_risk_weight=0.05,
        effect_type=_E.PURE,
        effect_targets=frozenset({_T.MEMORY}),
    ),
    # --- Vector store ---
    OperationDefinition(
        name="vector_store_write",
        category="database",
        base_risk_weight=0.30,
        effect_type=_E.STATEFUL,
        effect_targets=frozenset({_T.DATABASE}),
    ),
    OperationDefinition(
        name="vector_store_query",
        category="database",
        base_risk_weight=0.03,
        effect_type=_E.PURE,
        effect_targets=frozenset({_T.DATABASE}),
    ),
    # --- Finance ---
    OperationDefinition(
        name="charge_payment",
        category="finance",
        base_risk_weight=0.95,
        effect_type=_E.IRREVERSIBLE,
        effect_targets=frozenset({_T.NETWORK, _T.DATABASE}),
    ),
    # --- Validation ---
    OperationDefinition(
        name="validate_input",
        category="io",
        base_risk_weight=0.02,
        effect_type=_E.PURE,
        effect_targets=frozenset(),
    ),
    # --- Infrastructure ---
    OperationDefinition(
        name="quarantine_resource",
        category="resource",
        base_risk_weight=0.65,
        effect_type=_E.STATEFUL,
        effect_targets=frozenset({_T.NETWORK, _T.SYSTEM_CONFIG}),
    ),
    OperationDefinition(
        name="rollback_deployment",
        category="resource",
        base_risk_weight=0.70,
        effect_type=_E.EXTERNAL,
        effect_targets=frozenset({_T.NETWORK, _T.SYSTEM_CONFIG}),
    ),
    OperationDefinition(
        name="provision_infrastructure",
        category="resource",
        base_risk_weight=0.60,
        effect_type=_E.EXTERNAL,
        effect_targets=frozenset({_T.NETWORK, _T.SYSTEM_CONFIG}),
    ),
    # --- Social / communication ---
    OperationDefinition(
        name="post_social_media",
        category="communication",
        base_risk_weight=0.65,
        effect_type=_E.IRREVERSIBLE,
        effect_targets=frozenset({_T.NETWORK, _T.USER_FACING}),
    ),
    OperationDefinition(
        name="scrape_url",
        category="network",
        base_risk_weight=0.40,
        effect_type=_E.EXTERNAL,
        effect_targets=frozenset({_T.NETWORK}),
    ),
    # --- Database ---
    OperationDefinition(
        name="run_migration",
        category="database",
        base_risk_weight=0.90,
        effect_type=_E.IRREVERSIBLE,
        effect_targets=frozenset({_T.DATABASE}),
    ),
    # --- Security ---
    OperationDefinition(
        name="encrypt_data",
        category="security",
        base_risk_weight=0.05,
        effect_type=_E.PURE,
        effect_targets=frozenset({_T.MEMORY}),
    ),
)

# ---------------------------------------------------------------------------
# Domain-specific composition entries
# ---------------------------------------------------------------------------

CUSTOM_COMPOSITIONS: dict[tuple[str, str], float] = {
    # Agent / autonomous risk pairs
    ("ingest_user_input", "execute_code"): 2.8,
    ("ingest_user_input", "spawn_agent"): 2.5,
    ("ingest_user_input", "charge_payment"): 3.0,
    ("ingest_user_input", "post_social_media"): 2.8,
    ("web_search", "execute_code"): 2.3,
    ("web_search", "post_social_media"): 2.5,
    ("spawn_agent", "open_port"): 2.5,
    ("spawn_agent", "web_search"): 2.0,
    ("coordinate_agents", "execute_code"): 2.0,
    ("coordinate_agents", "send_email"): 1.8,
    ("open_port", "invoke_api"): 2.5,
    ("web_search", "write_database"): 2.0,
    ("execute_code", "destroy_resource"): 2.5,
    # Finance
    ("read_credentials", "charge_payment"): 2.8,
    ("charge_payment", "send_email"): 2.0,
    # Infrastructure
    ("execute_code", "run_migration"): 2.5,
    ("rollback_deployment", "destroy_resource"): 2.5,
    ("provision_infrastructure", "open_port"): 2.0,
    ("provision_infrastructure", "execute_code"): 2.0,
    # Content / social
    ("scrape_url", "execute_code"): 2.3,
    ("classify_content", "quarantine_resource"): 1.5,
    ("execute_code", "post_social_media"): 2.3,
}


def build_registry():
    """Build registry: 20 defaults + custom operations."""
    registry = get_default_registry()
    for op in CUSTOM_OPS:
        registry.register(op)
    return registry


def build_engine(registry):
    """Build engine with custom compositional scorer."""
    engine = RiskScoringEngine(config=ScoringConfig(), registry=registry)
    custom_comp = CompositionalScorer(compositions=CUSTOM_COMPOSITIONS)
    engine._scorers = tuple(
        custom_comp if s.name == "compositional" else s
        for s in engine._scorers
    )
    return engine


def score_workflow(engine, path):
    """Load and score a single workflow file."""
    dag = load_workflow(path)
    g = to_networkx(dag)
    return engine.score(g)


def main() -> None:
    registry = build_registry()
    engine = build_engine(registry)

    yamls = sorted(WORKFLOWS_DIR.glob("*.yaml"))
    jsons = sorted(WORKFLOWS_DIR.glob("*.json"))
    files = yamls + jsons

    print(f"Scoring {len(files)} workflows\n")
    print(f"{'Workflow':<35} {'Nodes':>5} {'Edges':>5}  {'Score':>6}  {'Level':<8}  Expected")
    print("─" * 85)

    for path in files:
        profile = score_workflow(engine, path)
        meta = {}
        dag = load_workflow(path)
        if hasattr(dag, "metadata") and dag.metadata:
            meta = dag.metadata
        expected = meta.get("expected_risk", "?")

        level = profile.risk_level.value
        match = "✓" if level == expected else ("" if expected == "?" else "✗")

        print(
            f"  {profile.workflow_name:<33} {profile.node_count:>5} {profile.edge_count:>5}"
            f"  {profile.aggregate_score:>6.3f}  {level:<8}  {expected} {match}"
        )

    # Detailed breakdown for one interesting workflow
    print("\n" + "═" * 85)
    print("Detailed: autonomous-agent-swarm")
    print("═" * 85)

    profile = score_workflow(engine, WORKFLOWS_DIR / "autonomous_swarm.yaml")

    print(f"\nAggregate: {profile.aggregate_score:.4f} ({profile.risk_level.value.upper()})")
    print(f"Nodes: {profile.node_count}, Edges: {profile.edge_count}\n")

    print("Sub-scores:")
    for s in profile.sub_scores:
        bar = "█" * int(s.score * 20) + "░" * (20 - int(s.score * 20))
        print(f"  {s.name:20s} {bar} {s.score:.4f} (w={s.weight:.2f})")
        if s.flagged_nodes:
            print(f"    flagged: {list(s.flagged_nodes)}")
        if s.name == "compositional" and s.details.get("highest_risk_edge"):
            d = s.details
            print(f"    worst edge: {d['highest_risk_edge']}")
            print(f"    ops: {d['highest_risk_ops']}, C={d['composition_multiplier']}, risk={d['edge_risk']:.4f}")
        if s.name == "spectral":
            print(f"    fiedler={s.details['fiedler_value']:.4f}")

    print(f"\nCritical path: {' → '.join(profile.critical_paths[0])}")
    print(f"Chokepoints:   {list(profile.chokepoints)}")


if __name__ == "__main__":
    main()
