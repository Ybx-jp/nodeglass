"""Declarative DAG construction and scoring.

Build a workflow using the DAGBuilder, score it across 6 risk dimensions,
and generate a mitigation plan.

Usage:
    python examples/simple_workflow.py
"""

from workflow_eval.dag.builder import DAGBuilder
from workflow_eval.dag.models import to_networkx
from workflow_eval.mitigation.engine import MitigationEngine
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.scoring.engine import RiskScoringEngine
from workflow_eval.types import ScoringConfig


def main() -> None:
    # Build a user-cleanup workflow: auth → lookup → delete posts → delete account → notify
    dag = (
        DAGBuilder("user-cleanup")
        .add_step("auth", "authenticate")
        .then("lookup", "read_database")
        .then("delete-posts", "delete_record")
        .then("delete-account", "destroy_resource")
        .then("notify", "send_email")
        .build()
    )

    print(f"Workflow: {dag.name}")
    print(f"Nodes: {len(dag.nodes)}, Edges: {len(dag.edges)}")
    print(f"Pipeline: {' → '.join(n.id for n in dag.nodes)}")
    print()

    # Score it
    nx_dag = to_networkx(dag)
    registry = get_default_registry()
    engine = RiskScoringEngine(ScoringConfig(), registry)
    profile = engine.score(nx_dag)

    print(f"Risk: {profile.aggregate_score:.3f} ({profile.risk_level.value})")
    print()

    print("Sub-scores:")
    for sub in profile.sub_scores:
        bar = "█" * int(sub.score * 20) + "░" * (20 - int(sub.score * 20))
        print(f"  {sub.name:<20s} {bar} {sub.score:.3f}  (weight {sub.weight})")
    print()

    if profile.chokepoints:
        print(f"Chokepoints: {', '.join(profile.chokepoints)}")
    if profile.critical_paths:
        print(f"Critical path: {' → '.join(profile.critical_paths[0])}")
    print()

    # Generate mitigations
    mitigation_engine = MitigationEngine()
    plan = mitigation_engine.generate_plan(profile, nx_dag, registry)

    print(f"Mitigations ({plan.original_risk:.3f} → {plan.residual_risk:.3f}):")
    for m in plan.mitigations:
        targets = ", ".join(m.target_node_ids)
        print(f"  [{m.priority.value}] {m.action.value} on {targets}")
        print(f"         {m.reason}")


if __name__ == "__main__":
    main()
