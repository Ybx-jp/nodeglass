"""What-if workflow hardening: score → mitigate → apply → re-score.

Loads the risky-delete-cascade workflow (high risk), generates mitigation
suggestions, applies safeguards to the DAG, and compares before/after.

This demonstrates the full feedback loop: analyze a workflow's risk,
understand *why* it's risky, insert structural safeguards, and measure
the improvement.

Usage:
    python examples/harden_workflow.py
"""

from pathlib import Path

from workflow_eval.dag import load_workflow, to_networkx
from workflow_eval.dag.builder import DAGBuilder
from workflow_eval.mitigation.engine import MitigationEngine
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.scoring.engine import RiskScoringEngine
from workflow_eval.types import ScoringConfig, WorkflowDAG

WORKFLOW = Path(__file__).parent / "sample_workflows" / "risky_delete_cascade.yaml"


def display_risk(label: str, profile) -> None:
    """Print a compact risk summary with sub-score bars."""
    print(f"\n{'─' * 65}")
    print(f"  {label}")
    print(f"{'─' * 65}")
    print(f"  Risk: {profile.aggregate_score:.3f} ({profile.risk_level.value})")
    print(f"  Nodes: {profile.node_count}, Edges: {profile.edge_count}")
    print()
    for sub in profile.sub_scores:
        bar = "█" * int(sub.score * 20) + "░" * (20 - int(sub.score * 20))
        print(f"    {sub.name:<20s} {bar} {sub.score:.3f}")
    if profile.chokepoints:
        print(f"\n  Chokepoints: {', '.join(profile.chokepoints)}")
    if profile.critical_paths:
        print(f"  Critical path: {' → '.join(profile.critical_paths[0])}")


def harden_workflow(original: WorkflowDAG) -> WorkflowDAG:
    """Rebuild the workflow with safeguards inserted.

    The original risky-delete-cascade topology:

        call_api → mutate → exec ─┬─ del_records ─┐
                                   ├─ del_files ───┤→ notify
                                   └─ destroy ─────┘

Hardened topology inserts safeguards around the original flow:

        authenticate → authorize → call_api → mutate → read_state
            → read_database → exec ─┬─ del_records ─┐
                                     ├─ del_files ───┤→ notify
                                     └─ destroy ─────┘

    Safeguards added:
      - authenticate + authorize: gate entry before any state changes
      - read_state: checkpoint before the irreversible fan-out
      - read_database: backup/verify records before deletion

    Return a new WorkflowDAG with name "risky-delete-cascade-hardened".
    """
    return (
        DAGBuilder("risky-delete-cascade-hardened")
        .add_step("authenticate", "authenticate", params={"method": "jwt"})
        .then("authorize", "authorize", params={"scope": "admin"})
        .then("call_api", "invoke_api", params={"endpoint": "/users"})
        .then("mutate", "mutate_state")
        .then("read_state", "read_state")
        .then("read_database", "read_database", params={"table": "users"})
        .then("exec", "execute_code", params={"script": "cleanup.sh"})
        .parallel(["del_records"], "delete_record", params={"table": "users"})
        .parallel(["del_files"], "delete_file")
        .parallel(["destroy"], "destroy_resource")
        .join("notify", "send_notification")
        .build()
    )


def main() -> None:
    registry = get_default_registry()
    engine = RiskScoringEngine(ScoringConfig(), registry)
    mitigation_engine = MitigationEngine()

    # ── 1. Load and score the original ──────────────────────────────
    original = load_workflow(WORKFLOW)
    nx_original = to_networkx(original)
    original_profile = engine.score(nx_original)
    display_risk(f"ORIGINAL: {original.name}", original_profile)

    # ── 2. Generate mitigation suggestions ──────────────────────────
    plan = mitigation_engine.generate_plan(original_profile, nx_original, registry)

    print(f"\n{'═' * 65}")
    print("  SUGGESTED MITIGATIONS")
    print(f"{'═' * 65}")
    for m in plan.mitigations:
        targets = ", ".join(m.target_node_ids)
        print(f"  [{m.priority.value:>11}] {m.action.value:<25s} → {targets}")
    print(f"\n  Engine estimate: {plan.original_risk:.3f} → {plan.residual_risk:.3f}")

    # ── 3. Apply safeguards and re-score ────────────────────────────
    hardened = harden_workflow(original)
    nx_hardened = to_networkx(hardened)
    hardened_profile = engine.score(nx_hardened)
    display_risk(f"HARDENED: {hardened.name}", hardened_profile)

    # ── 4. Before / after comparison ────────────────────────────────
    delta = original_profile.aggregate_score - hardened_profile.aggregate_score
    pct = (delta / original_profile.aggregate_score) * 100

    print(f"\n{'═' * 65}")
    print("  COMPARISON")
    print(f"{'═' * 65}")
    print(f"  Original:  {original_profile.aggregate_score:.3f} ({original_profile.risk_level.value})")
    print(f"  Hardened:  {hardened_profile.aggregate_score:.3f} ({hardened_profile.risk_level.value})")
    print(f"  Reduction: -{delta:.3f} ({pct:.1f}% lower)")
    print(f"  Nodes:     {original_profile.node_count} → {hardened_profile.node_count}")
    print(f"  Edges:     {original_profile.edge_count} → {hardened_profile.edge_count}")
    print()

    # Show which sub-scores improved most
    print("  Sub-score deltas:")
    orig_subs = {s.name: s.score for s in original_profile.sub_scores}
    for sub in hardened_profile.sub_scores:
        orig = orig_subs.get(sub.name, 0)
        d = orig - sub.score
        arrow = "▼" if d > 0 else "▲" if d < 0 else "─"
        print(f"    {sub.name:<20s} {orig:.3f} → {sub.score:.3f}  {arrow} {abs(d):.3f}")


if __name__ == "__main__":
    main()
