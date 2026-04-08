"""CLI entry point for workflow-eval (NOD-40).

NOD-40 spec (Linear):
- workflow_eval/cli.py
- Commands:
  * workflow-eval analyze <path> — prints risk profile to stdout
  * workflow-eval serve — starts HTTP server via uvicorn
  * workflow-eval ontology — lists registered operations
- Uses argparse or click

AC:
- [ ] workflow-eval analyze examples/sample_workflows/risky_delete_cascade.yaml
      prints human-readable risk profile to stdout
- [ ] workflow-eval serve starts uvicorn on configurable port
- [ ] workflow-eval ontology lists all 20+ registered operations

Behavioral constraints:
- Human-readable output for analyze (not raw JSON)
- Configurable port for serve
- Lists all registered operations for ontology
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="workflow-eval",
        description="Risk scoring for AI agent workflows.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # analyze
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a workflow file and print its risk profile.",
    )
    analyze_parser.add_argument(
        "path", help="Path to a YAML or JSON workflow file.",
    )

    # serve
    serve_parser = subparsers.add_parser(
        "serve", help="Start the HTTP API server.",
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000).",
    )
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1).",
    )

    # ontology
    subparsers.add_parser(
        "ontology", help="List all registered operations.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "analyze":
        _cmd_analyze(args.path)
    elif args.command == "serve":
        _cmd_serve(args.host, args.port)
    elif args.command == "ontology":
        _cmd_ontology()


def _cmd_analyze(path: str) -> None:
    from workflow_eval.dag.models import to_networkx
    from workflow_eval.dag.schema import load_workflow
    from workflow_eval.mitigation.engine import MitigationEngine
    from workflow_eval.ontology.defaults import get_default_registry
    from workflow_eval.scoring.engine import RiskScoringEngine
    from workflow_eval.types import ScoringConfig

    dag = load_workflow(path)
    nx_dag = to_networkx(dag)
    registry = get_default_registry()
    engine = RiskScoringEngine(ScoringConfig(), registry)
    profile = engine.score(nx_dag)

    mitigation_engine = MitigationEngine()
    plan = mitigation_engine.generate_plan(profile, nx_dag, registry)

    print(f"Workflow: {profile.workflow_name}")
    print(f"Nodes: {profile.node_count}  Edges: {profile.edge_count}")
    print(f"Risk: {profile.aggregate_score:.3f} ({profile.risk_level.value})")
    print()

    print("Sub-scores:")
    for sub in profile.sub_scores:
        bar = "#" * int(sub.score * 20)
        print(f"  {sub.name:<20s} {sub.score:.3f}  (weight {sub.weight})  {bar}")
    print()

    if profile.critical_paths:
        print("Critical paths:")
        for p in profile.critical_paths:
            print(f"  {' → '.join(p)}")
        print()

    if profile.chokepoints:
        print(f"Chokepoints: {', '.join(profile.chokepoints)}")
        print()

    if plan.mitigations:
        print(f"Mitigations ({plan.original_risk:.3f} → {plan.residual_risk:.3f}):")
        for m in plan.mitigations:
            targets = ", ".join(m.target_node_ids)
            print(f"  [{m.priority.value}] {m.action.value} on {targets}")
            print(f"         {m.reason}")
        print()


def _cmd_serve(host: str, port: int) -> None:
    import uvicorn

    uvicorn.run("workflow_eval.api.app:app", host=host, port=port)


def _cmd_ontology() -> None:
    from workflow_eval.ontology.defaults import get_default_registry

    registry = get_default_registry()
    ops = sorted(registry.all(), key=lambda o: (o.category, o.name))

    print(f"Registered operations ({len(registry)}):")
    print()
    current_category = None
    for op in ops:
        if op.category != current_category:
            current_category = op.category
            print(f"  [{current_category}]")
        targets = ", ".join(sorted(t.value for t in op.effect_targets))
        print(f"    {op.name:<25s} risk={op.base_risk_weight:.2f}  "
              f"effect={op.effect_type.value:<14s} targets={targets}")
