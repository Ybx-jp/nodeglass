# workflow-eval

A geometric eval framework for analyzing risk in AI agent workflows.

Agents are being given increasingly powerful tools — file systems, databases, APIs, code execution, credentials. When an agent chains these operations together into a workflow, the risk isn't just the sum of the parts. A `read_file` followed by a `delete_record` is qualitatively different from two `read_file` operations. **workflow-eval** treats agent workflows as directed acyclic graphs and applies graph-theoretic scoring to quantify that structural risk.

## Why this exists

Most AI safety tooling focuses on what a model *says*. But as agents gain the ability to *act* — executing multi-step workflows with real side effects — the risk surface shifts from language to topology. The questions that matter become structural:

- How deep is the longest chain of irreversible operations?
- Is there a single node that, if it fails, takes down the entire workflow?
- How many operations fan out from a single decision point?
- What's the spectral gap of this graph, and what does it tell us about fragility?

These aren't questions you can answer by reading individual tool calls. You need the graph.

**workflow-eval** provides a framework to:

1. **Model** agent operations as a typed ontology with risk weights and effect classifications
2. **Construct** workflow DAGs — either declaratively (for static analysis) or at runtime (for live instrumentation)
3. **Score** workflows across six independent geometric dimensions
4. **Recommend** concrete mitigations based on structural risk patterns
5. **Persist** executions with predicted vs. actual outcomes for continuous evaluation

## Quick start

```bash
pip install workflow-eval
```

### Score a workflow

```python
from workflow_eval.dag.builder import DAGBuilder
from workflow_eval.dag.models import to_networkx
from workflow_eval.ontology.defaults import get_default_registry
from workflow_eval.scoring.engine import RiskScoringEngine
from workflow_eval.types import ScoringConfig

# Build a workflow
dag = (DAGBuilder("user-cleanup")
    .add_step("auth", "authenticate")
    .then("lookup", "read_database")
    .then("delete-posts", "delete_record")
    .then("delete-account", "destroy_resource")
    .then("notify", "send_email")
    .build())

# Score it
engine = RiskScoringEngine(ScoringConfig(), get_default_registry())
profile = engine.score(to_networkx(dag))

print(f"Risk: {profile.aggregate_score:.2f} ({profile.risk_level})")
print(f"Chokepoints: {profile.chokepoints}")
print(f"Critical path: {profile.critical_paths}")
for sub in profile.sub_scores:
    print(f"  {sub.name}: {sub.score:.2f}")
```

### Instrument a live workflow

```python
from workflow_eval.instrumentation import workflow_context, track_operation

@track_operation("authenticate")
async def login():
    return await auth_service.get_token()

@track_operation("delete_record", params={"table": "users"})
async def delete_user(user_id):
    await db.delete("users", user_id)

async with workflow_context("user-deletion") as wf:
    token = await login()
    await delete_user("u-123")

    # Check risk mid-flight
    risk = wf.get_current_risk()
    if risk.risk_level == "critical":
        raise AbortWorkflow("Risk too high to continue")
```

### Generate mitigations

```python
from workflow_eval.mitigation import MitigationEngine

mitigation_engine = MitigationEngine()
plan = mitigation_engine.generate_plan(profile, nx_dag, registry)

print(f"Residual risk: {plan.residual_risk:.2f} (from {plan.original_risk:.2f})")
for m in plan.mitigations:
    print(f"  [{m.priority}] {m.action} on {m.target_node_ids}: {m.reason}")
```

## How scoring works

Workflows are scored across six independent dimensions. Each scorer produces a 0–1 subscore, and the weighted aggregate determines the overall risk level.

| Scorer | Weight | What it measures |
|---|---|---|
| **Irreversibility** | 0.25 | Proportion of operations that can't be undone (deletes, destroys) |
| **Chain depth** | 0.20 | Length of the longest sequential dependency chain |
| **Fan-out** | 0.15 | Maximum number of operations triggered by a single node |
| **Centrality** | 0.15 | Betweenness centrality — identifies bottleneck nodes |
| **Compositional** | 0.15 | Risk weight of operations relative to their graph position |
| **Spectral** | 0.10 | Algebraic connectivity — structural fragility of the graph |

Risk levels: **low** (< 0.25), **medium** (0.25–0.50), **high** (0.50–0.75), **critical** (>= 0.75).

Weights are configurable via `ScoringConfig`. All scorers implement a common `Scorer` protocol, so custom scorers can be added without modifying the engine.

## The operation ontology

Every operation in a workflow maps to a typed definition with:

- **Risk weight** (0–1): How dangerous is this operation in isolation?
- **Effect type**: `PURE`, `STATEFUL`, `EXTERNAL`, or `IRREVERSIBLE`
- **Effect targets**: What does it touch? (`FILESYSTEM`, `DATABASE`, `NETWORK`, `CREDENTIALS`, `USER_FACING`, etc.)

The default registry includes 20 common agent operations across I/O, database, network, state, execution, auth, communication, and resource management. Custom operations can be registered or loaded from YAML.

## Mitigation engine

When scoring reveals risk, the mitigation engine analyzes the DAG and recommends concrete actions:

| Pattern | Action | Priority |
|---|---|---|
| Irreversible operations | Add confirmation gate | Required |
| Irreversible operations | Add rollback plan | Recommended |
| External API calls | Sandbox external calls | Recommended |
| High fan-out (> 3) | Reduce parallelism | Optional |
| Credential access | Add audit logging | Required |
| User-facing operations | Require authentication | Recommended |
| High-risk external calls | Add rate limiting | Optional |
| Uncertain predecessors of irreversible ops | Add retry logic | Recommended |

Mitigations are deduplicated, sorted by priority, and the engine estimates residual risk: `original_risk * (0.5 ^ required_mitigation_count)`.

## Storage

Scored workflows and execution traces persist to SQLite for longitudinal analysis:

```python
from workflow_eval.storage import SQLiteWorkflowRepository
from workflow_eval.instrumentation import WorkflowRecorder, workflow_context

repo = SQLiteWorkflowRepository("workflows.db")
recorder = WorkflowRecorder(repo)

async with workflow_context("my-workflow", recorder=recorder) as wf:
    # ... operations execute ...
    pass
# On exit: scores, stores, and records predicted vs actual outcome
```

## Architecture

```
src/workflow_eval/
  ontology/       # Operation definitions, effect types, registry
  dag/            # DAG models, builder, networkx conversion, validation
  scoring/        # 6 risk scorers + aggregation engine
  mitigation/     # 7 strategy rules + deduplicating engine
  storage/        # SQLite schema, repository protocol, migrations
  instrumentation/# Runtime context manager, decorator, persistence
  types.py        # Shared Pydantic models
```

The framework is layered: each module depends only on the layers below it. The ontology defines operations, the DAG layer structures them, the scoring engine evaluates them, the mitigation engine acts on scores, and the storage + instrumentation layers close the loop.

## Broader applications

While **workflow-eval** is built for AI agent workflows, the core idea — *scoring directed graphs of typed operations for structural risk* — applies anywhere a sequence of side-effecting actions needs oversight:

- **CI/CD pipelines**: Does this deployment graph have a single point of failure? Are irreversible production deployments gated by confirmation?
- **Data pipelines**: How fragile is this ETL graph? What happens if the transformation node between ingest and load fails?
- **Business process automation**: RPA workflows that touch databases, send emails, and modify records have the same structural risk patterns as agent workflows.
- **Infrastructure-as-code**: Terraform plans are DAGs of resource mutations. Scoring them before `apply` could prevent cascading deletions.
- **Multi-agent orchestration**: When agents coordinate through shared state, the combined workflow graph reveals emergent risks invisible to any single agent.

The operation ontology is extensible. The scorer protocol is pluggable. The graph is the universal abstraction.

## Development

```bash
git clone https://github.com/Ybx-jp/nodeglass.git
cd nodeglass
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v  # 653 tests
```

## License

MIT
