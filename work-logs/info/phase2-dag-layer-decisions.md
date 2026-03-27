# Phase 2 DAG Layer — Implementation Log & Design Decisions

**Date**: 2026-03-22 / 2026-03-26
**Branch**: `phase2/yolo-full-send`
**Issues**: NOD-11 (DAG model + networkx), NOD-12 (DAGBuilder fluent API), NOD-13 (YAML/JSON loading), NOD-14 (DAG validation), NOD-15 (example workflows)
**Milestone**: Phase 2: DAG Layer
**PR**: #3 (NOD-13, NOD-14, NOD-15 + schema field renames)

---

## NOD-11: WorkflowDAG model with networkx conversion

### What was built

**File**: `src/workflow_eval/dag/models.py` (89 lines)
**Tests**: `tests/test_dag_models.py` (27 tests across 6 classes)

Three public functions:

| Function | Signature | Purpose |
|---|---|---|
| `to_networkx` | `(dag: WorkflowDAG) -> nx.DiGraph` | Converts Pydantic model to networkx graph |
| `from_networkx` | `(g: nx.DiGraph) -> WorkflowDAG` | Reconstructs Pydantic model from networkx graph |
| `validate_unique_node_ids` | `(dag: WorkflowDAG) -> None` | Standalone uniqueness check (raises `ValueError`) |

Additionally, a `@model_validator` was added to `WorkflowDAG` in `types.py` to enforce node ID uniqueness at construction time.

### Design decision: free functions vs methods

**Decision**: `to_networkx()` and `from_networkx()` are free functions in `dag/models.py`, not methods on `WorkflowDAG`.

**Why**: `WorkflowDAG` is a frozen Pydantic model defined in `types.py`. Adding methods to it that return `nx.DiGraph` would create an import dependency from `types.py` → `networkx`, coupling the core type definitions to a specific graph library. Every downstream consumer of `types.py` (scoring, mitigation, storage, API) would transitively import networkx even if they never touch graph operations.

**Downstream effect**: This establishes the architectural pattern for the entire project — `types.py` remains a pure data-definition module with zero heavy dependencies. Any module that needs to work with networkx imports `dag.models` explicitly. The scoring layer (Phase 3) will import both `types` and `dag.models`, keeping the dependency graph clean and testable in isolation.

**Alternative considered**: A `WorkflowDAG.to_networkx()` instance method would be more discoverable. Rejected because frozen Pydantic models with `extra="forbid"` resist method addition patterns, and the import coupling was the bigger concern.

### Design decision: edge_type stored as string value in networkx

**Decision**: `to_networkx()` stores `edge.edge_type.value` (the string `"control_flow"`) on networkx edge attributes, not the `EdgeType` enum instance.

**Why**: networkx attributes are plain Python dicts — storing enum instances would work but creates fragility. JSON serialization of the graph (e.g. for debugging or export) would break. String values are universally interoperable. `from_networkx()` reconstructs the enum via `EdgeType(attrs.get("edge_type", ...))`.

**Downstream effect**: Any code that inspects networkx edge attributes directly (scoring layer, visualization) works with plain strings. The `from_networkx()` path validates the string back into the enum, catching typos at reconstruction time rather than silently passing bad data.

### Design decision: dual uniqueness validation

**Decision**: Node ID uniqueness is enforced in two places — a `@model_validator(mode="after")` on `WorkflowDAG`, and a standalone `validate_unique_node_ids()` function.

**Why**: The model validator catches duplicates at construction time (any code path that builds a `WorkflowDAG` via Pydantic). The standalone function exists for cases where a DAG is constructed via `model_construct()` (bypassing validators) — this happens in test fixtures and could happen in performance-sensitive bulk-loading paths. The standalone function raises `ValueError` with a specific node ID; the model validator raises Pydantic `ValidationError` wrapping the same `ValueError`.

**Downstream effect**: The DAG validation module (NOD-14) can call `validate_unique_node_ids()` as part of its comprehensive check suite without re-implementing the logic. The model validator means most code never needs to think about uniqueness — it's enforced by the type system.

### Design decision: from_networkx() defaults

**Decision**: Missing `params` defaults to `{}`, missing `edge_type` defaults to `EdgeType.CONTROL_FLOW`, missing `metadata` defaults to `{}`, missing `name` defaults to `""`.

**Why**: `from_networkx()` needs to handle manually-constructed networkx graphs that may not have all attributes set. Defaulting to the least-assuming values (empty params, control flow edges) makes the function usable as a bridge from external graph sources. `control_flow` is the correct default because it's the weakest semantic claim — it says "B runs after A" without asserting data dependency.

**Downstream effect**: The DAGBuilder (NOD-12) doesn't use `from_networkx()`, but the YAML/JSON schema loader (NOD-13) might construct intermediate networkx graphs. These defaults ensure the loader doesn't need to be exhaustive about optional fields.

### Design decision: dangling edges deferred to NOD-14

**Decision**: `WorkflowDAG` does not validate that edge source/target IDs exist in the node list. This validation is explicitly deferred to `validate_dag()` in NOD-14.

**Why**: The Pydantic model's job is structural validity (types, frozen, unique IDs). Semantic validation (do edges reference real nodes? are operations registered? is the graph acyclic?) belongs in the validation layer. Adding all checks to the model validator would make construction expensive and would couple the model to the registry.

**Downstream effect**: It is currently possible to construct a `WorkflowDAG` with dangling edges. `to_networkx()` will silently auto-create phantom nodes (networkx behavior). The round-trip through `from_networkx()` will then fail with `KeyError` because the phantom node has no `operation` attribute. This is a known gap — NOD-14's "edge reference integrity" check will catch it before the graph reaches the scoring engine. A comment with full context was added to the NOD-14 Linear issue.

### Review fixes applied (dag-tests1.md)

The test file went through a review cycle. Six items were raised:

| # | Finding | Action |
|---|---|---|
| 1 | "No model validator for unique node IDs" | **Flagged as inaccurate** — the validator was added to `types.py:75-81` and the test passes. The review was written against an older snapshot. |
| 2 | Tautology: `test_unique_ids_accepted` asserts fixture properties | **Removed**. Replaced with `test_validate_unique_node_ids_helper_raises` using `model_construct()` to bypass the validator and test the standalone function's error path. |
| 3 | Weak assertion: full-cycle test only checks `len(nodes)`, `len(edges)` | **Strengthened**. Now verifies all node operations, params, metadata, and edge types through the 4-step conversion chain. |
| 4.1 | No test for `validate_unique_node_ids` raising on duplicates | **Covered** by item 2's replacement test. |
| 4.2 | No error path tests for `from_networkx` | **Added** `test_node_missing_operation_raises` (KeyError) and `test_invalid_edge_type_raises` (ValueError). |
| 4.3 | No dangling edge test | **Deferred to NOD-14** with detailed Linear comment including suggested test code. |

---

## NOD-12: DAGBuilder fluent API

### What was built

**File**: `src/workflow_eval/dag/builder.py` (167 lines)
**Tests**: `tests/test_dag_builder.py` (27 tests across 6 classes)

A `DAGBuilder` class with five public methods:

| Method | Creates node? | Creates edges? | State transition |
|---|---|---|---|
| `add_step(id, op)` | Yes | From cursor if exists | Sets cursor, clears parallel heads |
| `then(id, op)` | Yes | From cursor | Advances cursor |
| `parallel([ids], op)` | Yes (multiple) | From cursor to each | Accumulates parallel heads |
| `join(id, op)` | Yes | From each parallel head | Consumes parallel heads, sets cursor |
| `build()` | No | No | Returns frozen `WorkflowDAG` |

### Design decision: cursor + parallel heads state model

**Decision**: The builder tracks two pieces of state — a single `_cursor` (the node that `then()` and `parallel()` fan out from) and a `_parallel_heads` list (accumulated by `parallel()`, consumed by `join()`).

**Why**: The Linear issue's example implies a linear state machine: you're always at one "current position" in the graph, and `parallel()` creates a fork that `join()` resolves. A single cursor + pending list is the simplest model that supports this. The alternative — tracking a full set of "active tips" — would enable more complex graph shapes but adds ambiguity (which tip does `then()` advance?).

**Downstream effect**: The builder can express any DAG that follows the pattern: linear chain → fan-out → converge → linear chain → fan-out → ... This covers the vast majority of real agent workflows (authenticate → lookup → do N things in parallel → report). DAGs with arbitrary cross-edges or diamond patterns require manual construction via `WorkflowDAG(nodes=..., edges=...)` or the YAML loader (NOD-13).

### Design decision: chained parallel() accumulates from same cursor

**Decision**: `.parallel(["a", "b"], op="x").parallel(["c", "d"], op="y")` creates 4 branches all fanning from the same predecessor, not a tree where `["c", "d"]` fan from `["a", "b"]`.

**Why**: This was explicitly discussed with the user. The motivation is that different parallel operations (e.g. `delete_record` for some nodes, `write_file` for others) should be expressible without needing per-node operation overrides in a single `parallel()` call. Chaining gives compositional control: each `parallel()` call specifies one operation for its batch of nodes.

**Downstream effect**: The `parallel()` method does not support per-node operations. This is intentional — it keeps the API simple and pushes heterogeneous parallel operations into the chaining pattern. If a future use case needs per-node operations in a single call (e.g. a list of `(id, op, params)` tuples), that would be a new method (e.g. `parallel_mixed()`), not a change to `parallel()`.

**State invariant**: `_cursor` does NOT advance when `parallel()` is called. The cursor stays on the predecessor node. Only `add_step()`, `then()`, and `join()` advance the cursor. This means all chained `parallel()` calls fan from the same point.

### Design decision: then() after parallel() is an error

**Decision**: Calling `then()` or `add_step()` while `_parallel_heads` is non-empty raises `ValueError`.

**Why**: If you have pending parallel branches and call `then()`, the intent is ambiguous — should the new node follow one branch? All branches? The cursor? Rather than guess, the builder forces you to `join()` first to explicitly converge the branches. This makes the graph structure predictable from the call chain.

**Downstream effect**: The error message ("Cannot then() with pending parallel branches; call join() first") guides the caller toward the correct usage pattern. This is a strict-by-default design that can be relaxed later if a valid use case emerges, but cannot easily be tightened later without breaking callers.

### Design decision: join() without parallel heads acts like then()

**Decision**: If `join()` is called when there are no pending parallel heads, it creates a single edge from the cursor — identical to `then()`.

**Why**: This was the "more robust" option for handling the edge case. The alternatives were: (a) raise an error (strict but annoying — forces callers to track whether they've called `parallel()` before `join()`), or (b) silently do nothing (dangerous — the node is added but has no incoming edges). Acting like `then()` is predictable, safe, and means `join()` always produces a connected node.

**Downstream effect**: Code that conditionally calls `parallel()` based on runtime data doesn't need to branch between `join()` and `then()` — it can always call `join()` and get a valid graph either way. This matters for the instrumentation SDK (NOD-28) which may auto-build DAGs from observed execution where parallel paths aren't known until runtime.

### Design decision: optional edge_type on all methods, no ontology coupling

**Decision**: Every method that creates edges (`add_step`, `then`, `parallel`, `join`) accepts an optional `edge_type` parameter defaulting to `EdgeType.CONTROL_FLOW`. The builder has no knowledge of the operation ontology.

**Why**: Three options were considered:
1. **Dynamic edge types based on source node operation** — requires the builder to know the ontology registry, coupling construction to Layer 1.
2. **Per-edge caller specification** — breaks the fluent API ergonomics.
3. **Optional parameter with sane default** — simple, explicit, no coupling.

Option 3 was chosen. The builder is a construction convenience, not a semantic analysis tool. The caller knows whether they're passing data or sequencing control. The default of `control_flow` is safe (least-assuming).

**Downstream effect**: The scoring engine (Phase 3) treats `data_flow` and `control_flow` edges differently — data flow edges contribute to the compositional risk scorer. If a builder-constructed DAG uses all `control_flow` edges, the compositional scorer will undercount synergistic risk between operations. This is acceptable because: (a) the builder is one of several construction paths (YAML loader, manual construction, instrumentation SDK), and (b) callers who care about accurate scoring can specify edge types explicitly.

### Design decision: builder validates eagerly, build() validates via model

**Decision**: The builder checks for duplicate node IDs at `_add_node()` time (eager). The `build()` method delegates to `WorkflowDAG(...)` which runs the model validator.

**Why**: Eager validation gives immediate feedback — if you call `.then("a", "read_file")` and `"a"` already exists, you get the error at that call site, not at `build()` time when you've lost the context of which step was the duplicate. The model validator in `build()` provides a safety net (e.g. the uniqueness check runs again through Pydantic).

**Downstream effect**: The builder's `_node_ids` set and the model's `_unique_node_ids` validator are redundant by design. The builder set catches errors early with good messages; the model validator catches anything that slips through (e.g. if someone subclasses `DAGBuilder` and bypasses `_add_node()`).

---

---

## NOD-13: YAML/JSON schema loading

### What was built

**File**: `src/workflow_eval/dag/schema.py` (~30 lines of logic)
**Tests**: `tests/test_dag_schema.py` (22 tests across 5 classes)

Single public function: `load_workflow(path: Path | str) -> WorkflowDAG`
- Supports `.yaml`, `.yml`, `.json` extensions
- YAML parsed via `yaml.safe_load`, JSON via `json.loads`
- Validates against `WorkflowDAG` Pydantic model

### Design decision: loader delegates all schema validation to Pydantic

**Decision**: `load_workflow()` parses YAML/JSON into a dict and calls `WorkflowDAG.model_validate(data)`. No hand-written schema checks.

**Why**: Pydantic already enforces required fields, types, `extra="forbid"`, enum values, and the unique node ID validator. Duplicating any of this in the loader would create two sources of truth that could drift. ~30 lines of logic is the result.

**Downstream effect**: Any schema change (e.g. adding a field to `DAGNode`) automatically applies to file loading with zero loader changes. The tradeoff is that Pydantic validation errors from malformed files can be verbose — but they're accurate, and the loader wraps them with file path context.

### Design decision: YAML gets an explicit dict check; JSON does not

**Decision**: The loader checks `isinstance(result, dict)` after `yaml.safe_load` and raises a clear `ValueError`. No such check for JSON.

**Why**: `yaml.safe_load` can return `None` (empty file) or a list (YAML array), which would confuse Pydantic with unhelpful errors like "expected dict, got NoneType". For JSON, `json.loads` on a non-dict feeds into `model_validate` which raises a descriptive `ValidationError` — no extra check needed.

**Downstream effect**: Error messages for malformed YAML files are human-readable ("Expected a YAML mapping, got ...") rather than cryptic Pydantic internals.

### Design decision: structural validation deferred to validate_dag()

**Decision**: The loader only parses and constructs — it does not check for dangling edges, orphan nodes, or cycles.

**Why**: The Linear issue explicitly states: "structural validation belongs in `validate_dag()` (NOD-14), not here." The loader is a boundary validator for file format, not graph semantics. A file can load successfully but still contain a DAG with structural warnings/errors that `validate_dag()` will flag.

**Downstream effect**: Callers must run `validate_dag()` after `load_workflow()` if they want semantic guarantees. This is a deliberate two-step pattern that keeps concerns separated and allows callers to decide how strict they want to be (e.g. the MCP server may want to load and validate in one shot, but a debugging tool may want to load a broken DAG for inspection).

### Review fixes applied (dag-tests3.md)

4 items found, 0 inaccuracies:
- Strengthened `test_metadata_preserved` to assert full dict
- Added tests for: malformed JSON (JSONDecodeError), non-dict JSON (ValidationError), duplicate node IDs through `load_workflow()` entrypoint

---

## NOD-14: DAG validation

### What was built

**File**: `src/workflow_eval/dag/validation.py`
**Tests**: `tests/test_dag_validation.py` (25 tests across 8 classes)

Public function: `validate_dag(dag, registry) -> list[ValidationIssue]`

5 independent checks:

| Check | Level | Code |
|---|---|---|
| Edge reference integrity (source_id/target_id exist) | ERROR | `dangling_source` / `dangling_target` |
| Operation resolution (every node op in registry) | ERROR | `unknown_operation` |
| Orphan nodes (no incoming AND no outgoing edges) | WARNING | `orphan_node` |
| Root detection (at least one node with no incoming) | WARNING | `no_root` |
| Cycle detection (via `nx.find_cycle`) | WARNING | `cycle_detected` |

Supporting types: `ValidationIssue` (level, code, message, node_ids), `ValidationLevel` (StrEnum: warning, error).

### Design decision: all checks run, never short-circuit

**Decision**: `validate_dag()` always runs all 5 checks and returns the full list.

**Why**: Callers decide how to handle issues — filter by level, reject on errors only, etc. Short-circuiting on the first error would hide secondary problems. This is critical for the MCP server (NOD-28) which needs to present all findings to the agent in one response.

**Downstream effect**: The function is always O(V+E) regardless of input. No "fast fail" mode exists. If a future caller needs early exit, they can wrap with a short-circuiting helper — but the core function stays complete.

### Design decision: cycles are warnings, not errors

**Decision**: Cycle detection produces WARNING-level issues, not ERROR.

**Why**: The spec is explicit: "flags cycles as warnings (does NOT reject the DAG)." Some agent workflows intentionally include retry loops. The scoring engine uses cycle presence in the spectral scorer as a risk signal, but the workflow itself is still valid to analyze.

**Downstream effect**: A DAG with cycles will pass validation (no errors), but the warnings will be available for scoring and display. Callers that want to reject cyclic DAGs can filter for the `cycle_detected` code.

### Design decision: edge integrity checked before cycle detection

**Decision**: `_check_edge_integrity` runs before `_check_cycles`.

**Why**: `to_networkx()` silently auto-creates phantom nodes from dangling edges. If cycle detection ran first, a dangling edge forming a loop (e.g. `a→ghost, ghost→a`) would produce a confusing cycle warning involving a node that doesn't exist in the original DAG. By flagging the dangling edge first, the caller gets both issues but can prioritize the root cause.

**Downstream effect**: Tested explicitly in `test_phantom_node_cycle_from_dangling_edges`. The ordering is a correctness concern, not just UX — downstream code that processes issues in order can treat the first error as the root cause.

### Design decision: single-node DAGs are not orphans

**Decision**: `_check_orphan_nodes` skips DAGs with 0 or 1 nodes.

**Why**: A single-node DAG with no edges is a valid trivial workflow, not an orphan. The orphan check only makes sense when there are multiple nodes and some are disconnected from the graph.

**Downstream effect**: Trivial DAGs (common in tests and examples) don't produce spurious warnings.

### Design decision: no_root issue has empty node_ids

**Decision**: When every node has incoming edges (implying a cycle), the `no_root` warning uses `node_ids=()`.

**Why**: All nodes are equally "not a root" — listing them all would be noise. The root cause is the cycle, which is flagged separately.

**Downstream effect**: Tested explicitly after review feedback. Consumers can detect this case by checking `code == "no_root" and len(node_ids) == 0`.

### Review fixes applied (dag-tests4.md)

6 items found, 0 inaccuracies:
- 3 weak assertions tightened: added level assertion to dangling source test, narrowed frozen test from `Exception` to `ValidationError`, exact issue count in multiple-issues test
- 3 coverage gaps filled: phantom-node cycle from dangling edges, multiple orphans in isolation, `no_root` node_ids assertion

---

## NOD-15: Example workflow YAML files

### What was built

**Files**: `examples/sample_workflows/` — 3 example files
**Tests**: `tests/test_dag_examples.py` (validation + structural distinctness)

| File | Nodes | Pattern | Operations | Expected risk |
|---|---|---|---|---|
| `safe_read_pipeline.yaml` | 3 | Linear chain | read_file, read_database, read_state | Low |
| `risky_delete_cascade.yaml` | 6 | Fan-out deletes | authenticate, read_database, delete_record ×2, delete_file, invoke_api | High/critical |
| `moderate_api_chain.json` | 4 | Linear chain | read_database, mutate_state, invoke_api, write_database | Medium |

### Design decision: structural distinctness verified by test

**Decision**: A dedicated test asserts that the 3 example files are structurally distinct (different node counts, edge counts, and operation sets).

**Why**: AC #3 requires structural distinctness. Rather than relying on visual inspection, the test makes this machine-verifiable and prevents future changes to the example files from accidentally making them isomorphic.

**Downstream effect**: When the scoring engine (Phase 3, NOD-25) is implemented, these files will serve as integration test fixtures. The structural distinctness guarantee means they'll exercise different scoring code paths.

---

## Schema field renames (PR #3, breaking)

**Change**: `DAGEdge.source` → `source_id`, `DAGEdge.target` → `target_id`. `DAGEdge` gained `condition: str | None` and `metadata: dict[str, Any]`. `DAGNode` gained `metadata: dict[str, Any]`.

**Why**: The `source`/`target` names conflicted with Pydantic's internal namespace and were ambiguous (source of what?). The `_id` suffix makes it clear these are foreign keys referencing node IDs. The new `metadata` and `condition` fields were needed for the validation and scoring layers.

**Scope**: All consumers updated: `models.py`, `builder.py`, all test files. This was done as part of PR #3 to batch the breaking change with the new code that needed the new field names.

---

## Module structure after Phase 2

```
src/workflow_eval/dag/
├── __init__.py          # Public API: DAGBuilder, ValidationIssue, ValidationLevel,
│                        #   from_networkx, load_workflow, to_networkx, validate_dag,
│                        #   validate_unique_node_ids
├── builder.py           # DAGBuilder fluent API (NOD-12)
├── models.py            # networkx conversion + standalone validator (NOD-11)
├── schema.py            # YAML/JSON loading (NOD-13)
└── validation.py        # DAG validation (NOD-14)

examples/sample_workflows/
├── safe_read_pipeline.yaml      # 3 read ops, linear (NOD-15)
├── risky_delete_cascade.yaml    # 6 nodes, fan-out deletes (NOD-15)
└── moderate_api_chain.json      # 4 nodes, linear API chain (NOD-15)
```

---

## Test coverage summary

| File | Tests | Classes | Key coverage |
|---|---|---|---|
| `test_dag_models.py` | 27 | 6 | Construction, uniqueness (model + standalone), to_networkx attrs, from_networkx round-trip + defaults + error paths, JSON round-trip, full cycle |
| `test_dag_builder.py` | 27 | 6 | build() basics, linear chain, parallel fan-out + chaining, join convergence, error paths, full pipeline matching Linear example |
| `test_dag_schema.py` | 22 | 5 | YAML/JSON loading, extension handling, malformed input, metadata preservation, duplicate IDs through loader |
| `test_dag_validation.py` | 25 | 8 | All 5 checks, phantom-node cycles, multiple orphans, no_root empty node_ids, multiple simultaneous issues |
| `test_dag_examples.py` | — | — | Example file loading, validate_dag pass, structural distinctness |

Total DAG layer tests: 104 (+ 115 Phase 1 = 219 total)

---

## Open items for subsequent issues

| Item | Target issue | Context |
|---|---|---|
| Builder doesn't support arbitrary cross-edges | — | Diamond patterns, skip connections, etc. require manual `WorkflowDAG` construction or the YAML loader. Not a gap for the MVP use case. |
| Scoring the example files | NOD-25 | NOD-15 AC #4 (risk level verification) deferred to Phase 3 scoring. |
| NOD-16: DAG unit tests consolidation | NOD-16 | Coverage is already strong (104 tests). May just need a consolidation pass. |
