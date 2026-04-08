"""Runtime instrumentation with context manager and decorator.

Demonstrates auto-building a DAG at runtime as operations execute,
checking risk mid-flight, and mixing decorators with context managers.

Usage:
    python examples/runtime_instrumentation.py
"""

import asyncio

from workflow_eval.instrumentation.sdk import track_operation, workflow_context


# Decorated functions — tracked automatically inside a workflow_context
@track_operation("authenticate")
async def login():
    return "token-abc123"


@track_operation("invoke_api", params={"endpoint": "/users", "method": "GET"})
async def fetch_users():
    return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]


@track_operation("delete_record", params={"table": "users"})
async def delete_user(user_id: int):
    pass  # simulated delete


@track_operation("send_email")
async def notify_admin():
    pass  # simulated notification


async def main() -> None:
    print("=== Runtime Instrumentation Demo ===")
    print()

    async with workflow_context("user-deletion-flow") as wf:
        # Decorated functions auto-track
        token = await login()
        print(f"  [1] Authenticated: {token}")

        users = await fetch_users()
        print(f"  [2] Fetched {len(users)} users")

        # Check risk mid-flight before destructive operations
        risk = wf.get_current_risk()
        print(f"  ... mid-flight risk: {risk.aggregate_score:.3f} ({risk.risk_level.value})")
        print()

        # Context manager style — mix freely with decorators
        async with wf.operation("read_database") as op:
            print("  [3] Read database (context manager)")

        await delete_user(users[0]["id"])
        print(f"  [4] Deleted user {users[0]['id']}")

        await notify_admin()
        print("  [5] Notified admin")

    # After exit: inspect the auto-built DAG
    dag = wf.get_dag()
    execution = wf.get_execution()
    final_risk = wf.get_current_risk()

    print()
    print("=== Results ===")
    print()
    print(f"Workflow: {dag.name}")
    print(f"Nodes: {len(dag.nodes)}, Edges: {len(dag.edges)}")
    print(f"Pipeline: {' → '.join(n.operation for n in dag.nodes)}")
    print()

    print(f"Final risk: {final_risk.aggregate_score:.3f} ({final_risk.risk_level.value})")
    print()

    print("Sub-scores:")
    for sub in final_risk.sub_scores:
        bar = "█" * int(sub.score * 20) + "░" * (20 - int(sub.score * 20))
        print(f"  {sub.name:<20s} {bar} {sub.score:.3f}")
    print()

    print("Execution records:")
    for rec in execution.records:
        print(f"  {rec.node_id:<25s} {rec.operation:<20s} → {rec.outcome.value}")


if __name__ == "__main__":
    asyncio.run(main())
