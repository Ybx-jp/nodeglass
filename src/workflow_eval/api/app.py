"""FastAPI application factory (NOD-37)."""

from __future__ import annotations

from fastapi import FastAPI

from workflow_eval.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="workflow-eval",
        description="Risk scoring for AI agent workflows.",
        version="0.1.0",
    )
    app.include_router(router)
    return app


app = create_app()
