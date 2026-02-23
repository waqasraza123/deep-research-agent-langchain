from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent_factory import AgentService
from .artifacts import (
    artifact_abs_path,
    ensure_required_artifacts,
    ensure_thread_dir,
    list_artifacts,
)
from .logging_config import configure_logging
from .settings import Settings


log = logging.getLogger("deep_research_agent.api")


class RunRequest(BaseModel):
    question: str = Field(..., min_length=5)
    urls: list[str] = []
    thread_id: str | None = None


def create_app(
    *,
    settings: Settings | None = None,
    service: AgentService | None = None,
) -> FastAPI:
    """
    Production: call with no args.
    Tests: pass settings and a fake service to avoid external dependencies.
    """
    configure_logging()
    settings = settings or Settings.load()
    service = service or AgentService(settings)

    app = FastAPI(title="Deep Research Agent")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.post("/run")
    def run(req: RunRequest) -> dict[str, Any]:
        thread_id = req.thread_id or str(uuid.uuid4())

        ensure_thread_dir(settings.runs_dir, thread_id)

        user_msg = req.question.strip()
        if req.urls:
            user_msg += "\n\nSources (use fetch_url on these):\n" + "\n".join(
                f"- {u}" for u in req.urls
            )

        agent = service.build_agent(thread_id)

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_msg}]},
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as e:
            log.exception("agent.invoke failed")
            warnings = ensure_required_artifacts(settings.runs_dir, thread_id)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": f"{type(e).__name__}: {e}",
                    "thread_id": thread_id,
                    "warnings": warnings,
                },
            )

        summary_text = None
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            last = result["messages"][-1]
            if isinstance(last, dict):
                summary_text = last.get("content")
            else:
                summary_text = getattr(last, "content", None)

        warnings = ensure_required_artifacts(settings.runs_dir, thread_id)

        return {
            "thread_id": thread_id,
            "summary": summary_text,
            "warnings": warnings,
            "artifacts": [a.__dict__ for a in list_artifacts(settings.runs_dir, thread_id)],
            "hint": f"Report should be at runs/{thread_id}/report.md",
        }

    @app.get("/threads/{thread_id}/artifacts")
    def artifacts(thread_id: str) -> list[dict[str, Any]]:
        return [a.__dict__ for a in list_artifacts(settings.runs_dir, thread_id)]

    @app.get("/threads/{thread_id}/artifacts/{rel_path:path}")
    def artifact_download(thread_id: str, rel_path: str):
        try:
            ap = artifact_abs_path(settings.runs_dir, thread_id, rel_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not ap.exists() or ap.is_dir():
            raise HTTPException(status_code=404, detail="Not found")

        return FileResponse(str(ap))

    return app


app = create_app()