from __future__ import annotations

import sqlite3
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langgraph.checkpoint.sqlite import SqliteSaver

from .model import create_chat_model
from .settings import Settings
from .tools import fetch_url


class AgentService:
    """
    Keeps shared resources (checkpointer) and builds agents per thread_id.
    Production-grade: one sqlite checkpointer shared, thread_id isolates state.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.runs_dir.mkdir(parents=True, exist_ok=True)

        db_path = self.settings.runs_dir / "checkpoints.sqlite"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._checkpointer = SqliteSaver(conn)

    def _backend(self, rt):
        return CompositeBackend(
            default=StateBackend(rt),
            routes={
                "/runs/": FilesystemBackend(
                    root_dir=str(self.settings.runs_dir),
                    virtual_mode=True,
                ),
            },
        )

    def build_agent(self, thread_id: str):
        run_dir = f"/runs/{thread_id}"

        system_prompt = f"""\
You are an expert research analyst.

You MUST write working artifacts to: {run_dir}

Deliverables (always):
1) {run_dir}/plan.md
2) {run_dir}/notes.md
3) {run_dir}/sources.json as a JSON array of objects: {{ "url": "...", "summary": "..." }}
4) {run_dir}/report.md a polished Markdown report with citations

Workflow:
- First create a brief plan in plan.md (bullets).
- Use fetch_url for provided sources and summarize into notes.md.
- Build sources.json from what you read.
- Write report.md. Keep it clean and client-ready.
- If sources are insufficient, state that clearly and label assumptions.

Return in chat:
- A short executive summary (5-10 lines) and the path to report.md.
"""

        # Wrap fetch_url so it uses runtime config
        def fetch(url: str) -> str:
            return fetch_url(
                url,
                timeout_s=self.settings.http_timeout_s,
                max_chars=self.settings.max_page_chars,
            )

        agent = create_deep_agent(
            model=create_chat_model(self.settings),
            tools=[fetch],
            system_prompt=system_prompt,
            backend=self._backend,
            checkpointer=self._checkpointer,
        )
        return agent