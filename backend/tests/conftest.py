from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from deep_research_agent.api import create_app
from deep_research_agent.settings import Settings
from deep_research_agent.artifacts import ensure_thread_dir


class FakeAgent:
    def __init__(self, runs_dir: Path, thread_id: str, should_fail: bool = False):
        self.runs_dir = runs_dir
        self.thread_id = thread_id
        self.should_fail = should_fail

    def invoke(self, *_args, **_kwargs):
        if self.should_fail:
            raise RuntimeError("boom")

        td = ensure_thread_dir(self.runs_dir, self.thread_id)

        (td / "plan.md").write_text("# Plan\n\n- Step 1\n- Step 2\n", encoding="utf-8")
        (td / "notes.md").write_text("# Notes\n\n- Note A\n", encoding="utf-8")
        (td / "sources.json").write_text(
            json.dumps([{"url": "https://example.com", "summary": "Example"}], indent=2),
            encoding="utf-8",
        )
        (td / "report.md").write_text(
            "# Report\n\nThis is a test report.\n", encoding="utf-8"
        )

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Done. Report at runs/{self.thread_id}/report.md",
                }
            ]
        }


class FakeService:
    def __init__(self, runs_dir: Path, should_fail: bool = False):
        self.runs_dir = runs_dir
        self.should_fail = should_fail

    def build_agent(self, thread_id: str, *args, **kwargs):
        return FakeAgent(self.runs_dir, thread_id, should_fail=self.should_fail)


@pytest.fixture()
def test_runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def test_settings(test_runs_dir: Path) -> Settings:
    return Settings(
        model_provider="ollama",
        ollama_model="llama3.1",
        temperature=0.2,
        runs_dir=test_runs_dir,
        max_page_chars=50_000,
        http_timeout_s=5.0,
        host="127.0.0.1",
        port=8000,
    )


@pytest.fixture()
def client(test_settings: Settings, test_runs_dir: Path) -> TestClient:
    app = create_app(settings=test_settings, service=FakeService(test_runs_dir))
    return TestClient(app)


@pytest.fixture()
def failing_client(test_settings: Settings, test_runs_dir: Path) -> TestClient:
    app = create_app(settings=test_settings, service=FakeService(test_runs_dir, should_fail=True))
    return TestClient(app)