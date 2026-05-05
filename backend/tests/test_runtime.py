from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_research_agent.runtime.budgets import BudgetExceeded, BudgetTracker
from deep_research_agent.runtime.contracts import RetryPolicy, RunBudget
from deep_research_agent.runtime.events import RunEventLogger
from deep_research_agent.runtime.mock_model import (
    DeterministicMockChatModel,
    write_mock_research_artifacts,
)
from deep_research_agent.runtime.model_registry import build_model_registry
from deep_research_agent.runtime.retries import classify_error, retry_sync
from deep_research_agent.settings import Settings


def test_model_registry_reports_configured_providers(tmp_path: Path):
    settings = Settings(model_provider="mock", runs_dir=tmp_path)
    models = build_model_registry(settings)

    providers = {m.provider for m in models}
    assert {"openai", "ollama", "llamacpp", "mock"} <= providers
    mock = next(m for m in models if m.provider == "mock")
    assert mock.requires_api_key is False
    assert mock.health_status == "mock"


def test_mock_model_is_deterministic_and_marks_output():
    model = DeterministicMockChatModel()
    payload = [{"role": "user", "content": "What should we test?\nMore context"}]

    first = model.invoke(payload).content
    second = model.invoke(payload).content

    assert first == second
    assert "[MOCK OUTPUT]" in first
    assert "What should we test?" in first


def test_mock_artifacts_are_marked(tmp_path: Path):
    metadata = write_mock_research_artifacts(
        thread_dir=tmp_path,
        thread_id="tid",
        question="Explain runtime tracing.",
        sources_meta=[],
    )

    assert metadata["mock"] is True
    assert "MOCK OUTPUT" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["mock"] is True


def test_budget_usage_incrementing_and_exceeded_behavior():
    tracker = BudgetTracker(RunBudget(max_model_calls=1))

    tracker.increment_model_calls()
    assert tracker.snapshot().model_calls == 1

    with pytest.raises(BudgetExceeded) as exc:
        tracker.increment_model_calls()
    assert "model_calls" in exc.value.reasons[0]


def test_retry_classification_and_attempts():
    assert classify_error(ValueError("bad request")) == "nonretryable"
    assert classify_error(TimeoutError("timeout")) == "retryable"

    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("timeout")
        return "ok"

    result, meta = retry_sync(flaky, RetryPolicy(max_attempts=2, initial_backoff_s=0))
    assert result == "ok"
    assert meta["attempts"] == 2


def test_event_writer_persists_jsonl_and_markdown(tmp_path: Path):
    logger = RunEventLogger(thread_id="tid", thread_dir=tmp_path)
    logger.log("run_started", message="hello", metadata={"provider": "mock"})
    logger.log("run_completed", message="done")

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "run_started"
    markdown = (tmp_path / "events.md").read_text(encoding="utf-8")
    assert "run_completed" in markdown
