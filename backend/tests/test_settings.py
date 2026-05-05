from __future__ import annotations

from pathlib import Path

from deep_research_agent.runtime.contracts import RunBudget
from deep_research_agent.settings import Settings


def test_settings_defaults_are_offline_constructible(tmp_path: Path):
    settings = Settings(runs_dir=tmp_path / "runs")

    assert settings.runs_dir.name == "runs"
    assert settings.model_provider == "openai"
    assert settings.checkpoint_path is None
    assert settings.default_follow_links is False
    assert settings.default_max_links_per_source == 0
    assert settings.allow_mock_fallback is False
    assert settings.review_gate_default is False
    assert settings.evidence_citation_threshold == 0.34
    assert settings.max_page_chars > 0
    assert settings.http_timeout_s > 0

    budget = settings.default_budget()
    assert isinstance(budget, RunBudget)
    assert budget.max_source_fetches >= 0
    assert budget.max_crawl_expansion >= 0


def test_settings_load_accepts_mock_provider(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    monkeypatch.setenv("BUDGET_MAX_CRAWL_EXPANSION", "2")
    monkeypatch.setenv("DEFAULT_FOLLOW_LINKS", "true")
    monkeypatch.setenv("DEFAULT_MAX_LINKS_PER_SOURCE", "3")
    monkeypatch.setenv("REVIEW_GATE_DEFAULT", "true")

    settings = Settings.load()

    assert settings.model_provider == "mock"
    assert settings.checkpoint_path is not None
    assert settings.default_follow_links is True
    assert settings.default_max_links_per_source == 3
    assert settings.review_gate_default is True
    assert settings.default_budget().max_crawl_expansion == 2
