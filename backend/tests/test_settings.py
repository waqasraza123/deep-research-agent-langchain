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
    assert settings.memory_enabled is True
    assert settings.source_reuse_enabled is True
    assert settings.source_audit_enabled is True
    assert settings.orchestration_enabled is True
    assert settings.synthesis_enabled is True
    assert settings.evaluation_enabled is True
    assert settings.protocol_selection_enabled is True
    assert settings.intelligence_profile == "balanced_research"
    assert settings.source_discovery_enabled is False
    assert settings.source_discovery_provider == "disabled"
    assert settings.max_discovery_queries == 8
    assert settings.max_selected_discovered_sources == 3
    assert settings.document_intelligence_enabled is True
    assert settings.chunk_max_chars == 3200
    assert settings.chunk_overlap_chars == 300
    assert settings.retrieval_enabled is True
    assert settings.embedding_provider == "disabled"
    assert settings.context_pack_max_chars == 11_000
    assert settings.verification_enabled is True
    assert settings.verification_gate_enabled is False
    assert settings.max_verification_tasks == 12
    assert settings.confidence_threshold_for_review == 0.55
    assert settings.max_memory_results == 20
    assert settings.source_scoring_threshold == 0.45
    assert settings.evaluation_threshold == 0.65
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
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    monkeypatch.setenv("MAX_MEMORY_RESULTS", "7")
    monkeypatch.setenv("PROTOCOL_SELECTION_ENABLED", "false")
    monkeypatch.setenv("INTELLIGENCE_PROFILE", "offline_mock")
    monkeypatch.setenv("SOURCE_DISCOVERY_ENABLED", "true")
    monkeypatch.setenv("SOURCE_DISCOVERY_PROVIDER", "mock")
    monkeypatch.setenv("MAX_DISCOVERY_QUERIES", "4")
    monkeypatch.setenv("MAX_SELECTED_DISCOVERED_SOURCES", "2")
    monkeypatch.setenv("DOCUMENT_INTELLIGENCE_ENABLED", "false")
    monkeypatch.setenv("CHUNK_MAX_CHARS", "900")
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", "80")
    monkeypatch.setenv("RETRIEVAL_ENABLED", "false")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("CONTEXT_PACK_MAX_CHARS", "5000")
    monkeypatch.setenv("VERIFICATION_ENABLED", "false")
    monkeypatch.setenv("VERIFICATION_GATE_ENABLED", "true")
    monkeypatch.setenv("MAX_VERIFICATION_TASKS", "6")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD_FOR_REVIEW", "0.72")

    settings = Settings.load()

    assert settings.model_provider == "mock"
    assert settings.checkpoint_path is not None
    assert settings.default_follow_links is True
    assert settings.default_max_links_per_source == 3
    assert settings.review_gate_default is True
    assert settings.memory_enabled is False
    assert settings.max_memory_results == 7
    assert settings.protocol_selection_enabled is False
    assert settings.intelligence_profile == "offline_mock"
    assert settings.source_discovery_enabled is True
    assert settings.source_discovery_provider == "mock"
    assert settings.source_discovery_max_queries == 4
    assert settings.max_discovery_queries == 4
    assert settings.source_discovery_max_selected_sources == 2
    assert settings.max_selected_discovered_sources == 2
    assert settings.document_intelligence_enabled is False
    assert settings.chunk_max_chars == 900
    assert settings.chunk_overlap_chars == 80
    assert settings.retrieval_enabled is False
    assert settings.embedding_provider == "mock"
    assert settings.context_pack_max_chars == 5000
    assert settings.verification_enabled is False
    assert settings.verification_gate_enabled is True
    assert settings.max_verification_tasks == 6
    assert settings.confidence_threshold_for_review == 0.72
    assert settings.default_budget().max_crawl_expansion == 2
