from __future__ import annotations

import json

from deep_research_agent.source_discovery import (
    SourceDiscoveryRequest,
    SourceDiscoverySettings,
    build_acquisition_plan,
    execute_source_discovery,
    expand_queries,
    write_source_discovery_artifacts,
)
from deep_research_agent.source_discovery.candidate_ranker import rank_and_select
from deep_research_agent.source_discovery.contracts import SourceCandidate
from deep_research_agent.source_discovery.providers import (
    DisabledSearchProvider,
    MockSearchProvider,
)

QUESTION = "Compare LangGraph and CrewAI for a production research agent backend"


def test_query_expansion_for_comparative_technical_question():
    plan = expand_queries(QUESTION, max_queries=10)
    texts = [q.text for q in plan.queries]

    assert plan.comparative is True
    assert plan.technical is True
    assert "LangGraph CrewAI comparison production agent orchestration" in texts
    assert "LangGraph official docs persistence checkpointing tools" in texts
    assert "CrewAI official docs agents tools memory" in texts
    assert any("failure modes production" in text for text in texts)
    assert any("benchmark evaluation" in text for text in texts)


def test_freshness_sensitive_query_generation():
    plan = expand_queries("What is the latest LangGraph release status in 2026?", max_queries=6)

    assert plan.freshness_required is True
    assert any(q.intent == "recent_current" for q in plan.queries)
    assert any(q.freshness_required for q in plan.queries)


def test_source_type_planning_for_technical_comparison():
    request = SourceDiscoveryRequest(
        question=QUESTION,
        settings=SourceDiscoverySettings(discovery_enabled=True, provider="mock"),
    )
    plan = build_acquisition_plan(request)

    assert "official_docs" in plan.required_source_types
    assert "source_code_repository" in plan.required_source_types
    assert "benchmark_report" in plan.preferred_source_types


def test_provider_interface_disabled_and_mock_results():
    query = expand_queries(QUESTION, max_queries=1).queries[0]

    disabled = DisabledSearchProvider().search(query, max_results=3)
    assert disabled.ok is False
    assert disabled.disabled_reason

    mock = MockSearchProvider().search(query, max_results=5)
    assert mock.ok is True
    assert mock.candidates
    assert any(candidate.provider == "mock" for candidate in mock.candidates)


def test_candidate_ranking_dedupe_and_selection_limits():
    request = SourceDiscoveryRequest(
        question=QUESTION,
        settings=SourceDiscoverySettings(
            discovery_enabled=True,
            provider="mock",
            max_selected_sources=2,
            allow_forums=False,
        ),
    )
    plan = build_acquisition_plan(request)
    candidates = [
        SourceCandidate(
            candidate_id="official",
            url="https://langchain-ai.github.io/langgraph/",
            title="LangGraph Documentation",
            snippet="Official LangGraph docs for persistence and checkpointing.",
            domain="langchain-ai.github.io",
            provider="test",
            query=plan.query_plan.queries[0].text,
            query_id=plan.query_plan.queries[0].query_id,
            query_intent=plan.query_plan.queries[0].intent,
            source_type_hint="official_docs",
            primary_source_likelihood=0.9,
            freshness_hint="current",
            authority_hint=0.9,
        ),
        SourceCandidate(
            candidate_id="official-duplicate",
            url="https://langchain-ai.github.io/langgraph/?utm_source=x",
            title="LangGraph Documentation",
            snippet="Duplicate docs.",
            domain="langchain-ai.github.io",
            provider="test",
            query=plan.query_plan.queries[0].text,
            query_id=plan.query_plan.queries[0].query_id,
            query_intent=plan.query_plan.queries[0].intent,
            source_type_hint="official_docs",
            primary_source_likelihood=0.9,
            freshness_hint="current",
            authority_hint=0.9,
        ),
        SourceCandidate(
            candidate_id="blog",
            url="https://vendor.example/blog/langgraph-crewai-best",
            title="Best LangGraph vs CrewAI Comparison",
            snippet="Promotional comparison blog.",
            domain="vendor.example",
            provider="test",
            query=plan.query_plan.queries[0].text,
            query_id=plan.query_plan.queries[0].query_id,
            query_intent=plan.query_plan.queries[0].intent,
            source_type_hint="tutorial_or_blog",
            primary_source_likelihood=0.2,
            freshness_hint="unknown",
            authority_hint=0.3,
        ),
    ]

    ranked, decisions = rank_and_select(candidates, plan)
    selected = [decision for decision in decisions if decision.decision == "selected"]
    duplicates = [decision for decision in decisions if decision.decision == "duplicate"]

    assert ranked[0].candidate_id == "official"
    assert len(selected) <= 2
    assert duplicates


def test_artifact_writing(tmp_path):
    batch = execute_source_discovery(
        SourceDiscoveryRequest(
            question=QUESTION,
            settings=SourceDiscoverySettings(
                discovery_enabled=True,
                provider="mock",
                max_queries=3,
                max_candidates_per_query=3,
                max_selected_sources=2,
            ),
        )
    )
    written = write_source_discovery_artifacts(tmp_path, batch)

    assert "source_acquisition_plan.json" in written
    assert "source_discovery_summary.md" in written
    payload = json.loads((tmp_path / "source_candidates.json").read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload


def test_api_preview_route(client):
    response = client.post(
        "/source-discovery/preview",
        json={
            "question": QUESTION,
            "settings": {
                "discovery_enabled": True,
                "provider": "mock",
                "max_queries": 3,
                "max_candidates_per_query": 3,
                "max_selected_sources": 2,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["selected_count"] <= 2
    assert body["candidates"]
    assert body["selected_candidates"]


def test_run_writes_discovery_skipped_artifact_when_disabled(client):
    response = client.post(
        "/run",
        json={"question": "Validate source discovery disabled behavior", "mock_mode": True},
    )
    assert response.status_code == 200
    thread_id = response.json()["thread_id"]

    artifact = client.get(f"/runs/{thread_id}/artifacts/source_discovery.md")
    assert artifact.status_code == 200
    assert "Discovery enabled: `False`" in artifact.text
    assert "disabled" in artifact.text.lower()


def test_run_mock_mode_merges_selected_discovered_sources(client):
    response = client.post(
        "/run",
        json={
            "question": QUESTION,
            "mock_mode": True,
            "intelligence_profile_id": "balanced_research",
            "max_sources": 3,
            "source_discovery": {
                "discovery_enabled": True,
                "provider": "mock",
                "max_queries": 4,
                "max_candidates_per_query": 3,
                "max_selected_sources": 2,
            },
        },
    )
    assert response.status_code == 200
    thread_id = response.json()["thread_id"]

    discovery = client.get(f"/runs/{thread_id}/source-discovery")
    assert discovery.status_code == 200
    assert discovery.json()["selection"]["selected_candidates"]

    sources = client.get(f"/runs/{thread_id}/artifacts/sources.json")
    assert sources.status_code == 200
    manifest = json.loads(sources.text)
    assert any(item["source_kind"] == "auto_discovered" for item in manifest)
