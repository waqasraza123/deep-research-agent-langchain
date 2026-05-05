from __future__ import annotations

import json
from pathlib import Path

from deep_research_agent.artifacts import ensure_thread_dir
from deep_research_agent.memory import (
    ArtifactReference,
    ExtractedTopic,
    MemoryRecord,
    MemoryRepository,
    MemoryRetriever,
    SourceCache,
    build_memory_graph,
    extract_entities_and_topics,
    write_memory_context_artifacts,
)
from deep_research_agent.memory.repository import normalize_question
from deep_research_agent.runs.repository import RunRepository
from deep_research_agent.source_intelligence.dedupe import content_hash, normalize_url


def _record(
    *,
    thread_id: str = "t1",
    question: str = "How should LangGraph memory use SQLite for research reuse?",
    url: str = "https://example.com/docs?utm_source=test",
    title: str = "LangGraph Memory Docs",
    last_seen_at: str = "2026-05-01T00:00:00Z",
) -> MemoryRecord:
    text = "LangGraph and SQLite can persist reusable research memory for FastAPI agents."
    normalized = normalize_url(url)
    extraction = extract_entities_and_topics(question=question, text=text, title=title, url=url)
    return MemoryRecord(
        memory_id=f"{thread_id}-{content_hash(text)}",
        thread_id=thread_id,
        question=question,
        normalized_question=normalize_question(question),
        source_url=url,
        normalized_url=normalized,
        canonical_url="https://example.com/docs",
        source_title=title,
        source_domain="example.com",
        content_hash=content_hash(text),
        extracted_text_hash=content_hash(text),
        source_type="html",
        first_seen_at=last_seen_at,
        last_seen_at=last_seen_at,
        run_count=1,
        quality_score=0.82,
        entities=extraction.entities,
        topics=extraction.topics or [ExtractedTopic(name="research memory", score=0.8)],
        summary=text,
        warnings=[],
        artifacts=[
            ArtifactReference(
                thread_id=thread_id,
                path="sources/source.txt",
                artifact_type="source",
            )
        ],
    )


def test_sqlite_memory_persistence_round_trip(tmp_path: Path):
    repo = MemoryRepository(tmp_path / "memory")
    repo.upsert(_record())

    reopened = MemoryRepository(tmp_path / "memory")
    results = reopened.search("LangGraph SQLite research memory")

    assert len(results) == 1
    assert results[0].source_domain == "example.com"
    assert any(entity.name == "LangGraph" for entity in results[0].entities)


def test_source_reuse_normalized_canonical_content_and_stale(tmp_path: Path):
    repo = MemoryRepository(tmp_path / "memory")
    repo.upsert(_record(last_seen_at="2020-01-01T00:00:00Z"))
    cache = SourceCache(repo, stale_after_days=30)

    normalized_decision = cache.evaluate(url="https://example.com/docs")
    assert normalized_decision.reuse_reason == "normalized_url_match"
    assert normalized_decision.reuse_allowed is False
    assert normalized_decision.freshness_warning

    canonical_decision = cache.evaluate(
        url="https://example.com/other",
        canonical_url="https://example.com/docs",
    )
    assert canonical_decision.reuse_reason == "canonical_url_match"

    text = "LangGraph and SQLite can persist reusable research memory for FastAPI agents."
    content_decision = cache.evaluate(url="https://other.test/copy", text=text)
    assert content_decision.reuse_reason == "content_hash_match"


def test_entity_and_topic_extraction_is_deterministic():
    text = (
        "Dr Alice Smith at OpenAI compared LangGraph, FastAPI, and SQLite in California "
        "on May 1, 2026. The policy mentions GDPR, privacy policy, $1.2 million, and 42%."
    )
    result = extract_entities_and_topics(
        question="Assess LangGraph policy memory for FastAPI agents",
        title="OpenAI LangGraph Memory Policy",
        text=text,
        url="https://openai.com/research",
    )

    names = {entity.name for entity in result.entities}
    types = {entity.entity_type.value for entity in result.entities}
    assert "Alice Smith" in names
    assert "LangGraph" in names
    assert "GDPR" in names
    assert "May 1, 2026" in names
    assert "42%" in names
    assert "framework_library" in types
    assert any("policy" in topic.name for topic in result.topics)


def test_memory_graph_links_repeated_sources_across_runs(tmp_path: Path):
    repo = MemoryRepository(tmp_path / "memory")
    repo.upsert(_record(thread_id="t1"))
    repo.upsert(_record(thread_id="t2", question="What changed in LangGraph memory?"))

    graph = build_memory_graph(repo.list_records())

    assert any(node.type == "source" for node in graph.nodes)
    assert any(edge.relation == "repeated_across_runs" for edge in graph.edges)


def test_memory_retrieval_and_context_artifact_writing(tmp_path: Path):
    repo = MemoryRepository(tmp_path / "memory")
    repo.upsert(_record())
    context = MemoryRetriever(repo).retrieve(
        question="Can FastAPI reuse prior LangGraph SQLite research memory?",
        urls=["https://example.com/docs"],
    )

    assert context.confidence_score > 0
    assert context.similar_previous_questions
    assert context.known_topics
    assert context.suggested_source_reuse_candidates[0].confidence_score > 0

    write_memory_context_artifacts(tmp_path, context)
    payload = json.loads((tmp_path / "memory_context.json").read_text(encoding="utf-8"))
    assert payload["confidence_score"] == context.confidence_score
    assert "prior context" in (tmp_path / "memory_context.md").read_text(encoding="utf-8").lower()


def test_memory_rebuild_route_indexes_existing_run_artifacts(client, test_runs_dir: Path):
    RunRepository(test_runs_dir).create(
        thread_id="memory-rebuild",
        question="How does LangGraph persist research memory?",
        urls=["https://example.com/docs"],
        settings_snapshot={"model_provider": "test"},
    )

    td = ensure_thread_dir(test_runs_dir, "memory-rebuild")
    source_dir = td / "sources"
    source_dir.mkdir(exist_ok=True)
    source_path = source_dir / "source.txt"
    source_path.write_text(
        "LangGraph uses durable checkpoints and SQLite can store research memory for agents.",
        encoding="utf-8",
    )
    (td / "sources.json").write_text(
        json.dumps(
            [
                {
                    "url": "https://example.com/docs",
                    "normalized_url": "https://example.com/docs",
                    "source_kind": "root",
                    "ok": True,
                    "skipped": False,
                    "final_url": "https://example.com/docs",
                    "canonical_url": "https://example.com/docs",
                    "title": "LangGraph SQLite Memory",
                    "local_path": "runs/memory-rebuild/sources/source.txt",
                    "document_kind": "html",
                }
            ]
        ),
        encoding="utf-8",
    )

    rebuilt = client.post("/runs/memory-rebuild/memory/rebuild")
    assert rebuilt.status_code == 200
    body = rebuilt.json()
    assert body["stored_memories"]
    assert any(artifact["path"] == "memory_graph.json" for artifact in body["artifacts"])

    search = client.get("/memory/search", params={"q": "LangGraph SQLite memory"})
    assert search.status_code == 200
    assert search.json()
