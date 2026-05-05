from __future__ import annotations

import json
from pathlib import Path

from deep_research_agent.retrieval import (
    HybridRankingConfig,
    MockEmbeddingProvider,
    build_retrieval_index,
    plan_retrieval_queries,
    rank_retrieval_results,
    rebuild_retrieval_artifacts,
)
from deep_research_agent.retrieval.artifact_writer import write_retrieval_artifacts
from deep_research_agent.retrieval.context_pack import build_context_pack
from deep_research_agent.retrieval.contracts import RetrievalChunk, RetrievalQuery
from deep_research_agent.retrieval.embeddings import cosine_similarity
from deep_research_agent.retrieval.indexer import extract_dates, extract_numbers
from deep_research_agent.retrieval.lexical import LexicalIndex, phrase_match_score, tokenize
from deep_research_agent.source_intelligence.dedupe import content_hash


def _write_source_run(run_dir: Path, thread_id: str = "retrieval-test") -> None:
    sources_dir = run_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    source_text = (
        "# Revenue Outlook\n\n"
        "Acme Corp reported 42% revenue growth in 2025. The annual report says "
        "cash flow improved after the WidgetPro launch.\n\n"
        "# Risk Factors\n\n"
        "The filing warns that supply chain failures and regulatory delays could "
        "reduce margins in 2026."
    )
    competitor_text = (
        "# Market View\n\n"
        "Beta Inc reported 12% revenue growth in 2025. Analysts said WidgetPro "
        "faces pricing pressure, but Acme Corp has stronger cash generation."
    )
    (sources_dir / "acme.txt").write_text(source_text, encoding="utf-8")
    (sources_dir / "beta.txt").write_text(competitor_text, encoding="utf-8")
    manifest = [
        {
            "ok": True,
            "skipped": False,
            "url": "https://example.com/acme-annual-report",
            "final_url": "https://example.com/acme-annual-report",
            "title": "Acme Annual Report",
            "source_id": "S1",
            "local_path": f"runs/{thread_id}/sources/acme.txt",
            "word_count": 38,
            "char_count": len(source_text),
            "content_hash": content_hash(source_text),
        },
        {
            "ok": True,
            "skipped": False,
            "url": "https://market.example/beta-view",
            "final_url": "https://market.example/beta-view",
            "title": "Beta Market View",
            "source_id": "S2",
            "local_path": f"runs/{thread_id}/sources/beta.txt",
            "word_count": 24,
            "char_count": len(competitor_text),
            "content_hash": content_hash(competitor_text),
        },
    ]
    (run_dir / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit = {
        "audits": [
            {
                "source_id": "S1",
                "final_source_score": 0.91,
                "authority_score": {"source_role": "primary"},
                "primary_source_likelihood": {"likelihood": 0.95},
                "freshness_score": {"status": "current"},
                "citation_readiness_score": {"score": 0.88, "citation_ready": True},
                "recommended_usage": "cite_directly",
                "warnings": [],
            },
            {
                "source_id": "S2",
                "final_source_score": 0.48,
                "authority_score": {"source_role": "secondary"},
                "primary_source_likelihood": {"likelihood": 0.2},
                "freshness_score": {"status": "recent"},
                "citation_readiness_score": {"score": 0.55, "citation_ready": True},
                "recommended_usage": "use_as_background",
                "warnings": [{"message": "Secondary commentary."}],
            },
        ]
    }
    (run_dir / "source_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def test_tokenization_removes_stopwords_and_normalizes():
    assert tokenize("The Acme Corp's revenue, and cash-flow!") == [
        "acme",
        "corp",
        "revenue",
        "cash-flow",
    ]


def test_lexical_scoring_and_phrase_matching():
    chunk = RetrievalChunk(
        chunk_id="c1",
        source_id="S1",
        url="https://example.com",
        title="Annual Report",
        text="Acme revenue growth improved cash flow after WidgetPro launch.",
        start_offset=0,
        end_offset=62,
        content_hash="h",
    )
    index = LexicalIndex([chunk])
    hits = index.search("Acme cash flow", limit=5)
    assert hits
    assert hits[0].score > 0
    phrase_score, phrases = phrase_match_score(chunk.text, ["cash flow"])
    assert phrase_score > 0
    assert phrases == ["cash flow"]


def test_entity_numeric_and_date_overlap(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_source_run(run_dir)
    index = build_retrieval_index(run_dir, thread_id="retrieval-test")
    assert index.chunks
    chunk = next(c for c in index.chunks if c.source_id == "S1")
    document = next(d for d in index.documents if d.source_id == "S1")
    assert document.document_id.startswith("D-")
    assert document.document_identity.source_id == "S1"
    assert chunk.document_id == document.document_id
    assert chunk.chunk_id.startswith("C-")
    assert chunk.chunk_identity.document_id == document.document_id
    assert "Acme Corp" in chunk.entities
    assert "42%" in chunk.numbers
    assert "2025" in chunk.dates
    assert extract_numbers("Growth was 42%") == ["42%"]
    assert "2026" in extract_dates("Updated in 2026")


def test_hybrid_ranking_diversity_and_source_quality_boost(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_source_run(run_dir)
    index = build_retrieval_index(run_dir, thread_id="retrieval-test")
    query = RetrievalQuery(
        query_id="q1",
        text="Acme Corp revenue growth 2025 annual report",
        entities=["Acme Corp"],
        dates=["2025"],
        numbers=["42%"],
        freshness_required=True,
    )
    results = rank_retrieval_results(
        index,
        query,
        config=HybridRankingConfig(top_k=4, max_chunks_per_source=1),
    )
    assert results[0].chunk.source_id == "S1"
    assert results[0].score.source_quality_score > results[-1].score.source_quality_score
    assert len({result.chunk.source_id for result in results}) >= 2
    assert any("source quality" in reason for reason in results[0].score.reasons)


def test_context_pack_respects_size_limits(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_source_run(run_dir)
    build = rebuild_retrieval_artifacts(
        run_dir,
        thread_id="retrieval-test",
        question="Compare Acme Corp revenue growth and risks in 2025",
        write_artifacts=False,
    )
    pack = build_context_pack(
        pack_type="agent_context_pack",
        question=build.question,
        queries=build.queries,
        results=build.results,
        index=build.index,
        max_chars=450,
    )
    assert pack.total_chars <= 450
    assert pack.items
    assert pack.items[0].citation_hint


def test_query_planning_generates_expected_query_types():
    queries = plan_retrieval_queries(
        "Compare Acme Corp and Beta Inc current risks, citations, and 2025 revenue",
        subquestions=["What risks did Acme Corp disclose?"],
    )
    query_types = {query.query_type for query in queries}
    assert {"main", "subquestion", "entity", "comparison", "risk", "freshness"}.issubset(
        query_types
    )


def test_mock_embedding_provider_is_deterministic():
    provider = MockEmbeddingProvider(dimensions=16)
    first = provider.embed_texts(["Acme revenue growth"])[0]
    second = provider.embed_texts(["Acme revenue growth"])[0]
    other = provider.embed_texts(["supply chain risk"])[0]
    assert first == second
    assert cosine_similarity(first, second) > cosine_similarity(first, other)


def test_artifact_writer_persists_retrieval_files(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_source_run(run_dir)
    build = rebuild_retrieval_artifacts(
        run_dir,
        thread_id="retrieval-test",
        question="Acme Corp revenue growth",
        write_artifacts=False,
    )
    paths = write_retrieval_artifacts(run_dir, build)
    assert "retrieval_index.json" in paths
    assert (run_dir / "context_packs.md").exists()
    assert (run_dir / "retrieval_coverage.md").exists()


def test_api_retrieval_search_and_rebuild_routes(client, test_runs_dir: Path):
    response = client.post(
        "/run",
        json={
            "question": "Compare Acme Corp and Beta Inc revenue growth",
            "mock_mode": True,
            "urls": ["https://example.com/acme"],
        },
    )
    assert response.status_code == 200
    thread_id = response.json()["thread_id"]
    run_dir = test_runs_dir / thread_id
    _write_source_run(run_dir, thread_id=thread_id)

    rebuild = client.post(f"/runs/{thread_id}/retrieval/rebuild")
    assert rebuild.status_code == 200
    assert rebuild.json()["index"]["chunks"] >= 2

    search = client.post(
        "/retrieval/search",
        json={"thread_id": thread_id, "query": "Acme Corp 42% revenue growth", "top_k": 3},
    )
    assert search.status_code == 200
    assert search.json()["results"]

    packs = client.get(f"/runs/{thread_id}/context-packs")
    assert packs.status_code == 200
    assert "agent_context_pack" in packs.json()["packs"]

    results = client.get(f"/runs/{thread_id}/retrieval-results")
    assert results.status_code == 200
    assert results.json()["results"]
