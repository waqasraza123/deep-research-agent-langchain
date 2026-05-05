from __future__ import annotations

import json
from pathlib import Path

from deep_research_agent.document_intelligence import (
    ChunkingConfig,
    build_document_intelligence_batch,
    profile_document,
    write_document_intelligence_artifacts,
)
from deep_research_agent.document_intelligence.chunker import chunk_document
from deep_research_agent.document_intelligence.citation_extractor import extract_citations
from deep_research_agent.document_intelligence.content_features import detect_content_features
from deep_research_agent.document_intelligence.normalizer import normalize_text
from deep_research_agent.document_intelligence.sectioner import section_document
from deep_research_agent.document_intelligence.table_extractor import extract_tables


def test_text_normalization_removes_boilerplate_and_unicode_spacing():
    raw = (
        "Header\n\nHello\u00a0 world\n\nMenu\n\n"
        "Repeated boilerplate block that is long enough.\n\n"
        "Repeated boilerplate block that is long enough.\n"
    )
    result = normalize_text(raw, source_id="S1", source_type="html")
    assert "Hello world" in result.normalized_text
    assert "Menu" not in result.normalized_text
    assert result.normalized_text.count("Repeated boilerplate") == 1


def test_pdf_line_wrap_cleanup_and_hyphenation():
    raw = (
        "Annual Report\nThis is a long sen-\ntence that should\ncontinue naturally."
        "\n\nConclusion\nDone."
    )
    result = normalize_text(raw, source_id="S1", source_type="pdf")
    assert "sentence" in result.normalized_text
    assert "should continue naturally" in result.normalized_text


def test_heading_detection_and_section_hierarchy():
    text = "# Overview\nIntro\n\n## Details\nFacts\n\n2.1 Legal Terms\nPolicy text"
    sections = section_document(text, source_id="S1")
    headings = [section.heading for section in sections]
    assert headings[:3] == ["Overview", "Details", "Legal Terms"]
    details = sections[1]
    assert details.parent_section_id == sections[0].section_id
    assert details.path == ["Overview", "Details"]


def test_chunk_boundaries_and_stable_ids():
    text = (
        "# Overview\n" + ("Alpha beta gamma. " * 120) + "\n\n# Details\n" + ("Delta 2025. " * 120)
    )
    sections = section_document(text, source_id="S1")
    chunks1 = chunk_document(
        text,
        source_id="S1",
        sections=sections,
        tables=[],
        config=ChunkingConfig(max_chars=500, overlap_chars=50),
    )
    chunks2 = chunk_document(
        text,
        source_id="S1",
        sections=sections,
        tables=[],
        config=ChunkingConfig(max_chars=500, overlap_chars=50),
    )
    assert len(chunks1) > 2
    assert [chunk.chunk_id for chunk in chunks1] == [chunk.chunk_id for chunk in chunks2]
    assert all(chunk.heading_path for chunk in chunks1)


def test_markdown_table_extraction():
    text = "| Plan | Price |\n| --- | ---: |\n| Pro | $20 |\n| Team | $50 |\n"
    tables = extract_tables(text, source_id="S1", source_type="md")
    assert len(tables) == 1
    assert tables[0].rows[0] == ["Plan", "Price"]
    assert "$20" in tables[0].readable_text


def test_csv_table_conversion():
    text = "name,score,date\nAlpha,42,2025-01-01\nBeta,55,2025-02-01\n"
    tables = extract_tables(text, source_id="S1", source_type="csv")
    assert len(tables) == 1
    assert tables[0].table_kind == "csv"
    assert tables[0].rows[1][0] == "Alpha"


def test_citation_detection():
    text = "See https://example.com/a and DOI 10.1000/xyz123.\n[1] Smith, J. (2024). Useful paper."
    citations, footnotes = extract_citations(text, source_id="S1")
    assert {citation.citation_type for citation in citations} >= {"url", "doi"}
    assert footnotes and footnotes[0].marker == "1"


def test_content_feature_detection():
    text = "Pricing starts at $20. API endpoint examples use curl. References include https://example.com."
    features = {feature.name: feature for feature in detect_content_features(text)}
    assert features["has_pricing"].present
    assert features["has_api_docs"].present
    assert features["has_references"].present


def test_artifact_writing(tmp_path: Path):
    source_path = tmp_path / "sources" / "s1.txt"
    source_path.parent.mkdir()
    source_path.write_text("# Overview\nFacts with $20.\n", encoding="utf-8")
    batch = build_document_intelligence_batch(
        thread_dir=tmp_path,
        thread_id="t1",
        sources=[
            {
                "ok": True,
                "url": "https://example.com",
                "source_id": "S1",
                "local_path": "runs/t1/sources/s1.txt",
                "document_kind": "md",
            }
        ],
    )
    paths = write_document_intelligence_artifacts(tmp_path, batch)
    assert "document_profiles.json" in paths
    assert (tmp_path / "document_profiles.md").exists()
    assert (tmp_path / "document_chunks.jsonl").read_text(encoding="utf-8").strip()
    payload = json.loads((tmp_path / "document_profiles.json").read_text(encoding="utf-8"))
    assert payload["profiles"][0]["source_id"] == "S1"


def test_profile_route(client):
    response = client.post(
        "/document-intelligence/profile",
        json={
            "raw_text": "# Intro\nThe API costs $20. See https://example.com/docs.\n",
            "source_id": "S1",
            "url": "https://example.com/docs",
            "source_type": "md",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source_id"] == "S1"
    assert body["sections"]
    assert body["chunks"]
    assert any(
        feature["name"] == "has_pricing" and feature["present"] for feature in body["features"]
    )


def test_profile_document_end_to_end():
    profile = profile_document(
        raw_text=(
            "# Intro\nThe API costs $20.\n\n"
            "| Item | Value |\n| --- | --- |\n| Speed | 10ms |\n"
        ),
        source={"source_id": "S1", "url": "https://example.com", "document_kind": "md"},
    )
    assert profile.sections
    assert profile.chunks
    assert profile.tables
    assert profile.quality_summary["structure_score"] > 0
