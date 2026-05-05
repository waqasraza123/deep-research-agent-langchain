from __future__ import annotations

import json
from pathlib import Path

from deep_research_agent.evidence.artifact_writer import rebuild_evidence_artifacts
from deep_research_agent.evidence.citation_mapper import SourceDocument, map_claim_citations
from deep_research_agent.evidence.claim_extractor import ClaimInput, extract_claims
from deep_research_agent.evidence.confidence import score_claims
from deep_research_agent.evidence.contracts import EvidenceSource
from deep_research_agent.evidence.contradiction import detect_contradictions


def test_claim_extraction_from_report_text():
    claims = extract_claims(
        [
            ClaimInput(
                origin="report",
                text=(
                    "# Report\n\n"
                    "- Acme Search supports private indexes for enterprise customers. [S1]\n"
                    "- Teams should avoid sending secrets to unsupported connectors.\n"
                ),
            )
        ]
    )

    assert len(claims) == 2
    assert claims[0].claim_type == "factual"
    assert claims[1].claim_type == "recommendation"
    assert claims[0].origin == "report"


def test_numeric_and_date_claim_detection():
    claims = extract_claims(
        [
            ClaimInput(
                origin="notes",
                text=(
                    "Revenue increased by 24% in 2025. "
                    "The migration completed on March 4, 2024."
                ),
            )
        ]
    )

    assert [claim.claim_type for claim in claims] == ["numeric", "numeric"]
    assert any("numeric" in " ".join(claim.notes).lower() for claim in claims)


def test_citation_candidate_scoring_uses_keywords_and_values():
    claim = extract_claims(
        [
            ClaimInput(
                origin="report",
                text="Acme Search latency decreased by 35% after the cache rollout.",
            )
        ]
    )[0]
    source = SourceDocument(
        source=EvidenceSource(
            source_id="S1",
            url="https://example.com/acme",
            title="Acme Search cache rollout",
            domain="example.com",
            quality_score=0.9,
        ),
        text="The Acme Search cache rollout reduced latency by 35% for production users.",
    )

    citation_map, quotes = map_claim_citations([claim], [source], threshold=0.25)

    citation = citation_map[claim.claim_id][0]
    assert citation.score >= 0.34
    assert "35%" in citation.value_matches
    assert citation.source_id == "S1"
    assert quotes


def test_unsupported_claim_detection_for_missing_evidence():
    claim = extract_claims(
        [ClaimInput(origin="report", text="The product always guarantees zero downtime.")]
    )[0]
    claims, _confidences, unsupported = score_claims([claim], [])

    assert claims[0].support_level == "unsupported"
    assert unsupported
    assert "No source citation candidate" in unsupported[0].reason


def test_contradiction_grouping_for_conflicting_numbers():
    claims = extract_claims(
        [
            ClaimInput(origin="report", text="Acme Search reduced latency by 35% in 2025."),
            ClaimInput(
                origin="source",
                text="Acme Search reduced latency by 12% in 2025.",
                source_ids=("S1",),
            ),
        ]
    )

    groups = detect_contradictions(claims)

    assert groups
    assert groups[0].contradiction_type == "conflicting_values"
    assert groups[0].severity == "high"


def test_confidence_scoring_rewards_support_and_penalizes_contradictions():
    claim = extract_claims(
        [ClaimInput(origin="report", text="Acme Search reduced latency by 35% in 2025.")]
    )[0]
    source = SourceDocument(
        source=EvidenceSource(source_id="S1", quality_score=0.9),
        text="Acme Search reduced latency by 35% in 2025 after cache changes.",
    )
    citation_map, _quotes = map_claim_citations([claim], [source], threshold=0.25)
    claim.citations = citation_map[claim.claim_id]

    claims, _confidences, unsupported = score_claims([claim], [source.source])
    assert claims[0].support_level in {"moderate", "strong"}
    assert not unsupported

    claims[0].contradiction_ids = ["K1"]
    claims, _confidences, unsupported = score_claims(claims, [source.source])
    assert claims[0].support_level == "contradicted"
    assert unsupported


def test_artifact_writing_builds_expected_files(tmp_path: Path):
    run_dir = tmp_path / "runs" / "t1"
    sources_dir = run_dir / "sources"
    sources_dir.mkdir(parents=True)
    source_text = "Acme Search reduced latency by 35% in 2025 after a cache rollout."
    (sources_dir / "s1.txt").write_text(source_text, encoding="utf-8")
    (run_dir / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": "S1",
                    "url": "https://example.com/acme",
                    "title": "Acme Search",
                    "local_path": "runs/t1/sources/s1.txt",
                    "ok": True,
                    "word_count": 500,
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "notes.md").write_text(
        "- Acme Search reduced latency by 35% in 2025.\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        "# Report\n\nAcme Search reduced latency by 35% in 2025. [S1]\n",
        encoding="utf-8",
    )

    ledger = rebuild_evidence_artifacts(run_dir, thread_id="t1")

    assert ledger.coverage.total_claims >= 2
    for name in (
        "evidence_ledger.json",
        "evidence_ledger.md",
        "unsupported_claims.md",
        "contradictions.md",
        "citation_map.json",
        "evidence_coverage.json",
    ):
        assert (run_dir / name).exists()
    coverage = json.loads((run_dir / "evidence_coverage.json").read_text(encoding="utf-8"))
    assert coverage["source_count"] == 1
