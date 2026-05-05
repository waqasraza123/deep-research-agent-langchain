from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from deep_research_agent.source_audit import audit_sources, write_source_audit_artifacts
from deep_research_agent.source_audit.authority import score_authority
from deep_research_agent.source_audit.bias import analyze_bias_risk
from deep_research_agent.source_audit.citation_readiness import score_citation_readiness
from deep_research_agent.source_audit.contracts import SourceAuditWarning
from deep_research_agent.source_audit.credibility import score_credibility
from deep_research_agent.source_audit.freshness import extract_dates, score_freshness
from deep_research_agent.source_audit.primary_source import assess_primary_source

NOW = datetime(2026, 5, 5, tzinfo=timezone.utc)


def _long_text(prefix: str, *, date_text: str = "Updated 2026-04-01") -> str:
    return (
        f"{prefix}. {date_text}. References include https://example.org/source and DOI: "
        "10.1234/example. Methodology and citations are included. "
        * 70
    )


def test_date_extraction_from_metadata_text_title_and_url():
    dates = extract_dates(
        url="https://example.gov/reports/2025/10/source",
        title="Report published March 3, 2026",
        text="Last modified 2026-04-15. Earlier version 2024-01-01.",
        metadata={"published_at": "2026-02-01T00:00:00Z"},
    )

    assert dates[0].isoformat() == "2026-04-15"
    assert "2026-03-03" in {d.isoformat() for d in dates}
    assert "2026-02-01" in {d.isoformat() for d in dates}


def test_freshness_classification_for_current_and_stale_sources():
    current = score_freshness(
        question="What are the latest 2026 API docs?",
        url="https://docs.example.com/api",
        title="API docs",
        text="Updated 2026-04-15",
        now=NOW,
    )
    stale = score_freshness(
        question="What are the latest 2026 API docs?",
        url="https://docs.example.com/api",
        title="API docs",
        text="Published 2020-01-01",
        now=NOW,
    )

    assert current.status == "current"
    assert current.freshness_matters is True
    assert stale.status == "stale"
    assert stale.score < current.score


def test_credibility_scoring_rewards_official_referenced_content_and_penalizes_spam():
    official = score_credibility(
        url="https://nist.gov/docs/reference",
        title="Official API Reference",
        text=_long_text("Official documentation by Jane Doe"),
        metadata={"word_count": 1200},
    )
    spam = score_credibility(
        url="https://contentfarm.example/best-tools-2026",
        title="Ultimate Best Top 10 Revolutionary Guaranteed Tool Guide",
        text="Buy now. Best best best ultimate revolutionary guaranteed. " * 20,
        metadata={"word_count": 80},
    )

    assert official.score > 0.75
    assert spam.score < official.score
    assert any("SEO" in reason for reason in spam.reasons)


def test_authority_detection_identifies_government_and_weak_blog_sources():
    gov = score_authority(
        url="https://www.sec.gov/rules/final/2026/example.pdf",
        title="Final Rule",
        text=_long_text("Federal register regulation"),
        source_type="pdf",
    )
    blog = score_authority(
        url="https://medium.com/example/ultimate-review",
        title="Ultimate Review",
        text="Opinion blog post",
    )

    assert gov.source_role == "primary"
    assert gov.score > 0.7
    assert blog.score < gov.score
    assert blog.source_role in {"secondary", "weak"}


def test_primary_source_detection_for_github_release_and_secondary_tutorial():
    release = assess_primary_source(
        url="https://github.com/example/project/releases/tag/v1.2.0",
        title="Release v1.2.0",
        text="Release notes for version 1.2.0.",
    )
    tutorial = assess_primary_source(
        url="https://randomblog.example/tutorial/project",
        title="Old tutorial copied from docs",
        text="This tutorial was originally published elsewhere.",
    )

    assert release.source_role == "primary"
    assert release.primary_type == "repository_or_release"
    assert tutorial.likelihood < release.likelihood


def test_bias_language_detection_marks_affiliate_and_vendor_comparison_language():
    risk = analyze_bias_risk(
        url="https://vendor.example/compare/us-vs-competitor",
        title="Best Alternative to Competitor",
        text="We may earn affiliate commission. Our revolutionary product is the only solution.",
    )

    assert risk.risk_level == "high"
    assert any("Affiliate" in signal for signal in risk.signals)


def test_citation_readiness_requires_title_content_date_and_authority():
    freshness = score_freshness(
        question="latest regulation in 2026",
        url="https://example.gov/rules/final",
        title="Final Rule",
        text="Updated 2026-04-01",
        now=NOW,
    )
    authority = score_authority(
        url="https://example.gov/rules/final",
        title="Final Rule",
        text=_long_text("Final rule"),
    )
    primary = assess_primary_source(
        url="https://example.gov/rules/final",
        title="Final Rule",
        text=_long_text("Final rule"),
    )
    ready = score_citation_readiness(
        url="https://example.gov/rules/final",
        title="Final Rule",
        word_count=800,
        freshness=freshness,
        authority=authority,
        primary=primary,
        warnings=[],
    )
    blocked = score_citation_readiness(
        url="https://example.com/search?q=x&utm_source=y",
        title=None,
        word_count=50,
        freshness=freshness,
        authority=authority,
        primary=primary,
        warnings=[SourceAuditWarning(code="short", severity="high", message="short")],
    )

    assert ready.citation_ready is True
    assert blocked.citation_ready is False
    assert blocked.blockers


def test_batch_ranking_and_gaps_are_deterministic():
    batch = audit_sources(
        [
            {
                "source_id": "S1",
                "url": "https://docs.example.com/reference",
                "title": "Official Reference",
                "document_kind": "html",
                "word_count": 900,
                "text": _long_text("Official documentation by Jane Doe"),
                "ok": True,
            },
            {
                "source_id": "S2",
                "url": "https://affiliate.example/review",
                "title": "Best Ultimate Review",
                "word_count": 120,
                "text": "Affiliate commission best ultimate guaranteed. Published 2020-01-01.",
                "ok": True,
            },
        ],
        question="latest API docs in 2026",
        now=NOW,
    )

    assert batch.summary.ranked_source_ids[0] == "S1"
    assert "S2" in batch.summary.sources_needing_verification
    assert batch.summary.instruction_block


def test_source_audit_artifact_writer(tmp_path: Path):
    batch = audit_sources(
        [
            {
                "source_id": "S1",
                "url": "https://example.gov/report.pdf",
                "title": "Official Report",
                "document_kind": "pdf",
                "word_count": 900,
                "text": _long_text("Official report by Agency"),
                "ok": True,
            }
        ],
        question="current official report",
        thread_id="audit-artifacts",
        now=NOW,
    )

    paths = write_source_audit_artifacts(tmp_path, batch)

    assert "source_audit.json" in paths
    assert "source_audit.md" in paths
    assert "source_rankings.json" in paths
    assert "source_warnings.md" in paths
    assert "citation_readiness.json" in paths
    payload = json.loads((tmp_path / "source_audit.json").read_text(encoding="utf-8"))
    assert payload["summary"]["source_count"] == 1


def test_source_audit_api_route(client):
    response = client.post(
        "/source-audit",
        json={
            "question": "What are the latest API docs in 2026?",
            "persist": False,
            "sources": [
                {
                    "source_id": "S1",
                    "url": "https://docs.example.com/reference",
                    "title": "Official API Reference",
                    "document_kind": "html",
                    "word_count": 900,
                    "text": _long_text("Official documentation by Jane Doe"),
                    "ok": True,
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["source_count"] == 1
    assert body["audits"][0]["source_id"] == "S1"
    assert body["audits"][0]["citation_readiness_score"]["score"] > 0


def test_run_source_audit_get_routes_with_mock_run(client):
    run = client.post(
        "/run",
        json={
            "question": "Audit mock source behavior",
            "mock_mode": True,
            "urls": ["https://example.invalid/source"],
        },
    )
    assert run.status_code == 200
    thread_id = run.json()["thread_id"]

    audit = client.get(f"/runs/{thread_id}/source-audit")
    readiness = client.get(f"/runs/{thread_id}/citation-readiness")

    assert audit.status_code == 200
    assert audit.json()["summary"]["source_count"] == 1
    assert audit.json()["audits"][0]["recommended_usage"] == "exclude_from_report"
    assert readiness.status_code == 200
    assert readiness.json()["sources"][0]["source_id"] == "S1"
