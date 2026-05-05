from __future__ import annotations

import json
from datetime import datetime, timezone

from deep_research_agent.artifacts import ensure_thread_dir
from deep_research_agent.temporal import (
    SourceTemporalMetadata,
    assess_source_currentness,
    build_timeline,
    detect_time_sensitive_question,
    detect_version_signals,
    extract_dates_from_text,
    extract_time_sensitive_claims,
    rebuild_temporal_artifacts,
)


def test_date_extraction_common_formats():
    text = (
        "Published March 5, 2024. Last updated 15 April 2025. "
        "Effective 2026-01-01. Release notes for May 2023. Legacy 2021 docs."
    )
    dates = extract_dates_from_text(text, source_id="S1", source_url="https://example.com")
    normalized = {item.normalized_date: item.date_type for item in dates}

    assert normalized["2024-03-05"] == "published"
    assert normalized["2025-04-15"] == "updated"
    assert normalized["2026-01-01"] == "effective"
    assert normalized["2023-05-01"] == "version_release"
    assert normalized["2021-01-01"] == "mentioned_date"


def test_time_sensitive_question_detection():
    sensitive, signals = detect_time_sensitive_question(
        "What is the latest API pricing and model capability in 2026?",
        current_year=2026,
    )

    assert sensitive is True
    assert {"latest", "pricing", "api docs", "2026"} & set(signals)


def test_source_version_signal_detection():
    signals = detect_version_signals(
        "This archived legacy migration guide covers v2 and release notes for 1.4.3 beta.",
        source_id="S1",
    )
    types = {signal.signal_type for signal in signals}

    assert "major_version" in types
    assert "semantic_version" in types
    assert "archived_docs" in types
    assert "legacy_docs" in types
    assert any(signal.outdated_hint for signal in signals)


def test_stale_source_classification_for_current_question():
    source = SourceTemporalMetadata(
        source_id="S1",
        source_url="https://example.com/old",
        newest_date="2020-01-01",
        oldest_date="2020-01-01",
    )

    assessed = assess_source_currentness(
        source,
        freshness_required=True,
        today=datetime(2026, 5, 5, tzinfo=timezone.utc).date(),
    )

    assert assessed.currentness_status == "stale"
    assert assessed.warnings == []


def test_timeline_event_creation():
    source = SourceTemporalMetadata(
        source_id="S1",
        source_url="https://example.com",
        extracted_dates=extract_dates_from_text("Published January 2, 2025.", source_id="S1"),
    )
    source.newest_date = "2025-01-02"
    claims = extract_time_sensitive_claims(
        report_text="As of March 2025, the current API is v2. [S1]",
        notes_text="",
        sources=[source],
    )

    events = build_timeline(sources=[source], claims=claims)

    assert any(event.event_type == "published" for event in events)
    assert any(event.event_type == "claim_date" for event in events)


def test_date_sensitive_claim_checking_flags_stale_source_risk():
    source = SourceTemporalMetadata(
        source_id="S1",
        source_url="https://example.com/old",
        newest_date="2020-01-01",
        oldest_date="2020-01-01",
        currentness_status="stale",
    )
    claims = extract_time_sensitive_claims(
        report_text="The latest pricing is $20 per month. [S1]",
        notes_text="",
        sources=[source],
    )

    assert claims
    assert claims[0].status == "stale_source_risk"


def test_temporal_artifact_writing(tmp_path):
    run_dir = tmp_path / "runs" / "temporal-test"
    run_dir.mkdir(parents=True)
    (run_dir / "sources").mkdir()
    (run_dir / "sources" / "s1.txt").write_text(
        "Release notes published March 1, 2026 for API v3. Stable docs.",
        encoding="utf-8",
    )
    (run_dir / "sources.json").write_text(
        json.dumps(
            [
                {
                    "source_id": "S1",
                    "url": "https://docs.example.com/releases/2026/03/01",
                    "title": "API release notes",
                    "ok": True,
                    "local_path": "runs/temporal-test/sources/s1.txt",
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        "# Report\n\nThe current API release is v3 as of March 2026. [S1]\n",
        encoding="utf-8",
    )
    (run_dir / "notes.md").write_text("# Notes\n\nCurrent docs reviewed. [S1]\n", encoding="utf-8")

    bundle = rebuild_temporal_artifacts(
        run_dir,
        thread_id="temporal-test",
        question="What is the current API release in 2026?",
        now=datetime(2026, 5, 5, tzinfo=timezone.utc),
    )

    assert bundle.currentness.freshness_required is True
    assert bundle.currentness.newest_source_date == "2026-03-01"
    for name in (
        "temporal_profile.json",
        "timeline.json",
        "currentness_assessment.json",
        "temporal_claims.json",
        "temporal_warnings.md",
    ):
        assert (run_dir / name).exists()


def test_temporal_rebuild_endpoint(client, test_runs_dir):
    thread_id = "temporal-api-test"
    td = ensure_thread_dir(test_runs_dir, thread_id)
    (td / "sources").mkdir()
    (td / "sources" / "s1.txt").write_text(
        "Published January 1, 2026. Latest release notes for API v2.",
        encoding="utf-8",
    )
    (td / "sources.json").write_text(
        json.dumps(
            [
                {
                    "source_id": "S1",
                    "url": "https://docs.example.com/api/v2",
                    "title": "Latest API docs",
                    "ok": True,
                    "local_path": f"runs/{thread_id}/sources/s1.txt",
                }
            ]
        ),
        encoding="utf-8",
    )
    (td / "report.md").write_text(
        "# Report\n\nThe latest API is v2 as of January 2026. [S1]\n",
        encoding="utf-8",
    )
    (td / "notes.md").write_text("# Notes\n\nReviewed current API docs. [S1]\n", encoding="utf-8")

    response = client.post(f"/runs/{thread_id}/temporal/rebuild")

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == thread_id
    assert body["timeline_events"] >= 1

    currentness = client.get(f"/runs/{thread_id}/currentness")
    assert currentness.status_code == 200
    assert currentness.json()["source_count"] == 1
