from __future__ import annotations

import json
from pathlib import Path

from deep_research_agent.evaluation.artifact_writer import rebuild_evaluation_artifacts
from deep_research_agent.evaluation.balance import assess_balance
from deep_research_agent.evaluation.benchmark import list_benchmark_cases
from deep_research_agent.evaluation.coverage import detect_coverage_gaps
from deep_research_agent.evaluation.freshness_review import assess_freshness
from deep_research_agent.evaluation.hallucination_risk import assess_hallucination_risk
from deep_research_agent.evaluation.regression_runner import run_regression_suite
from deep_research_agent.evaluation.rubric import default_rubric, score_rubric


def _write_basic_run(run_dir: Path, *, thread_id: str = "eval-test") -> None:
    (run_dir / "sources").mkdir(parents=True, exist_ok=True)
    (run_dir / "sources" / "s1.txt").write_text(
        "In 2026, Alpha reported 42 enterprise customers. Beta reported 39 customers. "
        "Independent reviewers noted limitations and tradeoffs.",
        encoding="utf-8",
    )
    sources = [
        {
            "source_id": "S1",
            "url": "https://example.org/alpha-beta-2026",
            "title": "Alpha Beta 2026 Review",
            "ok": True,
            "fetched_at": "2026-05-01T00:00:00Z",
            "local_path": f"runs/{thread_id}/sources/s1.txt",
            "word_count": 18,
        }
    ]
    (run_dir / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
    (run_dir / "notes.md").write_text(
        "# Notes\n\n- S1 says Alpha had 42 customers and Beta had 39 customers.\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        "# Report\n\nResearch date: as of 2026-05-05.\n\n"
        "Alpha reported 42 enterprise customers while Beta reported 39 customers [S1]. "
        "The comparison has limitations and tradeoffs [S1].\n",
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "question": "Compare Alpha and Beta using the latest 2026 evidence.",
                "input_snapshot": {
                    "question": "Compare Alpha and Beta using the latest 2026 evidence.",
                    "urls": [],
                },
            }
        ),
        encoding="utf-8",
    )


def test_default_rubric_has_required_criteria():
    rubric = default_rubric()
    keys = {criterion.key for criterion in rubric.criteria}
    assert "question_answered" in keys
    assert "citation_support" in keys
    assert "safety_and_overclaiming" in keys
    assert len(keys) >= 12


def test_coverage_gap_detection_flags_missing_entity_and_unused_source():
    gaps = detect_coverage_gaps(
        question="Compare Alpha and Beta for enterprise adoption.",
        report_text="Alpha has good enterprise adoption [S1].",
        notes_text="",
        sources=[
            {"source_id": "S1", "url": "https://example.org/a", "ok": True},
            {"source_id": "S2", "url": "https://example.org/b", "ok": True},
        ],
    )
    kinds = {gap.kind for gap in gaps}
    assert "missing_question_entity" in kinds
    assert "unused_fetched_url" in kinds


def test_hallucination_risk_detects_unsupported_numbers_dates_and_entities():
    risk = assess_hallucination_risk(
        report_text="Acme Corp reached 80% share in 2026 and Zeta Labs is guaranteed to win.",
        notes_text="Acme Corp was mentioned in notes.",
        source_texts=["Acme Corp had enterprise customers in 2025."],
    )
    kinds = {finding.kind for finding in risk.findings}
    assert "unsupported_number" in kinds
    assert "unsupported_date" in kinds
    assert "introduced_entity" in kinds
    assert "absolute_wording" in kinds
    assert risk.risk_score > 0


def test_balance_assessment_flags_vendor_bias_and_missing_counterarguments():
    balance = assess_balance(
        question="Compare Vendor Alpha versus Vendor Beta and recommend one.",
        report_text="Vendor Alpha is better.",
        sources=[
            {"source_id": "S1", "url": "https://vercel.com/report", "ok": True},
            {"source_id": "S2", "url": "https://vercel.com/docs", "ok": True},
        ],
    )
    assert balance.is_comparative_or_controversial is True
    assert balance.vendor_bias_warning
    assert balance.score < 0.5


def test_freshness_review_requires_dates_for_current_questions():
    freshness = assess_freshness(
        question="What is the latest pricing today?",
        report_text="The latest pricing is lower now.",
        sources=[{"source_id": "S1", "url": "https://example.org/pricing", "ok": True}],
    )
    assert freshness.is_time_sensitive is True
    assert freshness.source_dates_present is False
    assert freshness.score < 0.5


def test_evaluation_artifact_writing_builds_quality_files(tmp_path: Path):
    run_dir = tmp_path / "eval-test"
    _write_basic_run(run_dir)
    evaluation = rebuild_evaluation_artifacts(run_dir, thread_id="eval-test")
    assert evaluation.overall_score > 0
    assert (run_dir / "evaluation.json").exists()
    assert (run_dir / "coverage_gaps.md").exists()
    assert (run_dir / "hallucination_risk.json").exists()
    quality = json.loads((run_dir / "quality_score.json").read_text(encoding="utf-8"))
    assert quality["thread_id"] == "eval-test"


def test_rubric_scoring_explains_every_score():
    rubric, scores, citation_quality, overall = score_rubric(
        question="Compare Alpha and Beta.",
        report_text="Alpha has 42 customers and Beta has 39 customers [S1].",
        notes_text="Alpha and Beta were compared.",
        sources=[{"source_id": "S1", "url": "https://example.org", "ok": True}],
        artifacts_present={"plan.md", "notes.md", "sources.json", "report.md"},
        coverage_gaps=[],
        hallucination_risk=assess_hallucination_risk(
            report_text="Alpha has 42 customers and Beta has 39 customers [S1].",
            notes_text="Alpha has 42 customers and Beta has 39 customers.",
            source_texts=["Alpha has 42 customers and Beta has 39 customers."],
        ),
        balance=assess_balance(
            question="Compare Alpha and Beta.",
            report_text="Alpha has 42 customers while Beta has 39 customers.",
            sources=[{"source_id": "S1", "url": "https://example.org", "ok": True}],
        ),
        freshness=assess_freshness(
            question="Compare Alpha and Beta.",
            report_text="Alpha has 42 customers.",
            sources=[{"source_id": "S1", "url": "https://example.org", "ok": True}],
        ),
    )
    assert rubric.criteria
    assert citation_quality.score >= 0
    assert overall >= 0
    assert all(score.reasons for score in scores)


def test_benchmark_case_loading_and_regression_runner(tmp_path: Path):
    cases = list_benchmark_cases()
    assert {case.case_id for case in cases} >= {
        "current-ai-search-quality",
        "unsupported-market-claim",
    }
    result = run_regression_suite(output_dir=tmp_path / "benchmarks", case_ids=[cases[0].case_id])
    assert result.total_cases == 1
    assert result.results[0].artifacts


def test_evaluation_rebuild_endpoint(client):
    response = client.post("/run", json={"question": "test question"})
    assert response.status_code == 200
    thread_id = response.json()["thread_id"]

    rebuild = client.post(f"/runs/{thread_id}/evaluation/rebuild")
    assert rebuild.status_code == 200
    assert rebuild.json()["evaluation"]["thread_id"] == thread_id

    evaluation = client.get(f"/runs/{thread_id}/evaluation")
    assert evaluation.status_code == 200
    quality = client.get(f"/runs/{thread_id}/quality-score")
    assert quality.status_code == 200
    assert "overall_score" in quality.json()


def test_benchmark_endpoints(client):
    cases = client.get("/benchmarks/cases")
    assert cases.status_code == 200
    assert any(case["case_id"] == "current-ai-search-quality" for case in cases.json())

    run = client.post("/benchmarks/run", json={"case_ids": ["current-ai-search-quality"]})
    assert run.status_code == 200
    assert run.json()["total_cases"] == 1
