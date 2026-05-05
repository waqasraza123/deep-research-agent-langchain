from __future__ import annotations

import json
from pathlib import Path

from deep_research_agent.artifacts import ensure_thread_dir
from deep_research_agent.evidence.artifact_writer import rebuild_evidence_artifacts
from deep_research_agent.verification import (
    ClaimChallenger,
    ConfidenceCalibrator,
    DeterministicVerifier,
    ResearchCritic,
    VerificationConfig,
    VerificationTask,
    VerificationTaskGenerator,
    VerificationTaskStatus,
    VerificationTaskType,
    rebuild_verification_artifacts,
)


def _write_run(run_dir: Path, *, thread_id: str = "verify-test") -> None:
    (run_dir / "sources").mkdir(parents=True, exist_ok=True)
    (run_dir / "sources" / "s1.txt").write_text(
        "Acme Search reduced latency by 35% in 2025 after a cache rollout. "
        "The rollout had tradeoffs and deployment risks. "
        "Acme Search published updated release notes in 2025.",
        encoding="utf-8",
    )
    (run_dir / "sources" / "s2.txt").write_text(
        "Acme Search reduced latency by 12% in 2025 in a smaller benchmark.",
        encoding="utf-8",
    )
    sources = [
        {
            "source_id": "S1",
            "url": "https://docs.example.com/acme",
            "title": "Acme Search Release Notes 2025",
            "ok": True,
            "fetched_at": "2026-05-01T00:00:00Z",
            "local_path": f"runs/{thread_id}/sources/s1.txt",
            "word_count": 500,
        },
        {
            "source_id": "S2",
            "url": "https://review.example.org/acme",
            "title": "Independent Acme benchmark",
            "ok": True,
            "fetched_at": "2026-05-01T00:00:00Z",
            "local_path": f"runs/{thread_id}/sources/s2.txt",
            "word_count": 120,
        },
    ]
    (run_dir / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
    (run_dir / "notes.md").write_text(
        "# Notes\n\n- S1 says Acme Search reduced latency by 35% in 2025.\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        "# Report\n\n"
        "Acme Search reduced latency by 35% in 2025 [S1]. "
        "Acme Search reached 80% market share in 2026. "
        "Acme Search is the best choice for production. "
        "Teams should recommend Acme Search for all production deployments. "
        "The latest Acme Search benchmark is current.\n",
        encoding="utf-8",
    )
    (run_dir / "source_audit.json").write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "question": "Should teams use Acme Search?",
                "generated_at": "2026-05-05T00:00:00Z",
                "audits": [
                    {
                        "source_id": "S1",
                        "recommended_usage": "cite_directly",
                        "freshness_score": {"status": "current", "score": 0.9},
                        "authority_score": {"source_role": "primary", "score": 0.9},
                        "primary_source_likelihood": {"likelihood": 0.8},
                        "bias_risk_score": {"risk_level": "low"},
                        "warnings": [],
                    },
                    {
                        "source_id": "S2",
                        "recommended_usage": "use_with_caution",
                        "freshness_score": {"status": "unknown", "score": 0.2},
                        "authority_score": {"source_role": "secondary", "score": 0.5},
                        "primary_source_likelihood": {"likelihood": 0.2},
                        "bias_risk_score": {"risk_level": "medium"},
                        "warnings": [{"code": "limited_context", "severity": "medium"}],
                    },
                ],
                "summary": {},
            }
        ),
        encoding="utf-8",
    )
    rebuild_evidence_artifacts(run_dir, thread_id=thread_id)


def test_critic_detects_weak_claims_and_freshness(tmp_path: Path):
    run_dir = tmp_path / "verify-test"
    _write_run(run_dir)

    findings = ResearchCritic().audit(run_dir, thread_id="verify-test")
    kinds = {finding.kind for finding in findings}

    assert "weak_numeric_support" in kinds
    assert "overconfident_claim" in kinds
    assert "recommendation_without_evidence" in kinds
    assert "stale_source_risk" in kinds


def test_numeric_date_and_recommendation_task_creation(tmp_path: Path):
    run_dir = tmp_path / "verify-test"
    _write_run(run_dir)

    findings = ResearchCritic().audit(run_dir, thread_id="verify-test")
    plan = VerificationTaskGenerator().build_plan(
        thread_id="verify-test",
        critic_findings=findings,
        artifacts_used=["report.md", "sources.json"],
        config=VerificationConfig(max_verification_tasks=20, max_high_priority_tasks=20),
    )
    task_types = {task.task_type for task in plan.tasks}

    assert VerificationTaskType.VERIFY_NUMERIC_CLAIM in task_types
    assert VerificationTaskType.VERIFY_RECOMMENDATION in task_types
    assert VerificationTaskType.VERIFY_FRESHNESS in task_types
    assert all(task.reason for task in plan.tasks)


def test_date_verification_task_creation(tmp_path: Path):
    run_dir = tmp_path / "date-test"
    _write_run(run_dir, thread_id="date-test")
    (run_dir / "report.md").write_text(
        "# Report\n\nThe Acme Search migration completed on March 4, 2024.\n",
        encoding="utf-8",
    )
    rebuild_evidence_artifacts(run_dir, thread_id="date-test")

    findings = ResearchCritic().audit(run_dir, thread_id="date-test")
    plan = VerificationTaskGenerator().build_plan(
        thread_id="date-test",
        critic_findings=findings,
        artifacts_used=["report.md"],
    )

    assert any(task.task_type == VerificationTaskType.VERIFY_DATE_CLAIM for task in plan.tasks)


def test_unsupported_claim_verification(tmp_path: Path):
    run_dir = tmp_path / "verify-test"
    _write_run(run_dir)
    task = VerificationTask(
        task_id="VT-unsupported",
        task_type=VerificationTaskType.VERIFY_UNSUPPORTED_CLAIM,
        claim_or_question="Zeta Labs guarantees zero downtime for every deployment.",
        source_artifact="report.md",
        priority=1,
        reason="Strong claim needs support.",
        expected_evidence_type="source_text",
    )

    result = DeterministicVerifier().run_task(run_dir, task)

    assert result.status == VerificationTaskStatus.UNSUPPORTED
    assert result.reasons


def test_contradiction_verification(tmp_path: Path):
    run_dir = tmp_path / "verify-test"
    _write_run(run_dir)
    task = VerificationTask(
        task_id="VT-contradiction",
        task_type=VerificationTaskType.VERIFY_CONTRADICTION,
        claim_or_question="Resolve conflicting Acme Search latency reductions.",
        source_artifact="evidence_ledger.json",
        priority=1,
        reason="Evidence ledger reported conflicting values.",
        expected_evidence_type="contradiction_group",
    )

    result = DeterministicVerifier().run_task(run_dir, task)

    assert result.status == VerificationTaskStatus.CONTRADICTED
    assert result.evidence


def test_confidence_calibration_and_claim_rewrites(tmp_path: Path):
    run_dir = tmp_path / "verify-test"
    _write_run(run_dir)
    batch = rebuild_verification_artifacts(run_dir, thread_id="verify-test")

    calibration = ConfidenceCalibrator().calibrate(
        run_dir,
        thread_id="verify-test",
        results=batch.results,
    )
    suggestions = ClaimChallenger().suggestions_for_results(batch.results)

    assert 0 <= calibration.report_confidence_after <= 1
    assert calibration.penalties
    assert suggestions
    assert any("best" not in item["calibrated"].lower() for item in suggestions)


def test_verification_artifact_writing(tmp_path: Path):
    run_dir = tmp_path / "verify-test"
    _write_run(run_dir)

    batch = rebuild_verification_artifacts(run_dir, thread_id="verify-test")

    assert batch.summary.total_tasks == len(batch.results)
    for name in (
        "verification_plan.json",
        "verification_plan.md",
        "verification_tasks.json",
        "verification_results.json",
        "verification_report.md",
        "confidence_calibration.json",
        "confidence_calibration.md",
        "claim_rewrite_suggestions.md",
    ):
        assert (run_dir / name).exists()


def test_verification_rebuild_endpoint(client, test_runs_dir: Path):
    thread_id = "endpoint-verify"
    run_dir = ensure_thread_dir(test_runs_dir, thread_id)
    _write_run(run_dir, thread_id=thread_id)

    rebuild = client.post(f"/runs/{thread_id}/verification/rebuild")
    assert rebuild.status_code == 200
    assert rebuild.json()["summary"]["thread_id"] == thread_id

    verification = client.get(f"/runs/{thread_id}/verification")
    assert verification.status_code == 200
    assert verification.json()["summary"]["thread_id"] == thread_id

    confidence = client.get(f"/runs/{thread_id}/confidence-calibration")
    assert confidence.status_code == 200
    assert "report_confidence_after" in confidence.json()

    rewrites = client.get(f"/runs/{thread_id}/claim-rewrite-suggestions")
    assert rewrites.status_code == 200
    assert "suggestions" in rewrites.json()
