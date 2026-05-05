from __future__ import annotations

import hashlib
from pathlib import Path

from deep_research_agent.artifacts import now_iso_utc

from .artifact_writer import write_verification_artifacts
from .claim_challenger import ClaimChallenger
from .confidence_calibration import ConfidenceCalibrator
from .contracts import (
    VerificationBatch,
    VerificationConfig,
    VerificationResult,
    VerificationSummary,
    VerificationTaskStatus,
)
from .critic import ResearchCritic
from .fact_check_tasks import VerificationTaskGenerator
from .verifier import DeterministicVerifier


def rebuild_verification_artifacts(
    run_dir: Path,
    *,
    thread_id: str,
    config: VerificationConfig | None = None,
) -> VerificationBatch:
    batch = build_verification_batch(run_dir, thread_id=thread_id, config=config)
    write_verification_artifacts(run_dir, batch)
    return batch


def build_verification_batch(
    run_dir: Path,
    *,
    thread_id: str,
    config: VerificationConfig | None = None,
) -> VerificationBatch:
    config = config or VerificationConfig()
    critic = ResearchCritic()
    critic_input = critic.load_input(run_dir, thread_id=thread_id)
    findings = critic.audit(run_dir, thread_id=thread_id, config=config)
    plan = VerificationTaskGenerator().build_plan(
        thread_id=thread_id,
        critic_findings=findings,
        artifacts_used=critic_input.available_artifacts,
        config=config,
    )
    verifier = DeterministicVerifier()
    results = verifier.run_plan(run_dir, plan)
    results_by_id = {result.task_id: result for result in results}
    for task in plan.tasks:
        result = results_by_id.get(task.task_id)
        if result is None:
            continue
        task.status = result.status
        task.result = result
        task.confidence_after = result.confidence_after
    calibration = ConfidenceCalibrator().calibrate(run_dir, thread_id=thread_id, results=results)
    rewrite_suggestions = ClaimChallenger().suggestions_for_results(results)
    summary = build_verification_summary(
        thread_id=thread_id,
        results=results,
        confidence_after=calibration.report_confidence_after,
        plan_task_priorities={task.task_id: task.priority for task in plan.tasks},
        warnings=plan.warnings,
    )
    batch_id = "VB-" + hashlib.sha1(
        f"{thread_id}:{plan.plan_id}:{calibration.calibration_id}".encode("utf-8")
    ).hexdigest()[:12]
    return VerificationBatch(
        batch_id=batch_id,
        thread_id=thread_id,
        generated_at=now_iso_utc(),
        plan=plan,
        results=results,
        findings=[finding for result in results for finding in result.findings],
        confidence_calibration=calibration,
        summary=summary,
        claim_rewrite_suggestions=rewrite_suggestions,
    )


def build_verification_summary(
    *,
    thread_id: str,
    results: list[VerificationResult],
    confidence_after: float,
    plan_task_priorities: dict[str, int],
    warnings: list[str],
) -> VerificationSummary:
    counts = {status: 0 for status in VerificationTaskStatus}
    for result in results:
        counts[result.status] += 1
    high_priority_open = len(
        [
            result
            for result in results
            if plan_task_priorities.get(result.task_id, 5) <= 2
            and result.status
            in {
                VerificationTaskStatus.CONTRADICTED,
                VerificationTaskStatus.UNSUPPORTED,
                VerificationTaskStatus.NOT_ENOUGH_INFORMATION,
            }
        ]
    )
    next_actions: list[str] = []
    if counts[VerificationTaskStatus.CONTRADICTED]:
        next_actions.append("Resolve contradicted claims before treating the report as final.")
    if counts[VerificationTaskStatus.UNSUPPORTED]:
        next_actions.append("Rewrite or remove unsupported claims, or fetch targeted sources.")
    if counts[VerificationTaskStatus.NOT_ENOUGH_INFORMATION]:
        next_actions.append("Add source text or retrieval context for unresolved tasks.")
    if not next_actions:
        next_actions.append("Keep verification artifacts with the report for auditability.")

    summary_warnings = list(warnings)
    if high_priority_open:
        summary_warnings.append(
            f"{high_priority_open} high-priority verification issue(s) remain unresolved."
        )
    return VerificationSummary(
        thread_id=thread_id,
        generated_at=now_iso_utc(),
        total_tasks=len(results),
        verified=counts[VerificationTaskStatus.VERIFIED],
        partially_verified=counts[VerificationTaskStatus.PARTIALLY_VERIFIED],
        contradicted=counts[VerificationTaskStatus.CONTRADICTED],
        unsupported=counts[VerificationTaskStatus.UNSUPPORTED],
        not_enough_information=counts[VerificationTaskStatus.NOT_ENOUGH_INFORMATION],
        skipped=counts[VerificationTaskStatus.SKIPPED],
        high_priority_open_issues=high_priority_open,
        confidence_after=confidence_after,
        warnings=summary_warnings,
        recommended_next_actions=next_actions,
    )
