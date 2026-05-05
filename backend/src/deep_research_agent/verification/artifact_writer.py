from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import (
    ConfidenceCalibration,
    VerificationBatch,
    VerificationPlan,
    VerificationResult,
    VerificationTask,
    VerificationTaskStatus,
    model_to_plain,
)

VERIFICATION_ARTIFACTS = (
    "verification_plan.json",
    "verification_plan.md",
    "verification_tasks.json",
    "verification_results.json",
    "verification_report.md",
    "confidence_calibration.json",
    "confidence_calibration.md",
    "claim_rewrite_suggestions.md",
)


def write_verification_artifacts(run_dir: Path, batch: VerificationBatch) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "verification_plan.json", model_to_plain(batch.plan))
    (run_dir / "verification_plan.md").write_text(
        render_verification_plan_md(batch.plan), encoding="utf-8"
    )
    _write_json(
        run_dir / "verification_tasks.json",
        [model_to_plain(task) for task in batch.plan.tasks],
    )
    _write_json(
        run_dir / "verification_results.json",
        {
            "thread_id": batch.thread_id,
            "batch_id": batch.batch_id,
            "generated_at": batch.generated_at,
            "summary": model_to_plain(batch.summary),
            "results": [model_to_plain(result) for result in batch.results],
            "findings": [model_to_plain(finding) for finding in batch.findings],
            "claim_rewrite_suggestions": batch.claim_rewrite_suggestions,
        },
    )
    (run_dir / "verification_report.md").write_text(
        render_verification_report_md(batch), encoding="utf-8"
    )
    _write_json(
        run_dir / "confidence_calibration.json",
        model_to_plain(batch.confidence_calibration),
    )
    (run_dir / "confidence_calibration.md").write_text(
        render_confidence_calibration_md(batch.confidence_calibration), encoding="utf-8"
    )
    (run_dir / "claim_rewrite_suggestions.md").write_text(
        render_claim_rewrite_suggestions_md(batch.claim_rewrite_suggestions),
        encoding="utf-8",
    )


def render_verification_plan_md(plan: VerificationPlan) -> str:
    lines = [
        "# Verification Plan",
        "",
        f"- Thread: `{plan.thread_id}`",
        f"- Plan: `{plan.plan_id}`",
        f"- Generated: `{plan.generated_at}`",
        f"- Method: `{plan.method}`",
        f"- Tasks: {len(plan.tasks)}",
        "",
        "## Artifacts Used",
        "",
    ]
    if plan.artifacts_used:
        lines.extend(f"- `{artifact}`" for artifact in plan.artifacts_used)
    else:
        lines.append("- None")
    if plan.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in plan.warnings)
    lines.extend(["", "## Tasks", ""])
    if not plan.tasks:
        lines.append("No verification tasks were generated.")
    for task in plan.tasks:
        lines.extend(_task_lines(task))
    lines.extend(["", "## Critic Findings", ""])
    if not plan.critic_findings:
        lines.append("No critic findings were detected.")
    for finding in plan.critic_findings:
        lines.extend(
            [
                f"### {finding.finding_id}",
                "",
                f"- Kind: `{finding.kind}`",
                f"- Severity: `{finding.severity}`",
                f"- Priority: P{finding.priority}",
                f"- Task type: `{finding.suggested_task_type.value}`",
                f"- Reason: {finding.reason}",
                f"- Claim: {finding.claim_or_question}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_verification_report_md(batch: VerificationBatch) -> str:
    summary = batch.summary
    lines = [
        "# Verification Report",
        "",
        f"- Thread: `{batch.thread_id}`",
        f"- Batch: `{batch.batch_id}`",
        f"- Generated: `{batch.generated_at}`",
        f"- Report confidence: {summary.confidence_after:.3f}",
        "",
        "## Summary",
        "",
        f"- Total tasks: {summary.total_tasks}",
        f"- Verified: {summary.verified}",
        f"- Partially verified: {summary.partially_verified}",
        f"- Unsupported: {summary.unsupported}",
        f"- Contradicted: {summary.contradicted}",
        f"- Not enough information: {summary.not_enough_information}",
        f"- High-priority open issues: {summary.high_priority_open_issues}",
        "",
    ]
    if summary.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
        lines.append("")
    if summary.recommended_next_actions:
        lines.extend(["## Recommended Next Actions", ""])
        lines.extend(f"- {action}" for action in summary.recommended_next_actions)
        lines.append("")
    lines.extend(["## Results", ""])
    if not batch.results:
        lines.append("No verification tasks were run.")
    for result in batch.results:
        task = next((item for item in batch.plan.tasks if item.task_id == result.task_id), None)
        claim = task.claim_or_question if task else result.task_id
        lines.extend(
            [
                f"### {result.task_id}",
                "",
                f"- Status: `{result.status.value}`",
                f"- Confidence: {result.confidence_before:.3f} -> {result.confidence_after:.3f}",
                f"- Claim/question: {claim}",
                "- Reasons:",
            ]
        )
        lines.extend(f"  - {reason}" for reason in result.reasons)
        if result.evidence:
            lines.append("- Evidence:")
            for evidence in result.evidence[:4]:
                source = evidence.source_id or evidence.source_artifact or "artifact"
                excerpt = _one_line(evidence.excerpt, max_chars=220)
                lines.append(f"  - `{source}` {evidence.score:.2f}: {excerpt}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_confidence_calibration_md(calibration: ConfidenceCalibration) -> str:
    lines = [
        "# Confidence Calibration",
        "",
        f"- Thread: `{calibration.thread_id}`",
        f"- Calibration: `{calibration.calibration_id}`",
        f"- Generated: `{calibration.generated_at}`",
        f"- Confidence: {calibration.report_confidence_before:.3f} -> "
        f"{calibration.report_confidence_after:.3f}",
        f"- Label: `{calibration.confidence_label}`",
        f"- Source diversity: {calibration.source_diversity_score:.3f}",
        "",
        "## Factors",
        "",
    ]
    lines.extend(f"- {factor}" for factor in calibration.factors)
    lines.extend(["", "## Penalties", ""])
    if calibration.penalties:
        lines.extend(f"- {penalty}" for penalty in calibration.penalties)
    else:
        lines.append("- None")
    return "\n".join(lines).rstrip() + "\n"


def render_claim_rewrite_suggestions_md(suggestions: list[dict[str, Any]]) -> str:
    lines = ["# Claim Rewrite Suggestions", ""]
    if not suggestions:
        lines.append("No claim rewrite suggestions were generated.")
        return "\n".join(lines).rstrip() + "\n"
    lines.append(
        "These are suggestions only. The verification engine does not silently replace report text."
    )
    lines.append("")
    for item in suggestions:
        lines.extend(
            [
                f"## {item.get('suggestion_id', 'suggestion')}",
                "",
                f"- Task: `{item.get('task_id', '')}`",
                f"- Status: `{item.get('status', '')}`",
                f"- Reason: {item.get('reason', '')}",
                "",
                "Original:",
                "",
                str(item.get("original", "")),
                "",
                "Calibrated:",
                "",
                str(item.get("calibrated", "")),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _task_lines(task: VerificationTask) -> list[str]:
    return [
        f"### {task.task_id}",
        "",
        f"- Type: `{task.task_type.value}`",
        f"- Priority: P{task.priority}",
        f"- Status: `{task.status.value}`",
        f"- Expected evidence: `{task.expected_evidence_type}`",
        "- Candidate sources: "
        + (", ".join(f"`{sid}`" for sid in task.candidate_source_ids) or "none"),
        f"- Reason: {task.reason}",
        f"- Claim/question: {task.claim_or_question}",
        "",
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _one_line(value: str, *, max_chars: int) -> str:
    clean = " ".join((value or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:") + "..."


def result_counts(results: list[VerificationResult]) -> dict[VerificationTaskStatus, int]:
    return {
        status: len([result for result in results if result.status == status])
        for status in VerificationTaskStatus
    }
