from __future__ import annotations

import hashlib

from deep_research_agent.artifacts import now_iso_utc

from .contracts import (
    CriticFinding,
    VerificationConfig,
    VerificationPlan,
    VerificationTask,
    VerificationTaskType,
)


class VerificationTaskGenerator:
    """Create bounded, deterministic fact-check tasks from critic findings."""

    def build_plan(
        self,
        *,
        thread_id: str,
        critic_findings: list[CriticFinding],
        artifacts_used: list[str],
        config: VerificationConfig | None = None,
    ) -> VerificationPlan:
        config = config or VerificationConfig()
        sorted_findings = sorted(
            critic_findings,
            key=lambda item: (
                item.priority,
                -_severity_weight(item.severity),
                item.suggested_task_type.value,
                item.finding_id,
            ),
        )
        selected: list[CriticFinding] = []
        high_count = 0
        selected_task_types: set[VerificationTaskType] = set()
        for finding in sorted_findings:
            if len(selected) >= config.max_verification_tasks:
                break
            if (
                finding.priority <= 2
                and high_count >= config.max_high_priority_tasks
                and finding.suggested_task_type in selected_task_types
            ):
                continue
            if not _task_type_enabled(finding.suggested_task_type, config):
                continue
            selected.append(finding)
            selected_task_types.add(finding.suggested_task_type)
            if finding.priority <= 2:
                high_count += 1

        tasks = [_task_from_finding(finding, idx) for idx, finding in enumerate(selected, start=1)]
        plan_id = "VP-" + hashlib.sha1(
            f"{thread_id}:{','.join(task.task_id for task in tasks)}".encode("utf-8")
        ).hexdigest()[:12]
        warnings: list[str] = []
        if len(critic_findings) > len(tasks):
            warnings.append(
                f"Task generation bounded {len(critic_findings)} critic findings to "
                f"{len(tasks)} verification task(s)."
            )
        return VerificationPlan(
            plan_id=plan_id,
            thread_id=thread_id,
            generated_at=now_iso_utc(),
            max_verification_tasks=config.max_verification_tasks,
            max_high_priority_tasks=config.max_high_priority_tasks,
            tasks=tasks,
            critic_findings=critic_findings,
            artifacts_used=artifacts_used,
            warnings=warnings,
        )


def _task_from_finding(finding: CriticFinding, ordinal: int) -> VerificationTask:
    digest = hashlib.sha1(
        f"{ordinal}:{finding.finding_id}:{finding.suggested_task_type.value}".encode("utf-8")
    ).hexdigest()[:10]
    return VerificationTask(
        task_id=f"VT-{digest}",
        task_type=finding.suggested_task_type,
        claim_or_question=finding.claim_or_question,
        source_artifact=finding.source_artifact,
        priority=finding.priority,
        reason=finding.reason,
        expected_evidence_type=finding.expected_evidence_type,
        candidate_source_ids=finding.candidate_source_ids,
        confidence_before=finding.confidence_before,
        confidence_after=finding.confidence_before,
        notes=[
            f"Created from critic finding {finding.finding_id}.",
            "Priority is deterministic from finding severity and evidence risk.",
        ],
        metadata={"critic_finding_id": finding.finding_id, "critic_kind": finding.kind},
    )


def _task_type_enabled(task_type: VerificationTaskType, config: VerificationConfig) -> bool:
    if task_type == VerificationTaskType.VERIFY_NUMERIC_CLAIM:
        return config.verify_numbers
    if task_type == VerificationTaskType.VERIFY_DATE_CLAIM:
        return config.verify_dates
    if task_type == VerificationTaskType.VERIFY_RECOMMENDATION:
        return config.verify_recommendations
    if task_type == VerificationTaskType.VERIFY_PRIMARY_SOURCE_SUPPORT:
        return config.require_primary_source_for_sensitive_claims
    if task_type == VerificationTaskType.VERIFY_FRESHNESS:
        return config.freshness_verification_required
    if task_type == VerificationTaskType.VERIFY_CONTRADICTION:
        return config.contradiction_verification_required
    return True


def _severity_weight(severity: str) -> int:
    return {
        "critical": 5,
        "high": 4,
        "medium": 3,
        "low": 2,
        "info": 1,
    }.get(severity, 3)
