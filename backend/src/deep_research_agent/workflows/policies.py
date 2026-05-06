from __future__ import annotations

import json

from .contracts import (
    CompiledWorkflow,
    WarningSeverity,
    WorkflowExecutionResult,
    WorkflowReadiness,
    WorkflowReadinessStatus,
    WorkflowStageDefinition,
    WorkflowStageType,
    WorkflowWarning,
    model_to_plain,
)

SECRET_MARKERS = ("api_key", "token", "secret", "password", "authorization", "bearer")
FRESHNESS_TERMS = ("current", "latest", "today", "pricing", "version", "2026", "now")


def evaluate_pre_execution_policies(workflow: CompiledWorkflow) -> list[WorkflowWarning]:
    warnings: list[WorkflowWarning] = []
    stage_types = {stage.stage_type for stage in workflow.stages}
    settings = workflow.settings_snapshot
    policies = workflow.policies
    for policy in policies:
        denied = set(policy.denied_stage_types)
        for stage in workflow.stages:
            if stage.stage_type in denied:
                warnings.append(
                    WorkflowWarning(
                        code="denied_stage_present",
                        severity=WarningSeverity.critical,
                        stage_id=stage.stage_id,
                        message=(
                            f"Stage {stage.stage_type.value} is denied by "
                            f"policy {policy.policy_id}."
                        ),
                        affected_stages=[stage.stage_id],
                        recommended_action="Remove the denied stage or use a different template.",
                    )
                )
        if not policy.external_network_allowed and WorkflowStageType.source_fetching in stage_types:
            warnings.append(
                WorkflowWarning(
                    code="external_network_denied",
                    severity=WarningSeverity.high,
                    message="Policy denies external network access but source_fetching is present.",
                    affected_stages=["source_fetching"],
                )
            )
        if policy.model_required and not _model_available(settings) and not policy.mock_allowed:
            warnings.append(
                WorkflowWarning(
                    code="model_required_unavailable",
                    severity=WarningSeverity.critical,
                    message="Workflow requires a model and mock fallback is not allowed.",
                    recommended_action="Configure a model provider or enable explicit mock mode.",
                )
            )
        if policy.review_required or policy.sensitive_domain:
            warnings.append(
                WorkflowWarning(
                    code="human_review_required",
                    severity=WarningSeverity.high,
                    message="Sensitive workflow requires human review before use.",
                    recommended_action="Have a qualified reviewer inspect the output.",
                )
            )
        if policy.strict_citations:
            warnings.append(
                WorkflowWarning(
                    code="strict_citations_enabled",
                    severity=WarningSeverity.medium,
                    message=(
                        "Strict citation policy is enabled; uncited claims should "
                        "block readiness."
                    ),
                )
            )
        if policy.freshness_required or any(
            term in workflow.question.lower() for term in FRESHNESS_TERMS
        ):
            warnings.append(
                WorkflowWarning(
                    code="freshness_review_required",
                    severity=WarningSeverity.medium,
                    message=(
                        "Question is currentness-sensitive; stale or undated sources "
                        "reduce readiness."
                    ),
                )
            )
    return warnings


def evaluate_stage_policies(
    stage: WorkflowStageDefinition,
    workflow: CompiledWorkflow,
) -> list[WorkflowWarning]:
    warnings: list[WorkflowWarning] = []
    for policy in workflow.policies:
        if stage.stage_type in policy.denied_stage_types:
            warnings.append(
                WorkflowWarning(
                    code="stage_denied",
                    severity=WarningSeverity.critical,
                    stage_id=stage.stage_id,
                    message=f"Stage {stage.stage_type.value} is denied.",
                    affected_stages=[stage.stage_id],
                )
            )
        if (
            stage.stage_type == WorkflowStageType.source_fetching
            and not policy.external_network_allowed
        ):
            warnings.append(
                WorkflowWarning(
                    code="source_refetch_denied",
                    severity=WarningSeverity.high,
                    stage_id=stage.stage_id,
                    message="External network/source refetch is denied for this workflow.",
                )
            )
    return warnings


def evaluate_post_execution_policies(result: WorkflowExecutionResult) -> WorkflowReadiness:
    warnings = [warning.message for warning in result.warnings]
    blocking: list[str] = []
    required_review = False
    if result.missing_artifacts:
        blocking.extend(
            f"Missing required artifact: {artifact}" for artifact in result.missing_artifacts
        )
    if result.errors:
        blocking.extend(result.errors)
    if any(warning.code == "human_review_required" for warning in result.warnings):
        required_review = True
    gate_status = (result.quality_gate_result or {}).get("status")
    if gate_status and gate_status not in {"passed", "skipped"}:
        warnings.append(f"Quality gate status is {gate_status}.")
    if result.status.value == "failed":
        status = WorkflowReadinessStatus.failed
        usable = False
        confidence = 0.0
        reason = "Workflow execution failed."
    elif blocking:
        status = WorkflowReadinessStatus.blocked
        usable = False
        confidence = 0.1
        reason = "Required workflow outputs are missing or failed."
    elif result.degraded or gate_status in {"failed", "error"}:
        status = WorkflowReadinessStatus.degraded
        usable = True
        confidence = 0.45
        reason = "Workflow completed with degraded stages or failed gate."
    elif warnings or required_review:
        status = WorkflowReadinessStatus.ready_with_warnings
        usable = True
        confidence = 0.72
        reason = "Workflow outputs are usable with warnings."
    else:
        status = WorkflowReadinessStatus.ready
        usable = True
        confidence = 0.9
        reason = "Workflow outputs are ready."
    return WorkflowReadiness(
        status=status,
        usable=usable,
        confidence=confidence,
        reason=reason,
        blocking_issues=blocking,
        warnings=warnings[:50],
        recommended_next_actions=_recommended_actions(status, required_review),
        required_human_review=required_review,
    )


def build_policy_warnings(workflow: CompiledWorkflow) -> list[WorkflowWarning]:
    return evaluate_pre_execution_policies(workflow)


def redact_secrets(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            lower = str(key).lower()
            if any(marker in lower for marker in SECRET_MARKERS):
                out[key] = "[REDACTED]"
            else:
                out[key] = redact_secrets(item)
        return out
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def settings_fingerprint(settings: dict) -> str:
    import hashlib

    payload = json.dumps(redact_secrets(settings), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_policy_artifacts(
    run_dir, workflow: CompiledWorkflow, warnings: list[WorkflowWarning]
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "workflow_policies.json").write_text(
        json.dumps(model_to_plain(workflow.policies), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "workflow_policy_warnings.json").write_text(
        json.dumps(model_to_plain(warnings), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "workflow_policies.md").write_text(_render_policies(workflow), encoding="utf-8")
    (run_dir / "workflow_policy_warnings.md").write_text(
        _render_warnings(warnings), encoding="utf-8"
    )


def _model_available(settings: dict) -> bool:
    provider = str(settings.get("model_provider") or "").lower()
    if provider == "mock":
        return True
    if provider == "openai":
        return bool(settings.get("openai_api_key"))
    if provider in {"ollama", "llamacpp"}:
        return True
    return bool(provider)


def _recommended_actions(status: WorkflowReadinessStatus, required_review: bool) -> list[str]:
    actions: list[str] = []
    if status in {WorkflowReadinessStatus.blocked, WorkflowReadinessStatus.failed}:
        actions.append("Inspect failed stages and regenerate missing required artifacts.")
    if status == WorkflowReadinessStatus.degraded:
        actions.append("Review skipped optional subsystems and failed quality gates.")
    if required_review:
        actions.append("Complete human review before relying on the output.")
    if not actions:
        actions.append("Archive workflow artifacts with the run.")
    return actions


def _render_policies(workflow: CompiledWorkflow) -> str:
    lines = ["# Workflow Policies", ""]
    for policy in workflow.policies:
        lines.append(f"## {policy.policy_id}")
        lines.append(f"- Network allowed: {policy.external_network_allowed}")
        lines.append(f"- Model required: {policy.model_required}")
        lines.append(f"- Mock allowed: {policy.mock_allowed}")
        lines.append(f"- Human review: {policy.review_required}")
        lines.append(f"- Quality gate: {policy.quality_gate_id or policy.run_quality_gate}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_warnings(warnings: list[WorkflowWarning]) -> str:
    lines = ["# Workflow Policy Warnings", ""]
    if not warnings:
        lines.append("No policy warnings.")
    for warning in warnings:
        lines.append(f"- `{warning.severity.value}` `{warning.code}`: {warning.message}")
    return "\n".join(lines).rstrip() + "\n"
