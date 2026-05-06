from __future__ import annotations

from .contracts import ResearchPass, stable_id, utc_now

PASS_NAMES = {
    "request_analysis": "Request analysis",
    "blueprint_generation": "Blueprint generation",
    "source_inventory": "Source inventory",
    "source_unitization": "Source unitization",
    "evidence_unitization": "Evidence unitization",
    "agent_execution": "Existing agent execution",
    "report_critique": "Report critique",
    "claim_verification": "Claim verification",
    "confidence_calibration": "Confidence calibration",
    "final_kernel_summary": "Final kernel summary",
}


def create_pass(pass_type: str, *, required: bool, thread_id: str) -> ResearchPass:
    return ResearchPass(
        pass_id=stable_id("pass", thread_id, pass_type),
        pass_type=pass_type,  # type: ignore[arg-type]
        name=PASS_NAMES.get(pass_type, pass_type.replace("_", " ")),
        required=required,
    )


def mark_running(record: ResearchPass) -> ResearchPass:
    record.status = "running"
    record.started_at = utc_now()
    return record


def mark_completed(
    record: ResearchPass, output_artifacts: list[str] | None = None, metrics: dict | None = None
) -> ResearchPass:
    record.status = "completed"
    record.completed_at = utc_now()
    if output_artifacts:
        record.output_artifacts.extend(output_artifacts)
    if metrics:
        record.metrics.update(metrics)
    return record


def mark_skipped(record: ResearchPass, reason: str) -> ResearchPass:
    record.status = "skipped"
    record.skipped_reason = reason
    record.completed_at = utc_now()
    return record


def mark_failed(record: ResearchPass, reason: str) -> ResearchPass:
    record.status = "failed"
    record.failure_reason = reason
    record.completed_at = utc_now()
    return record
