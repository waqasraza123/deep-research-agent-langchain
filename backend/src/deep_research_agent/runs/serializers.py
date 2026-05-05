from __future__ import annotations

from typing import Any

from .contracts import ResearchRun, model_to_dict


def run_summary(run: ResearchRun) -> dict[str, Any]:
    return {
        "thread_id": run.thread_id,
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
        "question": run.question,
        "urls": run.urls,
        "status": run.status.value,
        "current_stage": run.current_stage.value,
        "artifact_count": len(run.artifacts),
        "artifacts": run.artifacts,
        "has_errors": bool(run.errors),
        "error_count": len(run.errors),
        "warnings": run.warnings,
        "budget_summary": run.budget_summary,
        "review_status": run.review.status.value,
        "report_path": run.output_summary.report_path,
        "resume_point": model_to_dict(run.resume_point) if run.resume_point else None,
    }


def run_detail(run: ResearchRun) -> dict[str, Any]:
    return model_to_dict(run)
