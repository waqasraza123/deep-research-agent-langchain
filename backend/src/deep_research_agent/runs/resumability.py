from __future__ import annotations

from pathlib import Path

from deep_research_agent.artifacts import REQUIRED_FILES, list_artifacts, safe_thread_id

from .contracts import ResearchRun, ResumePointKind, RunResumePoint, RunStatus

RESUME_REQUIREMENTS: list[tuple[ResumePointKind, list[str], str]] = [
    (ResumePointKind.AFTER_EVIDENCE_BUILDING, list(REQUIRED_FILES), "required artifacts exist"),
    (ResumePointKind.AFTER_REPORT_WRITING, ["report.md"], "report.md exists"),
    (ResumePointKind.AFTER_AGENT_ANALYSIS, ["notes.md"], "notes.md exists"),
    (ResumePointKind.AFTER_SOURCE_FETCHING, ["sources.json"], "sources.json exists"),
    (ResumePointKind.AFTER_PLANNING, ["plan.md"], "plan.md exists"),
]


def inspect_resume_point(
    runs_dir: Path, thread_id: str, run: ResearchRun | None = None
) -> RunResumePoint:
    safe_thread_id(thread_id)
    present = [artifact.path for artifact in list_artifacts(runs_dir, thread_id)]
    present_set = set(present)

    if run and run.status == RunStatus.COMPLETED:
        return RunResumePoint(
            resumable=False,
            reason="run already completed",
            present_artifacts=present,
        )

    if run and run.status == RunStatus.CANCELLED and not present:
        return RunResumePoint(
            resumable=False,
            reason="run was cancelled before resumable artifacts were created",
            present_artifacts=present,
        )

    for point, required, reason in RESUME_REQUIREMENTS:
        missing = [path for path in required if path not in present_set]
        if not missing:
            return RunResumePoint(
                resumable=True,
                point=point,
                reason=reason,
                required_artifacts=required,
                present_artifacts=present,
                missing_artifacts=[],
            )

    return RunResumePoint(
        resumable=False,
        reason="no recognized resume artifacts found",
        present_artifacts=present,
        missing_artifacts=list(REQUIRED_FILES),
    )
