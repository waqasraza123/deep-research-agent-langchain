from __future__ import annotations

from .contracts import (
    WorkflowInput,
    WorkflowMode,
    WorkflowRebuildRequest,
)


class WorkflowPreviewRequest(WorkflowInput):
    write_artifacts: bool = False


class WorkflowCompileRequest(WorkflowInput):
    write_artifacts: bool = True


class WorkflowRunRequest(WorkflowInput):
    run_now: bool = True


class WorkflowQualityGateRequest(WorkflowInput):
    quality_gate_id: str | None = None
    run_quality_gate: bool = True
    mode: WorkflowMode | None = WorkflowMode.offline_benchmark


__all__ = [
    "WorkflowCompileRequest",
    "WorkflowPreviewRequest",
    "WorkflowQualityGateRequest",
    "WorkflowRebuildRequest",
    "WorkflowRunRequest",
]
