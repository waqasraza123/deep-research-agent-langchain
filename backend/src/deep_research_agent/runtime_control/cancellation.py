from __future__ import annotations

from pathlib import Path

from .contracts import (
    CancellationRequest,
    ControlRequestStatus,
    ResearchJobStatus,
    RuntimeEventType,
)
from .events import RuntimeEventWriter
from .repository import RuntimeRepository
from .state_machine import is_terminal


class CancellationService:
    def __init__(self, *, repository: RuntimeRepository, runs_dir: Path):
        self.repository = repository
        self.events = RuntimeEventWriter(repository=repository, runs_dir=runs_dir)

    def request_cancel(
        self,
        *,
        job_id: str,
        requested_by: str = "operator",
        reason: str = "",
        force: bool = False,
    ):
        job = self.repository.get_job(job_id)
        request = CancellationRequest(
            job_id=job_id,
            requested_by=requested_by,
            reason=reason,
            force=force,
        )
        self.repository.request_cancel(request)
        self.events.emit(
            job,
            RuntimeEventType.CANCELLATION_REQUESTED,
            message=reason or "cancellation requested",
            data={"force": force, "requested_by": requested_by},
        )
        if is_terminal(job.status):
            request.status = ControlRequestStatus.REJECTED
            self.repository.request_cancel(request)
            return self.repository.get_job(job_id)
        if job.status == ResearchJobStatus.QUEUED:
            self.repository.mark_job_status(job_id, ResearchJobStatus.CANCELLED, validate=False)
            request.status = ControlRequestStatus.COMPLETED
            self.repository.request_cancel(request)
            job = self.repository.get_job(job_id)
            self.events.emit(job, RuntimeEventType.JOB_CANCELLED, message="queued job cancelled")
            return job
        if force:
            warning = (
                "Force cancellation updated runtime state immediately; any blocking model call "
                "already in progress may continue until the underlying call returns."
            )
            self.repository.append_warning(job_id, warning)
            self.repository.mark_job_status(job_id, ResearchJobStatus.CANCELLED, validate=False)
            request.status = ControlRequestStatus.COMPLETED
            self.repository.request_cancel(request)
            job = self.repository.get_job(job_id)
            self.events.emit(
                job,
                RuntimeEventType.JOB_CANCELLED,
                message=warning,
                severity="warning",
            )
            return job
        self.repository.mark_job_status(job_id, ResearchJobStatus.CANCELLING, validate=False)
        request.status = ControlRequestStatus.ACKNOWLEDGED
        self.repository.request_cancel(request)
        return self.repository.get_job(job_id)
