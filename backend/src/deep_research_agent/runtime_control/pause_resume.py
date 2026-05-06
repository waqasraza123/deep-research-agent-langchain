from __future__ import annotations

from pathlib import Path

from .contracts import (
    ControlRequestStatus,
    PauseRequest,
    ResearchJobStatus,
    ResumeRequest,
    RuntimeEventType,
)
from .events import RuntimeEventWriter
from .repository import RuntimeRepository
from .state_machine import is_terminal


class PauseResumeService:
    def __init__(self, *, repository: RuntimeRepository, runs_dir: Path):
        self.repository = repository
        self.events = RuntimeEventWriter(repository=repository, runs_dir=runs_dir)

    def request_pause(self, *, job_id: str, requested_by: str = "operator", reason: str = ""):
        job = self.repository.get_job(job_id)
        request = PauseRequest(job_id=job_id, requested_by=requested_by, reason=reason)
        if is_terminal(job.status):
            request.status = ControlRequestStatus.REJECTED
            self.repository.request_pause(request)
            return job
        self.repository.request_pause(request)
        self.events.emit(
            job,
            RuntimeEventType.PAUSE_REQUESTED,
            message=reason or "pause requested",
            data={"requested_by": requested_by},
        )
        if job.status == ResearchJobStatus.QUEUED:
            self.repository.mark_job_status(job_id, ResearchJobStatus.PAUSED, validate=False)
            request.status = ControlRequestStatus.COMPLETED
            self.repository.request_pause(request)
            job = self.repository.get_job(job_id)
            self.events.emit(job, RuntimeEventType.JOB_PAUSED, message="queued job paused")
            return job
        self.repository.mark_job_status(job_id, ResearchJobStatus.PAUSING, validate=False)
        request.status = ControlRequestStatus.ACKNOWLEDGED
        self.repository.request_pause(request)
        return self.repository.get_job(job_id)

    def request_resume(self, *, job_id: str, requested_by: str = "operator", reason: str = ""):
        job = self.repository.get_job(job_id)
        request = ResumeRequest(job_id=job_id, requested_by=requested_by, reason=reason)
        if job.status != ResearchJobStatus.PAUSED:
            request.status = ControlRequestStatus.REJECTED
            self.repository.request_resume(request)
            return job
        self.repository.request_resume(request)
        self.repository.mark_job_status(job_id, ResearchJobStatus.RESUME_REQUESTED, validate=False)
        job = self.repository.mark_job_status(
            job_id, ResearchJobStatus.QUEUED, validate=False
        )
        self.repository.enqueue_job(job_id)
        request.status = ControlRequestStatus.COMPLETED
        self.repository.request_resume(request)
        self.events.emit(
            job,
            RuntimeEventType.RESUME_REQUESTED,
            message=reason or "resume requested",
        )
        self.events.emit(job, RuntimeEventType.JOB_RESUMED, message="job requeued")
        return job
