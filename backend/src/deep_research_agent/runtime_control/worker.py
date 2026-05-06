from __future__ import annotations

import time
import uuid
from datetime import datetime

from deep_research_agent.agent_factory import AgentService
from deep_research_agent.settings import Settings

from .artifact_writer import RuntimeArtifactWriter
from .budgets import RuntimeBudgetTracker
from .contracts import (
    ResearchJob,
    ResearchJobStatus,
    RetryPolicy,
    RuntimeErrorRecord,
    RuntimeEventType,
)
from .errors import RuntimeBudgetExceeded, RuntimeLeaseError
from .events import RuntimeEventWriter
from .repository import RuntimeRepository
from .retries import classify_error, retry_after
from .stages import RuntimeStageExecutor, StageExecutionCancelled, StageExecutionPaused


class RuntimeWorker:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: RuntimeRepository,
        service: AgentService | None = None,
        worker_id: str | None = None,
        retry_policy: RetryPolicy | None = None,
    ):
        self.settings = settings
        self.repository = repository
        self.service = service
        self.worker_id = worker_id or f"worker-{uuid.uuid4()}"
        self.retry_policy = retry_policy or RetryPolicy(
            max_attempts=getattr(settings, "runtime_max_attempts", 3),
            dead_letter_after_attempts=getattr(settings, "runtime_max_attempts", 3),
            backoff_initial_seconds=getattr(settings, "runtime_retry_backoff_initial_seconds", 2.0),
            backoff_multiplier=getattr(settings, "runtime_retry_backoff_multiplier", 2.0),
            backoff_max_seconds=getattr(settings, "runtime_retry_backoff_max_seconds", 60.0),
        )
        self.events = RuntimeEventWriter(repository=repository, runs_dir=settings.runs_dir)
        self.stage_executor = RuntimeStageExecutor(
            settings=settings, repository=repository, service=service
        )
        self._stop = False
        self.current_lease_id: str | None = None

    def process_next_job(self) -> ResearchJob | None:
        job = self.repository.dequeue_next_job()
        if job is None:
            return None
        return self.process_job(job.job_id)

    def process_job(self, job_id: str) -> ResearchJob:
        job = self.repository.get_job(job_id)
        if job.status in {
            ResearchJobStatus.COMPLETED,
            ResearchJobStatus.CANCELLED,
            ResearchJobStatus.DEAD_LETTERED,
        }:
            return job
        now = datetime.now(job.created_at.tzinfo)
        if job.retry_after and job.retry_after > now:
            self.repository.enqueue_job(job_id)
            return job
        lease_seconds = int(getattr(self.settings, "runtime_lease_seconds", 120))
        try:
            lease = self.repository.acquire_lease(
                job_id, worker_id=self.worker_id, lease_seconds=lease_seconds
            )
        except RuntimeLeaseError:
            return self.repository.get_job(job_id)
        self.current_lease_id = lease.lease_id
        try:
            self.repository.mark_job_status(job_id, ResearchJobStatus.LEASED, validate=False)
            job = self.repository.get_job(job_id)
            self.events.emit(
                job,
                RuntimeEventType.JOB_LEASED,
                message=self.worker_id,
                data={"lease_id": lease.lease_id},
            )
            self.repository.mark_job_status(job_id, ResearchJobStatus.RUNNING, validate=False)
            job = self.repository.get_job(job_id)
            job.attempts += 1
            self.repository.update_job(job)
            self.events.emit(job, RuntimeEventType.JOB_STARTED, message="worker started job")
            self._run_stages(job)
            final = self.repository.get_job(job_id)
            return final
        except StageExecutionCancelled:
            return self.repository.get_job(job_id)
        except StageExecutionPaused:
            return self.repository.get_job(job_id)
        except Exception as exc:
            return self._handle_failure(job_id, exc)
        finally:
            if self.current_lease_id:
                self.repository.release_lease(self.current_lease_id)
                self.current_lease_id = None

    def _run_stages(self, job: ResearchJob) -> None:
        while not self._stop:
            self.heartbeat_current_job()
            fresh = self.repository.get_job(job.job_id)
            if fresh.status in {ResearchJobStatus.CANCELLING, ResearchJobStatus.CANCELLED}:
                self.repository.mark_job_status(
                    fresh.job_id, ResearchJobStatus.CANCELLED, validate=False
                )
                self.events.emit(fresh, RuntimeEventType.JOB_CANCELLED, message="cancelled")
                return
            if fresh.status in {ResearchJobStatus.PAUSING, ResearchJobStatus.PAUSED}:
                self.repository.mark_job_status(
                    fresh.job_id, ResearchJobStatus.PAUSED, validate=False
                )
                self.events.emit(fresh, RuntimeEventType.JOB_PAUSED, message="paused")
                return
            if not self.stage_executor.execute_next_stage(fresh):
                self.repository.mark_completed(fresh.job_id)
                fresh = self.repository.get_job(fresh.job_id)
                self.events.emit(fresh, RuntimeEventType.JOB_COMPLETED, message="job completed")
                RuntimeArtifactWriter(
                    runs_dir=self.settings.runs_dir, thread_id=fresh.thread_id
                ).write_job(fresh)
                return

    def _handle_failure(self, job_id: str, exc: BaseException) -> ResearchJob:
        job = self.repository.get_job(job_id)
        error = RuntimeErrorRecord(
            error_id=str(uuid.uuid4()),
            error_type=type(exc).__name__,
            message=str(exc),
            stage=job.stage,
            retryable=classify_error(exc, self.retry_policy) == "retryable",
        )
        job.error = error
        self.repository.update_job(job)
        if isinstance(exc, RuntimeBudgetExceeded):
            self.events.emit(
                job,
                RuntimeEventType.BUDGET_EXCEEDED,
                severity="error",
                message=str(exc),
                data={"reasons": exc.reasons},
            )
        if error.retryable and job.attempts < min(job.max_attempts, self.retry_policy.max_attempts):
            tracker = RuntimeBudgetTracker(
                repository=self.repository, runs_dir=self.settings.runs_dir, job=job
            )
            tracker.increment_retries()
            tracker.persist()
            retry_at = retry_after(job.attempts, self.retry_policy)
            self.repository.mark_job_status(job_id, ResearchJobStatus.FAILED, validate=False)
            self.repository.requeue_job(job_id, retry_after=retry_at)
            job = self.repository.get_job(job_id)
            self.events.emit(
                job,
                RuntimeEventType.RETRY_SCHEDULED,
                severity="warning",
                message=str(exc),
                data={"retry_after": retry_at.isoformat()},
            )
            return job
        self.repository.mark_failed(job_id, error)
        if getattr(self.settings, "runtime_dead_letter_enabled", True):
            self.repository.mark_dead_lettered(job_id, reason=str(exc))
            job = self.repository.get_job(job_id)
            RuntimeArtifactWriter(
                runs_dir=self.settings.runs_dir, thread_id=job.thread_id
            ).write_json(
                "runtime_dead_letter.json",
                {"job_id": job.job_id, "thread_id": job.thread_id, "error": error},
            )
            self.events.emit(
                job,
                RuntimeEventType.JOB_DEAD_LETTERED,
                severity="error",
                message=str(exc),
            )
            return job
        self.events.emit(job, RuntimeEventType.JOB_FAILED, severity="error", message=str(exc))
        return self.repository.get_job(job_id)

    def heartbeat_current_job(self):
        if not self.current_lease_id:
            return None
        return self.repository.heartbeat_lease(
            self.current_lease_id,
            lease_seconds=int(getattr(self.settings, "runtime_lease_seconds", 120)),
        )

    def check_control_requests(self, job_id: str) -> ResearchJob:
        return self.repository.get_job(job_id)

    def execute_stage(self, job_id: str, stage) -> None:
        self.stage_executor.execute_stage(self.repository.get_job(job_id), stage)

    def run_loop(self) -> None:
        interval = float(getattr(self.settings, "runtime_worker_poll_interval_seconds", 1.0))
        while not self._stop:
            self.process_next_job()
            time.sleep(interval)

    def stop(self) -> None:
        self._stop = True
