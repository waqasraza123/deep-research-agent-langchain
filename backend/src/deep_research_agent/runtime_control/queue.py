from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .artifact_writer import RuntimeArtifactWriter
from .contracts import ResearchJob, RuntimeBudget, RuntimeEventType
from .events import RuntimeEventWriter
from .idempotency import compute_idempotency_key, normalize_urls, redact_settings_snapshot
from .repository import RuntimeRepository


class RuntimeQueue:
    def __init__(self, *, repository: RuntimeRepository, runs_dir: Path):
        self.repository = repository
        self.runs_dir = runs_dir
        self.events = RuntimeEventWriter(repository=repository, runs_dir=runs_dir)

    def submit_job(
        self,
        *,
        question: str,
        urls: list[str],
        settings_snapshot: dict[str, Any] | None = None,
        thread_id: str | None = None,
        idempotency_key: str | None = None,
        priority: int = 0,
        budget: RuntimeBudget | None = None,
        max_attempts: int = 3,
        resubmit_completed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[ResearchJob, bool]:
        clean_urls = normalize_urls(urls)
        safe_settings = redact_settings_snapshot(settings_snapshot or {})
        key = compute_idempotency_key(
            question=question,
            urls=clean_urls,
            settings_snapshot=safe_settings,
            explicit_key=idempotency_key,
        )
        if not resubmit_completed:
            existing = self.repository.get_job_by_idempotency_key(key, active_only=True)
            if existing is not None:
                return existing, False

        job = ResearchJob(
            job_id=str(uuid.uuid4()),
            thread_id=thread_id or str(uuid.uuid4()),
            idempotency_key=key,
            question=question.strip(),
            urls=clean_urls,
            settings_snapshot=safe_settings,
            priority=priority,
            budget=budget or RuntimeBudget(),
            max_attempts=max_attempts,
            metadata=metadata or {},
        )
        self.repository.create_job(job)
        writer = RuntimeArtifactWriter(runs_dir=self.runs_dir, thread_id=job.thread_id)
        writer.write_input_snapshot(job=job, request={"question": question, "urls": urls})
        writer.write_job(job)
        self.events.emit(job, RuntimeEventType.JOB_SUBMITTED, message=job.question)
        self.repository.enqueue_job(job.job_id)
        self.events.emit(job, RuntimeEventType.JOB_QUEUED, message="job queued")
        return job, True

