from __future__ import annotations

from typing import Any

from .contracts import RunStatus
from .repository import RunCancelledError, RunRepository


class RunLifecycle:
    def __init__(self, repository: RunRepository, thread_id: str):
        self.repository = repository
        self.thread_id = thread_id

    def transition(self, status: RunStatus) -> None:
        self.repository.raise_if_cancelled(self.thread_id)
        self.repository.transition(self.thread_id, status)

    def checkpoint(self) -> None:
        self.repository.raise_if_cancelled(self.thread_id)
        self.repository.refresh_artifacts(self.thread_id)

    def complete(self, *, require_review: bool, summary: str = "") -> None:
        self.repository.raise_if_cancelled(self.thread_id)
        self.repository.set_output_summary(self.thread_id, summary=summary)
        self.repository.refresh_artifacts(self.thread_id)
        self.repository.transition(
            self.thread_id,
            RunStatus.WAITING_FOR_REVIEW if require_review else RunStatus.COMPLETED,
        )

    def fail(self, error: BaseException, *, details: dict[str, Any] | None = None) -> None:
        if isinstance(error, RunCancelledError):
            return
        self.repository.record_error(self.thread_id, error, details=details, fail_run=True)
