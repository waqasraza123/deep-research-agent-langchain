from __future__ import annotations

from .contracts import ReviewState, RunReviewStatus, RunStatus, utc_now
from .repository import RunRepository


class ReviewRequestError(ValueError):
    pass


def get_review(repository: RunRepository, thread_id: str) -> RunReviewStatus:
    return repository.get(thread_id).review


def approve_review(
    repository: RunRepository,
    thread_id: str,
    *,
    reviewer: str,
    notes: str = "",
) -> RunReviewStatus:
    run = repository.get(thread_id)
    now = utc_now()
    review = run.review
    review.status = ReviewState.APPROVED
    review.reviewer = reviewer
    review.notes = notes
    review.approved_at = now
    review.updated_at = now
    repository.update_review(thread_id, review)
    if run.status == RunStatus.WAITING_FOR_REVIEW:
        repository.transition(thread_id, RunStatus.COMPLETED)
    return repository.get(thread_id).review


def request_changes(
    repository: RunRepository,
    thread_id: str,
    *,
    reviewer: str,
    notes: str = "",
    requested_changes: list[str] | None = None,
) -> RunReviewStatus:
    review = repository.get(thread_id).review
    review.status = ReviewState.CHANGES_REQUESTED
    review.reviewer = reviewer
    review.notes = notes
    review.requested_changes = requested_changes or []
    review.updated_at = utc_now()
    repository.update_review(thread_id, review)
    return repository.get(thread_id).review


def reject_review(
    repository: RunRepository,
    thread_id: str,
    *,
    reviewer: str,
    notes: str = "",
) -> RunReviewStatus:
    run = repository.get(thread_id)
    now = utc_now()
    review = run.review
    review.status = ReviewState.REJECTED
    review.reviewer = reviewer
    review.notes = notes
    review.rejected_at = now
    review.updated_at = now
    repository.update_review(thread_id, review)
    fresh = repository.get(thread_id)
    if fresh.status == RunStatus.WAITING_FOR_REVIEW:
        repository.record_error(
            thread_id,
            "Run rejected during operator review",
            details={"reviewer": reviewer, "notes": notes},
            fail_run=True,
        )
    return repository.get(thread_id).review
