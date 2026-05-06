from __future__ import annotations

from .contracts import ResearchJobStatus, StageStatus
from .errors import InvalidRuntimeTransitionError

TERMINAL_JOB_STATUSES = {
    ResearchJobStatus.CANCELLED,
    ResearchJobStatus.COMPLETED,
    ResearchJobStatus.DEAD_LETTERED,
}

VALID_JOB_TRANSITIONS: dict[ResearchJobStatus, set[ResearchJobStatus]] = {
    ResearchJobStatus.QUEUED: {
        ResearchJobStatus.LEASED,
        ResearchJobStatus.CANCELLING,
        ResearchJobStatus.CANCELLED,
        ResearchJobStatus.PAUSED,
    },
    ResearchJobStatus.LEASED: {
        ResearchJobStatus.RUNNING,
        ResearchJobStatus.QUEUED,
        ResearchJobStatus.CANCELLING,
        ResearchJobStatus.CANCELLED,
        ResearchJobStatus.FAILED,
    },
    ResearchJobStatus.RUNNING: {
        ResearchJobStatus.PAUSING,
        ResearchJobStatus.CANCELLING,
        ResearchJobStatus.COMPLETED,
        ResearchJobStatus.FAILED,
        ResearchJobStatus.QUEUED,
    },
    ResearchJobStatus.PAUSING: {
        ResearchJobStatus.PAUSED,
        ResearchJobStatus.CANCELLING,
        ResearchJobStatus.CANCELLED,
    },
    ResearchJobStatus.PAUSED: {
        ResearchJobStatus.RESUME_REQUESTED,
        ResearchJobStatus.CANCELLING,
        ResearchJobStatus.CANCELLED,
    },
    ResearchJobStatus.RESUME_REQUESTED: {
        ResearchJobStatus.QUEUED,
        ResearchJobStatus.CANCELLING,
        ResearchJobStatus.CANCELLED,
    },
    ResearchJobStatus.CANCELLING: {ResearchJobStatus.CANCELLED},
    ResearchJobStatus.FAILED: {
        ResearchJobStatus.QUEUED,
        ResearchJobStatus.DEAD_LETTERED,
        ResearchJobStatus.CANCELLING,
    },
    ResearchJobStatus.DEAD_LETTERED: {ResearchJobStatus.QUEUED},
    ResearchJobStatus.CANCELLED: set(),
    ResearchJobStatus.COMPLETED: set(),
}

VALID_STAGE_TRANSITIONS: dict[StageStatus, set[StageStatus]] = {
    StageStatus.PENDING: {
        StageStatus.RUNNING,
        StageStatus.SKIPPED,
        StageStatus.CANCELLED,
    },
    StageStatus.RUNNING: {
        StageStatus.COMPLETED,
        StageStatus.FAILED,
        StageStatus.CANCELLED,
    },
    StageStatus.FAILED: {StageStatus.PENDING},
    StageStatus.COMPLETED: set(),
    StageStatus.SKIPPED: set(),
    StageStatus.CANCELLED: set(),
}


def validate_job_transition(
    current: ResearchJobStatus | str,
    target: ResearchJobStatus | str,
    *,
    explicit_restore: bool = False,
) -> None:
    current_status = ResearchJobStatus(current)
    target_status = ResearchJobStatus(target)
    if (
        current_status == ResearchJobStatus.DEAD_LETTERED
        and target_status == ResearchJobStatus.QUEUED
        and explicit_restore
    ):
        return
    if current_status in TERMINAL_JOB_STATUSES:
        raise InvalidRuntimeTransitionError(current_status, target_status)
    if target_status not in VALID_JOB_TRANSITIONS.get(current_status, set()):
        raise InvalidRuntimeTransitionError(current_status, target_status)


def validate_stage_transition(current: StageStatus | str, target: StageStatus | str) -> None:
    current_status = StageStatus(current)
    target_status = StageStatus(target)
    if target_status not in VALID_STAGE_TRANSITIONS.get(current_status, set()):
        raise InvalidRuntimeTransitionError(current_status, target_status)


def is_terminal(status: ResearchJobStatus | str) -> bool:
    return ResearchJobStatus(status) in TERMINAL_JOB_STATUSES

