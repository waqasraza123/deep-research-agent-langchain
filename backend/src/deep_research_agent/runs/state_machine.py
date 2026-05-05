from __future__ import annotations

from .contracts import RunStage, RunStatus


class InvalidRunTransitionError(ValueError):
    def __init__(self, current: RunStatus, target: RunStatus):
        super().__init__(f"Invalid run transition: {current.value} -> {target.value}")
        self.current = current
        self.target = target


ACTIVE_STATUSES = {
    RunStatus.CREATED,
    RunStatus.PLANNING,
    RunStatus.FETCHING_SOURCES,
    RunStatus.ANALYZING,
    RunStatus.WRITING_REPORT,
    RunStatus.BUILDING_EVIDENCE,
    RunStatus.WAITING_FOR_REVIEW,
}

TERMINAL_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
}

VALID_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.CREATED: {RunStatus.PLANNING, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.PLANNING: {RunStatus.FETCHING_SOURCES, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.FETCHING_SOURCES: {RunStatus.ANALYZING, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.ANALYZING: {RunStatus.WRITING_REPORT, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.WRITING_REPORT: {RunStatus.BUILDING_EVIDENCE, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.BUILDING_EVIDENCE: {
        RunStatus.WAITING_FOR_REVIEW,
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.WAITING_FOR_REVIEW: {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
    RunStatus.CANCELLED: set(),
}

STATUS_TO_STAGE: dict[RunStatus, RunStage] = {
    RunStatus.CREATED: RunStage.CREATED,
    RunStatus.PLANNING: RunStage.PLANNING,
    RunStatus.FETCHING_SOURCES: RunStage.SOURCE_FETCHING,
    RunStatus.ANALYZING: RunStage.AGENT_ANALYSIS,
    RunStatus.WRITING_REPORT: RunStage.REPORT_WRITING,
    RunStatus.BUILDING_EVIDENCE: RunStage.EVIDENCE_BUILDING,
    RunStatus.WAITING_FOR_REVIEW: RunStage.REVIEW,
    RunStatus.COMPLETED: RunStage.COMPLETED,
    RunStatus.FAILED: RunStage.FAILED,
    RunStatus.CANCELLED: RunStage.CANCELLED,
}


def validate_transition(current: RunStatus, target: RunStatus) -> None:
    if target not in VALID_TRANSITIONS[current]:
        raise InvalidRunTransitionError(current, target)


def transition_status(current: RunStatus, target: RunStatus) -> RunStage:
    validate_transition(current, target)
    return STATUS_TO_STAGE[target]


def is_active(status: RunStatus) -> bool:
    return status in ACTIVE_STATUSES
