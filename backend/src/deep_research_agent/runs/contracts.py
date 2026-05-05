from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    FETCHING_SOURCES = "fetching_sources"
    ANALYZING = "analyzing"
    WRITING_REPORT = "writing_report"
    BUILDING_EVIDENCE = "building_evidence"
    WAITING_FOR_REVIEW = "waiting_for_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStage(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    SOURCE_FETCHING = "source_fetching"
    AGENT_ANALYSIS = "agent_analysis"
    REPORT_WRITING = "report_writing"
    EVIDENCE_BUILDING = "evidence_building"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReviewState(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    REJECTED = "rejected"


class ResumePointKind(str, Enum):
    AFTER_PLANNING = "after_planning"
    AFTER_SOURCE_FETCHING = "after_source_fetching"
    AFTER_AGENT_ANALYSIS = "after_agent_analysis"
    AFTER_REPORT_WRITING = "after_report_writing"
    AFTER_EVIDENCE_BUILDING = "after_evidence_building"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()


def parse_run(data: dict[str, Any]) -> "ResearchRun":
    validate = getattr(ResearchRun, "model_validate", None)
    if callable(validate):
        return validate(data)
    return ResearchRun.parse_obj(data)


class RunError(BaseModel):
    type: str
    message: str
    stage: RunStage | None = None
    status: RunStatus | None = None
    timestamp: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)


class RunInputSnapshot(BaseModel):
    question: str
    urls: list[str] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)


class RunOutputSummary(BaseModel):
    summary: str = ""
    artifacts: list[str] = Field(default_factory=list)
    artifact_count: int = 0
    report_path: str | None = None
    budget_summary: dict[str, Any] | None = None


class RunReviewStatus(BaseModel):
    status: ReviewState = ReviewState.NOT_REQUIRED
    reviewer: str | None = None
    notes: str = ""
    requested_changes: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    approved_at: datetime | None = None
    rejected_at: datetime | None = None


class RunResumePoint(BaseModel):
    resumable: bool
    point: ResumePointKind | None = None
    reason: str
    required_artifacts: list[str] = Field(default_factory=list)
    present_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)


class RunCancellationRequest(BaseModel):
    requested: bool = True
    requested_at: datetime = Field(default_factory=utc_now)
    requested_by: str | None = None
    reason: str | None = None


class ResearchRun(BaseModel):
    thread_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    question: str
    urls: list[str] = Field(default_factory=list)
    input_snapshot: RunInputSnapshot
    status: RunStatus = RunStatus.CREATED
    current_stage: RunStage = RunStage.CREATED
    artifacts: list[str] = Field(default_factory=list)
    errors: list[RunError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    budget_summary: dict[str, Any] | None = None
    review: RunReviewStatus = Field(default_factory=RunReviewStatus)
    output_summary: RunOutputSummary = Field(default_factory=RunOutputSummary)
    resume_point: RunResumePoint | None = None
    cancellation: RunCancellationRequest | None = None
