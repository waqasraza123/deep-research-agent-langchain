from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JsonModel(BaseModel):
    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class ResearchJobStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUME_REQUESTED = "resume_requested"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"


class ResearchStage(StrEnum):
    CREATED = "created"
    INPUT_SNAPSHOT = "input_snapshot"
    PLANNING = "planning"
    SOURCE_FETCHING = "source_fetching"
    SOURCE_PROCESSING = "source_processing"
    AGENT_EXECUTION = "agent_execution"
    ARTIFACT_BACKFILL = "artifact_backfill"
    INTELLIGENCE_POSTPROCESSING = "intelligence_postprocessing"
    VERIFICATION = "verification"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


ORDERED_STAGES: tuple[ResearchStage, ...] = (
    ResearchStage.INPUT_SNAPSHOT,
    ResearchStage.PLANNING,
    ResearchStage.SOURCE_FETCHING,
    ResearchStage.SOURCE_PROCESSING,
    ResearchStage.AGENT_EXECUTION,
    ResearchStage.ARTIFACT_BACKFILL,
    ResearchStage.INTELLIGENCE_POSTPROCESSING,
    ResearchStage.VERIFICATION,
    ResearchStage.FINALIZATION,
)


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LeaseStatus(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    RELEASED = "released"
    STOLEN = "stolen"
    CANCELLED = "cancelled"


class RuntimeEventType(StrEnum):
    JOB_SUBMITTED = "job_submitted"
    JOB_QUEUED = "job_queued"
    JOB_LEASED = "job_leased"
    JOB_STARTED = "job_started"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    STAGE_SKIPPED = "stage_skipped"
    ARTIFACT_WRITTEN = "artifact_written"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLATION_REQUESTED = "cancellation_requested"
    JOB_CANCELLED = "job_cancelled"
    PAUSE_REQUESTED = "pause_requested"
    JOB_PAUSED = "job_paused"
    RESUME_REQUESTED = "resume_requested"
    JOB_RESUMED = "job_resumed"
    RETRY_SCHEDULED = "retry_scheduled"
    JOB_FAILED = "job_failed"
    JOB_DEAD_LETTERED = "job_dead_lettered"
    JOB_COMPLETED = "job_completed"
    RECOVERY_DETECTED = "recovery_detected"
    RECOVERY_APPLIED = "recovery_applied"


class RuntimeLease(JsonModel):
    lease_id: str
    job_id: str
    worker_id: str
    acquired_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    heartbeat_at: datetime | None = None
    heartbeat_count: int = 0
    status: LeaseStatus = LeaseStatus.ACTIVE
    lost_reason: str | None = None


class RuntimeBudget(JsonModel):
    max_runtime_seconds: int = Field(default=900, ge=0)
    max_stage_seconds: int = Field(default=300, ge=0)
    max_source_fetches: int = Field(default=3, ge=0)
    max_model_calls: int = Field(default=25, ge=0)
    max_artifact_bytes: int = Field(default=50_000_000, ge=0)
    max_retries: int = Field(default=3, ge=0)
    max_events: int = Field(default=5000, ge=0)
    max_total_sources: int = Field(default=20, ge=0)
    max_total_extracted_chars: int = Field(default=300_000, ge=0)
    fail_on_budget_exceeded: bool = False


class RuntimeBudgetUsage(JsonModel):
    runtime_seconds: float = 0.0
    stage_runtime_seconds: dict[str, float] = Field(default_factory=dict)
    source_fetches: int = 0
    model_calls: int = 0
    artifact_bytes: int = 0
    retries: int = 0
    events: int = 0
    total_sources: int = 0
    total_extracted_chars: int = 0
    exceeded: bool = False
    exceeded_reasons: list[str] = Field(default_factory=list)


class RuntimeErrorRecord(JsonModel):
    error_id: str
    error_type: str
    message: str
    stage: ResearchStage | None = None
    retryable: bool = False
    traceback_ref: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    context: dict[str, Any] = Field(default_factory=dict)


class ResearchStageRecord(JsonModel):
    stage_id: str
    job_id: str
    thread_id: str
    stage: ResearchStage
    status: StageStatus = StageStatus.PENDING
    required: bool = True
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    attempts: int = 0
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    checkpoint_marker: str | None = None
    resumable: bool = True
    skip_reason: str | None = None
    error: RuntimeErrorRecord | None = None
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class ResearchJob(JsonModel):
    job_id: str
    thread_id: str
    idempotency_key: str | None = None
    question: str
    urls: list[str] = Field(default_factory=list)
    settings_snapshot: dict[str, Any] = Field(default_factory=dict)
    status: ResearchJobStatus = ResearchJobStatus.QUEUED
    stage: ResearchStage = ResearchStage.CREATED
    priority: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    cancelled_at: datetime | None = None
    paused_at: datetime | None = None
    resume_requested_at: datetime | None = None
    attempts: int = 0
    max_attempts: int = 3
    retry_after: datetime | None = None
    lease: RuntimeLease | None = None
    budget: RuntimeBudget = Field(default_factory=RuntimeBudget)
    budget_usage: RuntimeBudgetUsage = Field(default_factory=RuntimeBudgetUsage)
    error: RuntimeErrorRecord | None = None
    warnings: list[str] = Field(default_factory=list)
    artifact_summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeEvent(JsonModel):
    event_id: str
    job_id: str
    thread_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    event_type: RuntimeEventType
    stage: ResearchStage | None = None
    severity: str = "info"
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[str] = Field(default_factory=list)


class RetryPolicy(JsonModel):
    max_attempts: int = Field(default=3, ge=1)
    backoff_initial_seconds: float = Field(default=2.0, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1)
    backoff_max_seconds: float = Field(default=60.0, ge=0)
    retryable_error_types: list[str] = Field(
        default_factory=lambda: [
            "TimeoutError",
            "ConnectionError",
            "RuntimeTransientError",
            "HTTPStatusError",
        ]
    )
    non_retryable_error_types: list[str] = Field(
        default_factory=lambda: [
            "ValueError",
            "ValidationError",
            "PermissionError",
            "SecurityError",
            "PathTraversalError",
            "MissingConfigurationError",
        ]
    )
    dead_letter_after_attempts: int = Field(default=3, ge=1)


class ControlRequestStatus(StrEnum):
    REQUESTED = "requested"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    REJECTED = "rejected"


class CancellationRequest(JsonModel):
    job_id: str
    requested_at: datetime = Field(default_factory=utc_now)
    requested_by: str = "operator"
    reason: str = ""
    force: bool = False
    status: ControlRequestStatus = ControlRequestStatus.REQUESTED


class PauseRequest(JsonModel):
    job_id: str
    requested_at: datetime = Field(default_factory=utc_now)
    requested_by: str = "operator"
    reason: str = ""
    status: ControlRequestStatus = ControlRequestStatus.REQUESTED


class ResumeRequest(JsonModel):
    job_id: str
    requested_at: datetime = Field(default_factory=utc_now)
    requested_by: str = "operator"
    reason: str = ""
    from_stage: ResearchStage | None = None
    status: ControlRequestStatus = ControlRequestStatus.REQUESTED


class RuntimeDiagnostics(JsonModel):
    runtime_enabled: bool
    queue_backend: str
    repository_backend: str
    worker_count_seen: int = 0
    queued_jobs: int = 0
    running_jobs: int = 0
    paused_jobs: int = 0
    failed_jobs: int = 0
    dead_lettered_jobs: int = 0
    completed_jobs: int = 0
    expired_leases: int = 0
    stale_running_jobs: int = 0
    last_recovery_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)
    sqlite_path: str | None = None
    settings_summary: dict[str, Any] = Field(default_factory=dict)


class RecoveryPlan(JsonModel):
    job_id: str
    thread_id: str
    resumable: bool
    next_stage: ResearchStage | None
    completed_stages: list[ResearchStage] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    unsafe_to_resume_reasons: list[str] = Field(default_factory=list)
    recommended_action: str = "resume"


class DeadLetterRecord(JsonModel):
    job_id: str
    thread_id: str
    moved_at: datetime = Field(default_factory=utc_now)
    reason: str
    error: RuntimeErrorRecord | None = None
    attempts: int = 0
