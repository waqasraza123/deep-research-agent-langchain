from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DateType = Literal[
    "published",
    "updated",
    "accessed",
    "effective",
    "expired",
    "version_release",
    "deadline",
    "event_date",
    "mentioned_date",
    "unknown",
]

DatePrecision = Literal["day", "month", "year", "unknown"]

VersionSignalType = Literal[
    "major_version",
    "semantic_version",
    "release_notes",
    "changelog",
    "deprecation",
    "migration_guide",
    "old_docs",
    "archived_docs",
    "legacy_docs",
    "stable_docs",
    "current_docs",
    "latest_docs",
    "beta_docs",
    "preview_docs",
]

CurrentnessStatus = Literal[
    "current",
    "probably_current",
    "possibly_stale",
    "stale",
    "unknown",
]

ClaimTemporalStatus = Literal[
    "temporally_supported",
    "temporally_weak",
    "stale_source_risk",
    "missing_date_support",
    "contradictory_dates",
    "unknown",
]

WarningSeverity = Literal["info", "low", "medium", "high", "critical"]


class ExtractedDate(BaseModel):
    raw_text: str
    normalized_date: str | None = None
    date_type: DateType = "unknown"
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_id: str | None = None
    source_url: str | None = None
    text_offset: int | None = None
    surrounding_context: str = ""
    precision: DatePrecision = "unknown"
    origin: str = "text"


class SourceVersionSignal(BaseModel):
    source_id: str | None = None
    source_url: str | None = None
    signal_type: VersionSignalType
    raw_text: str
    normalized_version: str | None = None
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    text_offset: int | None = None
    surrounding_context: str = ""
    outdated_hint: bool = False
    current_hint: bool = False


class TimelineEvent(BaseModel):
    event_id: str
    date: str
    date_precision: DatePrecision = "unknown"
    event_type: DateType | Literal["claim_date", "version_signal"] = "unknown"
    description: str
    source_id: str | None = None
    source_url: str | None = None
    claim_id: str | None = None
    raw_text: str = ""
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    origin: str = ""


class SourceTemporalMetadata(BaseModel):
    source_id: str
    source_url: str = ""
    title: str | None = None
    local_path: str | None = None
    fetched_at: str | None = None
    extracted_dates: list[ExtractedDate] = Field(default_factory=list)
    version_signals: list[SourceVersionSignal] = Field(default_factory=list)
    best_publication_date: str | None = None
    best_update_date: str | None = None
    newest_date: str | None = None
    oldest_date: str | None = None
    currentness_status: CurrentnessStatus = "unknown"
    currentness_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TemporalWarning(BaseModel):
    warning_id: str
    severity: WarningSeverity = "medium"
    category: str
    message: str
    source_id: str | None = None
    source_url: str | None = None
    claim_id: str | None = None
    evidence: list[str] = Field(default_factory=list)


class CurrentnessAssessment(BaseModel):
    thread_id: str
    question: str = ""
    generated_at: str
    status: CurrentnessStatus = "unknown"
    freshness_required: bool = False
    freshness_signals: list[str] = Field(default_factory=list)
    source_count: int = 0
    newest_source_date: str | None = None
    oldest_source_date: str | None = None
    stale_sources: list[str] = Field(default_factory=list)
    unknown_date_sources: list[str] = Field(default_factory=list)
    source_assessments: list[SourceTemporalMetadata] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[TemporalWarning] = Field(default_factory=list)
    temporal_warning_block: str = ""


class TimeSensitiveClaim(BaseModel):
    claim_id: str
    text: str
    origin: Literal["report", "notes"] = "report"
    origin_ref: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    detected_dates: list[ExtractedDate] = Field(default_factory=list)
    detected_versions: list[str] = Field(default_factory=list)
    temporal_language: list[str] = Field(default_factory=list)
    status: ClaimTemporalStatus = "unknown"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    support_source_dates: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class TemporalProfile(BaseModel):
    thread_id: str
    question: str = ""
    generated_at: str
    freshness_required: bool = False
    freshness_signals: list[str] = Field(default_factory=list)
    sources: list[SourceTemporalMetadata] = Field(default_factory=list)
    extracted_dates: list[ExtractedDate] = Field(default_factory=list)
    version_signals: list[SourceVersionSignal] = Field(default_factory=list)
    timeline_events: list[TimelineEvent] = Field(default_factory=list)
    warnings: list[TemporalWarning] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TemporalResearchSummary(BaseModel):
    thread_id: str
    question: str = ""
    generated_at: str
    freshness_required: bool = False
    newest_source_date: str | None = None
    oldest_source_date: str | None = None
    source_count: int = 0
    stale_source_count: int = 0
    unknown_date_source_count: int = 0
    date_sensitive_claim_count: int = 0
    warning_count: int = 0
    currentness_status: CurrentnessStatus = "unknown"
    temporal_warning_block: str = ""


def model_to_plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value.dict()
