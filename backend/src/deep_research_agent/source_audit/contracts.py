from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RecommendedUsage = Literal[
    "cite_directly",
    "use_as_background",
    "use_with_caution",
    "verify_with_primary_source",
    "exclude_from_report",
]

FreshnessStatus = Literal["current", "recent", "possibly_stale", "stale", "unknown"]
BiasRiskLevel = Literal["low", "medium", "high"]
WarningSeverity = Literal["info", "low", "medium", "high"]
SourceRole = Literal["primary", "secondary", "weak", "unknown"]


class SourceAuditWarning(BaseModel):
    code: str
    severity: WarningSeverity = "low"
    message: str


class SourceCredibilityScore(BaseModel):
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    positive_signals: list[str] = Field(default_factory=list)
    negative_signals: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SourceFreshnessScore(BaseModel):
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    status: FreshnessStatus = "unknown"
    freshness_matters: bool = False
    detected_dates: list[str] = Field(default_factory=list)
    best_date: str | None = None
    age_days: int | None = None
    reasons: list[str] = Field(default_factory=list)


class SourceAuthorityScore(BaseModel):
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_role: SourceRole = "unknown"
    authority_signals: list[str] = Field(default_factory=list)
    weakness_signals: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SourceBiasRisk(BaseModel):
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: BiasRiskLevel = "low"
    signals: list[str] = Field(default_factory=list)
    mitigating_factors: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class PrimarySourceAssessment(BaseModel):
    likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    source_role: SourceRole = "unknown"
    primary_type: str | None = None
    signals: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class CitationReadiness(BaseModel):
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_ready: bool = False
    stable_url: bool = False
    has_title: bool = False
    has_date_or_version: bool = False
    sufficient_content: bool = False
    low_warning_count: bool = False
    low_duplication_risk: bool = False
    reasons: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)


class SourceAudit(BaseModel):
    source_id: str
    url: str
    domain: str
    title: str | None = None
    source_type: str = "unknown"
    credibility_score: SourceCredibilityScore
    freshness_score: SourceFreshnessScore
    authority_score: SourceAuthorityScore
    bias_risk_score: SourceBiasRisk
    primary_source_likelihood: PrimarySourceAssessment
    citation_readiness_score: CitationReadiness
    final_source_score: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[SourceAuditWarning] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    recommended_usage: RecommendedUsage = "use_with_caution"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceAuditSummary(BaseModel):
    source_count: int = 0
    usable_source_count: int = 0
    average_final_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ranked_source_ids: list[str] = Field(default_factory=list)
    recommended_primary_sources: list[str] = Field(default_factory=list)
    sources_needing_verification: list[str] = Field(default_factory=list)
    sources_to_avoid: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    freshness_gaps: list[str] = Field(default_factory=list)
    authority_gaps: list[str] = Field(default_factory=list)
    citation_risks: list[str] = Field(default_factory=list)
    instruction_block: str = ""


class SourceAuditBatch(BaseModel):
    thread_id: str | None = None
    question: str
    generated_at: str
    audits: list[SourceAudit] = Field(default_factory=list)
    summary: SourceAuditSummary = Field(default_factory=SourceAuditSummary)


def model_to_plain(value: BaseModel) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[attr-defined]
    return value.dict()
