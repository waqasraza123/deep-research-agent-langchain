from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class VerificationTaskType(str, Enum):
    VERIFY_NUMERIC_CLAIM = "verify_numeric_claim"
    VERIFY_DATE_CLAIM = "verify_date_claim"
    VERIFY_ENTITY_CLAIM = "verify_entity_claim"
    VERIFY_COMPARATIVE_CLAIM = "verify_comparative_claim"
    VERIFY_CAUSAL_CLAIM = "verify_causal_claim"
    VERIFY_RECOMMENDATION = "verify_recommendation"
    VERIFY_PRIMARY_SOURCE_SUPPORT = "verify_primary_source_support"
    VERIFY_FRESHNESS = "verify_freshness"
    VERIFY_CONTRADICTION = "verify_contradiction"
    VERIFY_MISSING_COUNTERARGUMENT = "verify_missing_counterargument"
    VERIFY_UNSUPPORTED_CLAIM = "verify_unsupported_claim"


class VerificationTaskStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    NOT_ENOUGH_INFORMATION = "not_enough_information"
    SKIPPED = "skipped"


FindingSeverity = Literal["info", "low", "medium", "high", "critical"]
EvidencePolarity = Literal["supports", "partially_supports", "contradicts", "context"]


class VerificationEvidence(BaseModel):
    evidence_id: str
    source_id: str | None = None
    source_title: str | None = None
    source_url: str | None = None
    source_artifact: str | None = None
    excerpt: str = ""
    evidence_type: str = "source_text"
    polarity: EvidencePolarity = "context"
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_values: list[str] = Field(default_factory=list)
    reason: str = ""


class VerificationFinding(BaseModel):
    finding_id: str
    task_id: str | None = None
    status: VerificationTaskStatus
    severity: FindingSeverity = "medium"
    claim_or_question: str
    explanation: str
    evidence: list[VerificationEvidence] = Field(default_factory=list)
    confidence_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    suggested_action: str = ""


class VerificationResult(BaseModel):
    task_id: str
    status: VerificationTaskStatus
    confidence_before: float = Field(default=0.35, ge=0.0, le=1.0)
    confidence_after: float = Field(default=0.35, ge=0.0, le=1.0)
    findings: list[VerificationFinding] = Field(default_factory=list)
    evidence: list[VerificationEvidence] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class CriticFinding(BaseModel):
    finding_id: str
    kind: str
    severity: FindingSeverity = "medium"
    priority: int = Field(default=3, ge=1, le=5)
    claim_or_question: str
    source_artifact: str = "report.md"
    reason: str
    expected_evidence_type: str = "source_text"
    suggested_task_type: VerificationTaskType
    candidate_source_ids: list[str] = Field(default_factory=list)
    confidence_before: float = Field(default=0.35, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationTask(BaseModel):
    task_id: str
    task_type: VerificationTaskType
    claim_or_question: str
    source_artifact: str
    priority: int = Field(default=3, ge=1, le=5)
    reason: str
    expected_evidence_type: str
    candidate_source_ids: list[str] = Field(default_factory=list)
    status: VerificationTaskStatus = VerificationTaskStatus.PENDING
    result: VerificationResult | None = None
    confidence_before: float = Field(default=0.35, ge=0.0, le=1.0)
    confidence_after: float = Field(default=0.35, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationPlan(BaseModel):
    plan_id: str
    thread_id: str
    generated_at: str
    method: str = "deterministic_offline_verification"
    max_verification_tasks: int = Field(default=12, ge=0, le=100)
    max_high_priority_tasks: int = Field(default=5, ge=0, le=100)
    tasks: list[VerificationTask] = Field(default_factory=list)
    critic_findings: list[CriticFinding] = Field(default_factory=list)
    artifacts_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ConfidenceCalibration(BaseModel):
    calibration_id: str
    thread_id: str
    generated_at: str
    report_confidence_before: float = Field(default=0.5, ge=0.0, le=1.0)
    report_confidence_after: float = Field(default=0.5, ge=0.0, le=1.0)
    task_confidences: dict[str, float] = Field(default_factory=dict)
    finding_confidences: dict[str, float] = Field(default_factory=dict)
    factors: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    source_diversity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_claim_count: int = 0
    contradiction_count: int = 0
    stale_source_count: int = 0
    primary_source_support_count: int = 0
    exact_value_support_count: int = 0
    confidence_label: Literal["very_low", "low", "medium", "high"] = "low"


class VerificationSummary(BaseModel):
    thread_id: str
    generated_at: str
    total_tasks: int = 0
    verified: int = 0
    partially_verified: int = 0
    contradicted: int = 0
    unsupported: int = 0
    not_enough_information: int = 0
    skipped: int = 0
    high_priority_open_issues: int = 0
    confidence_after: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)


class VerificationBatch(BaseModel):
    batch_id: str
    thread_id: str
    generated_at: str
    plan: VerificationPlan
    results: list[VerificationResult] = Field(default_factory=list)
    findings: list[VerificationFinding] = Field(default_factory=list)
    confidence_calibration: ConfidenceCalibration
    summary: VerificationSummary
    claim_rewrite_suggestions: list[dict[str, Any]] = Field(default_factory=list)


class VerificationConfig(BaseModel):
    max_verification_tasks: int = Field(default=12, ge=0, le=100)
    max_high_priority_tasks: int = Field(default=5, ge=0, le=100)
    verify_numbers: bool = True
    verify_dates: bool = True
    verify_recommendations: bool = True
    require_primary_source_for_sensitive_claims: bool = True
    freshness_verification_required: bool = True
    contradiction_verification_required: bool = True
    verification_gate_enabled: bool = False


def model_to_plain(value: BaseModel) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[attr-defined]
    return value.dict()
