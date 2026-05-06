from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

WarningSeverity = Literal["info", "low", "medium", "high", "critical"]
IntentLabel = Literal[
    "general_research",
    "factual_answer",
    "comparative_analysis",
    "technical_due_diligence",
    "implementation_planning",
    "library_or_framework_review",
    "source_code_research",
    "legal_policy_review",
    "market_research",
    "vendor_evaluation",
    "financial_risk_review",
    "medical_health_review",
    "academic_literature_review",
    "news_or_current_review",
    "risk_assessment",
    "unknown",
]
ComplexityLevel = Literal[
    "simple",
    "moderate",
    "deep",
    "multi_domain",
    "adversarial",
    "sensitive",
    "unknown",
]
PassType = Literal[
    "request_analysis",
    "blueprint_generation",
    "source_inventory",
    "source_unitization",
    "evidence_unitization",
    "agent_execution",
    "report_critique",
    "claim_verification",
    "confidence_calibration",
    "final_kernel_summary",
]
PassStatus = Literal["pending", "running", "completed", "skipped", "failed"]
SourceRole = Literal[
    "primary_evidence",
    "secondary_context",
    "background",
    "weak_reference",
    "risky_source",
    "duplicate",
    "unknown",
]
EvidenceType = Literal[
    "definition",
    "factual_statement",
    "numeric_metric",
    "date_or_version",
    "comparison",
    "causal_claim",
    "recommendation_support",
    "risk_signal",
    "quote",
    "table_row",
    "policy_statement",
    "code_or_api_reference",
    "unknown",
]
ClaimType = Literal[
    "factual",
    "comparative",
    "numeric",
    "temporal",
    "causal",
    "recommendation",
    "risk",
    "legal_policy",
    "medical_health",
    "financial",
    "technical",
    "unknown",
]
VerificationStatus = Literal[
    "pending",
    "verified",
    "partially_verified",
    "contradicted",
    "unsupported",
    "not_enough_information",
    "skipped",
    "failed",
]
CritiqueCategory = Literal[
    "unsupported_claim",
    "weak_citation",
    "stale_source",
    "source_quality",
    "overclaiming",
    "missing_counterargument",
    "missing_primary_source",
    "contradiction",
    "numeric_mismatch",
    "temporal_mismatch",
    "unsafe_source_content",
    "incomplete_answer",
    "unclear_uncertainty",
    "artifact_missing",
]
ConfidenceLevel = Literal["very_low", "low", "medium", "high", "very_high"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    payload = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def model_to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value.dict()
    if isinstance(value, list):
        return [model_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(k): model_to_plain(v) for k, v in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(model_to_plain(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, round(float(value), 4)))


def tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}", text or "")
        if token.lower()
        not in {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "are",
            "was",
            "were",
            "have",
            "has",
            "into",
            "not",
            "but",
        }
    }


class KernelWarning(BaseModel):
    warning_id: str
    subsystem: str
    code: str
    severity: WarningSeverity = "medium"
    message: str
    affected_artifacts: list[str] = Field(default_factory=list)
    affected_sources: list[str] = Field(default_factory=list)
    affected_claims: list[str] = Field(default_factory=list)
    recommended_action: str = ""


class ResearchKernelSettings(BaseModel):
    intelligence_kernel_enabled: bool = True
    offline_mode: bool = True
    mock_model_allowed: bool = True
    source_reasoning_enabled: bool = True
    critique_enabled: bool = True
    verification_enabled: bool = True
    confidence_enabled: bool = True
    max_reasoning_passes: int = Field(default=8, ge=1, le=50)
    max_source_units: int = Field(default=100, ge=1, le=1000)
    max_evidence_units: int = Field(default=500, ge=1, le=5000)
    max_claims: int = Field(default=200, ge=1, le=2000)
    max_claims_to_verify: int = Field(default=50, ge=0, le=500)
    max_artifact_bytes: int = Field(default=5_000_000, ge=1_000)
    strict_citation_mode: bool = False
    sensitive_domain_review_required: bool = True
    fail_on_critical_warnings: bool = False
    produce_markdown_artifacts: bool = True
    produce_json_artifacts: bool = True


class ResearchKernelInput(BaseModel):
    thread_id: str
    question: str
    urls: list[str] = Field(default_factory=list)
    settings_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    requested_outputs: list[str] = Field(default_factory=list)
    user_constraints: list[str] = Field(default_factory=list)
    raw_request_metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchIntent(BaseModel):
    intent_id: str
    label: IntentLabel
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    signals: dict[str, list[str]] = Field(default_factory=dict)


class ResearchComplexity(BaseModel):
    level: ComplexityLevel
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    detected_dimensions: list[str] = Field(default_factory=list)


class ResearchBlueprint(BaseModel):
    blueprint_id: str
    thread_id: str
    question: str
    normalized_question: str
    intent: ResearchIntent
    complexity: ResearchComplexity
    required_passes: list[PassType] = Field(default_factory=list)
    optional_passes: list[PassType] = Field(default_factory=list)
    skipped_passes: dict[str, str] = Field(default_factory=dict)
    source_requirements: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    verification_requirements: list[str] = Field(default_factory=list)
    citation_policy: dict[str, Any] = Field(default_factory=dict)
    freshness_policy: dict[str, Any] = Field(default_factory=dict)
    safety_policy: dict[str, Any] = Field(default_factory=dict)
    synthesis_policy: dict[str, Any] = Field(default_factory=dict)
    expected_artifacts: list[str] = Field(default_factory=list)
    operator_warnings: list[KernelWarning] = Field(default_factory=list)
    generated_at: str = Field(default_factory=utc_now)


class ResearchPass(BaseModel):
    pass_id: str
    pass_type: PassType
    name: str
    status: PassStatus = "pending"
    required: bool = True
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    skipped_reason: str | None = None
    failure_reason: str | None = None
    warnings: list[KernelWarning] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class SourceUnit(BaseModel):
    source_unit_id: str
    source_id: str
    url: str
    canonical_url: str | None = None
    normalized_url: str
    domain: str
    title: str | None = None
    source_type: str = "unknown"
    source_kind: str = "unknown"
    parent_url: str | None = None
    fetched_at: str | None = None
    content_hash: str | None = None
    extraction_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    trust_level: Literal["high", "medium", "low", "unknown", "risky"] = "unknown"
    source_role: SourceRole = "unknown"
    source_warnings: list[KernelWarning] = Field(default_factory=list)
    useful_for: list[str] = Field(default_factory=list)
    unsafe_for: list[str] = Field(default_factory=list)
    text_preview: str = ""


class EvidenceUnit(BaseModel):
    evidence_unit_id: str
    source_unit_id: str | None = None
    source_id: str | None = None
    url: str | None = None
    title: str | None = None
    text: str
    normalized_text: str
    section_hint: str | None = None
    evidence_type: EvidenceType = "unknown"
    entities: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_ready: bool = False
    warnings: list[KernelWarning] = Field(default_factory=list)


class ResearchClaim(BaseModel):
    claim_id: str
    text: str
    normalized_text: str
    claim_type: ClaimType = "unknown"
    source_artifact: str = "report.md"
    strength: Literal["weak", "normal", "strong", "absolute"] = "normal"
    entities: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    opposing_evidence_ids: list[str] = Field(default_factory=list)
    verification_status: VerificationStatus = "pending"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[KernelWarning] = Field(default_factory=list)


class VerificationTask(BaseModel):
    task_id: str
    claim_id: str
    task_type: str
    priority: int = Field(default=3, ge=1, le=5)
    reason: str
    expected_evidence: list[str] = Field(default_factory=list)
    candidate_evidence_ids: list[str] = Field(default_factory=list)
    status: VerificationStatus = "pending"
    result: str = ""
    confidence_before: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_after: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[KernelWarning] = Field(default_factory=list)


class CritiqueFinding(BaseModel):
    finding_id: str
    severity: WarningSeverity = "medium"
    category: CritiqueCategory
    message: str
    affected_artifacts: list[str] = Field(default_factory=list)
    affected_claims: list[str] = Field(default_factory=list)
    affected_sources: list[str] = Field(default_factory=list)
    recommended_action: str = ""


class ConfidenceCalibration(BaseModel):
    target_type: str
    target_id: str
    confidence_before: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_after: float = Field(default=0.5, ge=0.0, le=1.0)
    level: ConfidenceLevel = "medium"
    reasons: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    boosts: list[str] = Field(default_factory=list)
    warnings: list[KernelWarning] = Field(default_factory=list)


class KernelRunSummary(BaseModel):
    thread_id: str
    question: str
    intent: ResearchIntent
    complexity: ResearchComplexity
    passes_executed: list[str] = Field(default_factory=list)
    passes_skipped: list[str] = Field(default_factory=list)
    critical_warnings: list[KernelWarning] = Field(default_factory=list)
    high_warnings: list[KernelWarning] = Field(default_factory=list)
    source_count: int = 0
    evidence_unit_count: int = 0
    claim_count: int = 0
    verified_claim_count: int = 0
    unsupported_claim_count: int = 0
    contradicted_claim_count: int = 0
    final_confidence: ConfidenceCalibration
    generated_artifacts: list[str] = Field(default_factory=list)
    operator_next_actions: list[str] = Field(default_factory=list)


class KernelArtifactMetadata(BaseModel):
    name: str
    path: str
    type: str
    size_bytes: int = 0
    content_hash: str | None = None
    created_or_detected_at: str = Field(default_factory=utc_now)
    producer: str = "intelligence_kernel"
    required: bool = False
    exists: bool = False
    warnings: list[KernelWarning] = Field(default_factory=list)
