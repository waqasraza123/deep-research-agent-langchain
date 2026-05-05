from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal[
    "official_docs",
    "primary_source",
    "source_code",
    "release_notes",
    "standards_or_specification",
    "academic_paper",
    "systematic_review",
    "legal_text",
    "regulator_guidance",
    "court_or_agency_record",
    "market_data",
    "financial_filing",
    "company_disclosure",
    "vendor_documentation",
    "benchmark",
    "news",
    "expert_analysis",
    "community_discussion",
    "medical_guideline",
    "clinical_source",
    "implementation_example",
]

VerificationStrictness = Literal["low", "standard", "high", "very_high"]
CitationStrictness = Literal["standard", "strict", "primary_source_required"]
FreshnessStrictness = Literal["not_critical", "prefer_recent", "current_required"]
ReportLengthPreference = Literal["brief", "standard", "long"]
SynthesisProfile = Literal[
    "concise_answer",
    "deep_research_report",
    "technical_due_diligence",
    "comparative_report",
    "decision_memo",
    "risk_review",
    "literature_style_review",
    "implementation_plan",
]
WarningSeverity = Literal["info", "low", "medium", "high", "critical"]


def model_to_plain(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()


class ProtocolRule(BaseModel):
    rule_id: str = Field(..., min_length=2)
    description: str = Field(..., min_length=5)
    severity: WarningSeverity = "medium"
    applies_when: list[str] = Field(default_factory=list)
    required_action: str = Field(..., min_length=5)


class SourceRequirement(BaseModel):
    source_type: SourceType
    minimum_count: int = Field(default=1, ge=0, le=20)
    required: bool = True
    rationale: str = Field(..., min_length=5)
    freshness_days: int | None = Field(default=None, ge=1, le=3650)
    examples: list[str] = Field(default_factory=list)


class VerificationRequirement(BaseModel):
    strictness: VerificationStrictness = "standard"
    minimum_independent_sources: int = Field(default=1, ge=0, le=10)
    require_primary_source_for_decisive_claims: bool = False
    require_contradiction_scan: bool = True
    require_uncertainty_boundaries: bool = True
    notes: list[str] = Field(default_factory=list)


class CitationPolicy(BaseModel):
    strictness: CitationStrictness = "standard"
    require_inline_citations: bool = True
    require_source_ids: bool = True
    require_claim_level_citations: bool = False
    allow_uncited_background: bool = False
    primary_source_required_for: list[str] = Field(default_factory=list)
    disallowed_citation_sources: list[SourceType] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class FreshnessPolicy(BaseModel):
    strictness: FreshnessStrictness = "prefer_recent"
    max_age_days: int | None = Field(default=None, ge=1, le=3650)
    require_publication_dates: bool = False
    require_retrieval_date: bool = True
    stale_source_handling: str = "Flag stale or undated sources before relying on them."
    notes: list[str] = Field(default_factory=list)


class SynthesisPolicy(BaseModel):
    profile: SynthesisProfile = "deep_research_report"
    required_sections: list[str] = Field(default_factory=list)
    include_tradeoffs: bool = True
    include_uncertainties: bool = True
    include_next_steps: bool = True
    forbidden_overclaims: list[str] = Field(default_factory=list)
    required_uncertainty_language: list[str] = Field(default_factory=list)


class EvaluationPolicy(BaseModel):
    required: bool = True
    weights: dict[str, float] = Field(default_factory=dict)
    minimum_overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    fail_on_unsupported_decisive_claims: bool = False
    notes: list[str] = Field(default_factory=list)


class SafetyPolicy(BaseModel):
    conservative_language: bool = False
    professional_advice_disclaimer: bool = False
    require_human_review: bool = False
    safety_warnings: list[str] = Field(default_factory=list)
    operator_review_required_when: list[str] = Field(default_factory=list)


class ProtocolWarning(BaseModel):
    code: str = Field(..., min_length=2)
    message: str = Field(..., min_length=5)
    severity: WarningSeverity = "medium"
    review_recommended: bool = False


class ResearchProtocol(BaseModel):
    protocol_id: str = Field(..., min_length=2)
    name: str = Field(..., min_length=3)
    description: str = Field(..., min_length=10)
    matching_signals: list[str] = Field(default_factory=list)
    required_source_types: list[SourceRequirement] = Field(default_factory=list)
    preferred_source_types: list[SourceType] = Field(default_factory=list)
    disallowed_or_low_value_source_types: list[SourceType] = Field(default_factory=list)
    verification_strictness: VerificationRequirement = Field(default_factory=VerificationRequirement)
    citation_requirements: CitationPolicy = Field(default_factory=CitationPolicy)
    freshness_requirements: FreshnessPolicy = Field(default_factory=FreshnessPolicy)
    required_artifacts: list[str] = Field(default_factory=list)
    synthesis_profile: SynthesisPolicy = Field(default_factory=SynthesisPolicy)
    evaluation_weights: EvaluationPolicy = Field(default_factory=EvaluationPolicy)
    safety_warnings: SafetyPolicy = Field(default_factory=SafetyPolicy)
    operator_review_required_when: list[str] = Field(default_factory=list)
    rules: list[ProtocolRule] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntelligenceProfile(BaseModel):
    profile_id: str = Field(..., min_length=2)
    name: str = Field(..., min_length=3)
    description: str = Field(..., min_length=8)
    max_sources: int = Field(default=3, ge=0, le=50)
    max_chunks: int = Field(default=20, ge=0, le=500)
    follow_links_default: bool = False
    max_links_per_source_default: int = Field(default=0, ge=0, le=10)
    verification_strictness: VerificationStrictness = "standard"
    source_discovery_enabled: bool = True
    synthesis_enabled: bool = True
    evaluation_required: bool = True
    review_gate_recommended: bool = False
    citation_strictness: CitationStrictness = "standard"
    report_length_preference: ReportLengthPreference = "standard"
    notes: list[str] = Field(default_factory=list)


class PolicyPack(BaseModel):
    pack_id: str = Field(..., min_length=2)
    name: str = Field(..., min_length=3)
    description: str = Field(..., min_length=8)
    version: str = "1.0"
    applies_to_protocols: list[str] = Field(default_factory=list)
    source_requirements: list[SourceRequirement] = Field(default_factory=list)
    rules: list[ProtocolRule] = Field(default_factory=list)
    warnings: list[ProtocolWarning] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProtocolSelection(BaseModel):
    selected_protocol: ResearchProtocol
    intelligence_profile: IntelligenceProfile
    confidence_score: float = Field(ge=0.0, le=1.0)
    alternative_protocols: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[ProtocolWarning] = Field(default_factory=list)
    policy_packs: list[PolicyPack] = Field(default_factory=list)
    effective_source_requirements: list[SourceRequirement] = Field(default_factory=list)
    effective_verification: VerificationRequirement
    effective_citation_policy: CitationPolicy
    effective_freshness_policy: FreshnessPolicy
    effective_synthesis_policy: SynthesisPolicy
    effective_evaluation_policy: EvaluationPolicy
    review_recommended: bool = False
    instruction_block: str = ""

