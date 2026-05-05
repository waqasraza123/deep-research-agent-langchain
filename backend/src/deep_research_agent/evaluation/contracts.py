from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "low", "medium", "high", "critical"]
EvaluationConfidence = Literal["low", "medium", "high"]
GapKind = Literal[
    "unanswered_subquestion",
    "missing_question_entity",
    "unused_fetched_url",
    "source_used_not_cited",
    "unsupported_claim",
    "missing_primary_source",
    "missing_opposing_view",
    "missing_freshness_verification",
]
RiskKind = Literal[
    "unsupported_number",
    "unsupported_date",
    "weakly_supported_strong_claim",
    "introduced_entity",
    "unsupported_recommendation",
    "ignored_contradiction",
    "absolute_wording",
]


class EvaluationCriterion(BaseModel):
    key: str
    name: str
    description: str
    weight: float = Field(default=1.0, ge=0.0)
    minimum_score: float = Field(default=0.0, ge=0.0, le=1.0)


class EvaluationRubric(BaseModel):
    name: str = "deterministic_research_quality_v1"
    version: str = "1.0"
    criteria: list[EvaluationCriterion] = Field(default_factory=list)
    scoring_method: str = "deterministic_offline_heuristics"


class CriterionScore(BaseModel):
    criterion_key: str
    score: float = Field(ge=0.0, le=1.0)
    severity: Severity
    reasons: list[str] = Field(default_factory=list)
    suggested_fix: str = ""
    affected_artifacts: list[str] = Field(default_factory=list)


class CoverageGap(BaseModel):
    gap_id: str
    kind: GapKind
    severity: Severity
    description: str
    evidence: list[str] = Field(default_factory=list)
    suggested_fix: str = ""
    affected_artifacts: list[str] = Field(default_factory=list)


class HallucinationRiskFinding(BaseModel):
    finding_id: str
    kind: RiskKind
    severity: Severity
    text: str
    reason: str
    suggested_fix: str = ""
    affected_artifacts: list[str] = Field(default_factory=list)


class HallucinationRisk(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)
    severity: Severity
    findings: list[HallucinationRiskFinding] = Field(default_factory=list)
    checked_values: dict[str, list[str]] = Field(default_factory=dict)
    confidence: EvaluationConfidence = "low"


class BalanceAssessment(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    severity: Severity
    is_comparative_or_controversial: bool = False
    includes_both_sides: bool = False
    includes_tradeoffs: bool = False
    includes_counterarguments: bool = False
    includes_limitations: bool = False
    source_diversity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    vendor_bias_warning: str | None = None
    reasons: list[str] = Field(default_factory=list)


class FreshnessAssessment(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    severity: Severity
    is_time_sensitive: bool = False
    source_dates_present: bool = False
    stale_sources: list[str] = Field(default_factory=list)
    states_research_date: bool = False
    latest_wording_supported: bool = False
    primary_sources_needed: bool = False
    reasons: list[str] = Field(default_factory=list)


class CitationQualityAssessment(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    severity: Severity
    cited_source_count: int = 0
    total_source_count: int = 0
    unsupported_claim_count: int = 0
    weak_claim_count: int = 0
    reasons: list[str] = Field(default_factory=list)


class EvaluationRecommendation(BaseModel):
    priority: Severity
    title: str
    rationale: str
    suggested_action: str
    affected_artifacts: list[str] = Field(default_factory=list)


class ResearchEvaluation(BaseModel):
    evaluation_id: str
    thread_id: str
    generated_at: str
    question: str = ""
    overall_score: float = Field(ge=0.0, le=1.0)
    confidence: EvaluationConfidence = "low"
    rubric: EvaluationRubric
    criterion_scores: list[CriterionScore] = Field(default_factory=list)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)
    hallucination_risk: HallucinationRisk
    balance: BalanceAssessment
    freshness: FreshnessAssessment
    citation_quality: CitationQualityAssessment
    recommendations: list[EvaluationRecommendation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifacts_used: list[str] = Field(default_factory=list)
    method: str = "deterministic_offline_heuristics"


class BenchmarkCase(BaseModel):
    case_id: str
    question: str
    urls: list[str] = Field(default_factory=list)
    mocked_source_documents: list[dict[str, Any]] = Field(default_factory=list)
    report: str = ""
    notes: str = ""
    expected_artifacts: list[str] = Field(default_factory=list)
    expected_entities: list[str] = Field(default_factory=list)
    expected_min_scores: dict[str, float] = Field(default_factory=dict)
    known_traps: list[str] = Field(default_factory=list)
    required_warnings: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    case_id: str
    passed: bool
    overall_score: float = Field(ge=0.0, le=1.0)
    criterion_scores: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class RegressionSuiteResult(BaseModel):
    generated_at: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    results: list[BenchmarkResult] = Field(default_factory=list)
    artifacts_dir: str | None = None


def model_to_plain(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
