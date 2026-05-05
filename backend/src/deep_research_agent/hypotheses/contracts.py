from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HypothesisType(str, Enum):
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    TECHNICAL = "technical"
    RISK = "risk"
    RECOMMENDATION = "recommendation"
    MARKET = "market"
    LEGAL_POLICY = "legal_policy"
    UNKNOWN = "unknown"


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"


class ConfidenceLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class HypothesisEvidence(BaseModel):
    evidence_id: str
    hypothesis_id: str
    source_id: str | None = None
    claim_id: str | None = None
    artifact_path: str | None = None
    url: str | None = None
    title: str | None = None
    stance: str = Field(default="support", pattern="^(support|oppose|neutral)$")
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_text: str = ""
    overlap_terms: list[str] = Field(default_factory=list)
    matched_entities: list[str] = Field(default_factory=list)
    matched_values: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    source_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_readiness: float | None = Field(default=None, ge=0.0, le=1.0)
    primary_source: bool = False
    freshness_status: str | None = None


class ResearchHypothesis(BaseModel):
    hypothesis_id: str
    text: str = Field(..., min_length=8)
    normalized_text: str
    hypothesis_type: HypothesisType = HypothesisType.UNKNOWN
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    origin: str = "generated"
    origin_refs: list[str] = Field(default_factory=list)
    subquestion_ids: list[str] = Field(default_factory=list)
    competing_hypothesis_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    contradiction_ids: list[str] = Field(default_factory=list)
    confidence_update_id: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HypothesisTestResult(BaseModel):
    result_id: str
    hypothesis_id: str
    status: HypothesisStatus
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    opposition_score: float = Field(default=0.0, ge=0.0, le=1.0)
    net_evidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_evidence: list[HypothesisEvidence] = Field(default_factory=list)
    opposing_evidence: list[HypothesisEvidence] = Field(default_factory=list)
    neutral_evidence: list[HypothesisEvidence] = Field(default_factory=list)
    supporting_source_ids: list[str] = Field(default_factory=list)
    opposing_source_ids: list[str] = Field(default_factory=list)
    source_diversity: int = 0
    primary_source_count: int = 0
    citation_ready_count: int = 0
    contradiction_ids: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HypothesisConfidenceUpdate(BaseModel):
    update_id: str
    hypothesis_id: str
    prior_score: float = Field(default=0.35, ge=0.0, le=1.0)
    posterior_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.VERY_LOW
    factors: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    domain_sensitivity: str = "normal"
    needs_human_review: bool = True


class HypothesisContradiction(BaseModel):
    contradiction_id: str
    hypothesis_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    contradiction_type: str = "possible_conflict"
    severity: str = Field(default="medium", pattern="^(low|medium|high)$")
    explanation: str
    values: list[str] = Field(default_factory=list)


class HypothesisGraphNode(BaseModel):
    node_id: str
    node_type: str
    label: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HypothesisGraphEdge(BaseModel):
    source: str
    target: str
    relationship: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HypothesisGraph(BaseModel):
    graph_id: str
    thread_id: str
    generated_at: str
    nodes: list[HypothesisGraphNode] = Field(default_factory=list)
    edges: list[HypothesisGraphEdge] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HypothesisSummary(BaseModel):
    thread_id: str
    generated_at: str
    total_hypotheses: int = 0
    supported: int = 0
    partially_supported: int = 0
    contradicted: int = 0
    unsupported: int = 0
    inconclusive: int = 0
    needs_more_evidence: int = 0
    average_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    strongest_hypothesis_ids: list[str] = Field(default_factory=list)
    weakest_hypothesis_ids: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HypothesisSet(BaseModel):
    hypothesis_set_id: str
    thread_id: str
    question: str
    generated_at: str
    hypotheses: list[ResearchHypothesis] = Field(default_factory=list)
    test_results: list[HypothesisTestResult] = Field(default_factory=list)
    confidence_updates: list[HypothesisConfidenceUpdate] = Field(default_factory=list)
    contradictions: list[HypothesisContradiction] = Field(default_factory=list)
    graph: HypothesisGraph | None = None
    summary: HypothesisSummary
    metadata: dict[str, Any] = Field(default_factory=dict)


def model_to_plain(value: Any) -> Any:
    if isinstance(value, list):
        return [model_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: model_to_plain(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value
