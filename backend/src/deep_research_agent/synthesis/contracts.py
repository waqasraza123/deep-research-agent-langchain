from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ReportProfile = Literal[
    "concise_answer",
    "deep_research_report",
    "technical_due_diligence",
    "comparative_report",
    "decision_memo",
    "risk_review",
    "literature_style_review",
]

FindingOrigin = Literal[
    "evidence_ledger",
    "notes",
    "report",
    "source_summary",
    "source_audit",
    "strategy",
]

ClaimType = Literal[
    "factual",
    "comparative",
    "numeric",
    "date_sensitive",
    "causal",
    "recommendation",
    "unsupported_broad",
    "assumption",
    "risk",
    "question",
]

ConfidenceLabel = Literal[
    "source_backed",
    "strong",
    "moderate",
    "weak",
    "unsupported",
    "contradicted",
    "unknown",
]

ContradictionStatus = Literal["none", "possible", "contradicted"]


class SynthesisInput(BaseModel):
    thread_id: str
    question: str = ""
    generated_at: str
    notes_text: str = ""
    report_text: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ledger: dict[str, Any] | None = None
    source_audit: dict[str, Any] | None = None
    source_audit_text: str = ""
    strategy: dict[str, Any] | None = None
    subquestions: list[dict[str, Any]] = Field(default_factory=list)
    available_artifacts: list[str] = Field(default_factory=list)


class ResearchFinding(BaseModel):
    finding_id: str
    text: str
    normalized_text: str
    origin: FindingOrigin
    origin_ref: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    subquestion_ids: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    claim_type: ClaimType = "factual"
    confidence_label: ConfidenceLabel = "unknown"
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    contradiction_status: ContradictionStatus = "none"
    contradiction_ids: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    requires_human_review: bool = True


class FindingCluster(BaseModel):
    cluster_id: str
    cluster_kind: Literal[
        "topic",
        "subquestion",
        "entity",
        "source",
        "claim_type",
        "confidence",
        "contradiction",
    ]
    label: str
    finding_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    representative_finding_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArgumentNode(BaseModel):
    node_id: str
    kind: Literal[
        "main_answer",
        "supporting_claim",
        "counterclaim",
        "assumption",
        "weak_evidence",
        "unresolved_question",
        "implication",
        "risk",
    ]
    text: str
    finding_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    confidence_label: ConfidenceLabel = "unknown"
    requires_human_review: bool = True


class ArgumentRelation(BaseModel):
    relation_id: str
    source_node_id: str
    target_node_id: str
    relation_type: Literal["supports", "challenges", "depends_on", "implies", "qualifies"]
    rationale: str = ""


class ArgumentMap(BaseModel):
    thread_id: str
    question: str
    generated_at: str
    main_answer: str
    nodes: list[ArgumentNode] = Field(default_factory=list)
    relations: list[ArgumentRelation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ComparisonDimension(BaseModel):
    dimension_id: str
    label: str
    rationale: str = ""
    finding_ids: list[str] = Field(default_factory=list)


class ComparisonCell(BaseModel):
    option: str
    dimension_id: str
    finding_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    confidence_label: ConfidenceLabel = "unknown"
    source_ids: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class ComparisonMatrix(BaseModel):
    thread_id: str
    question: str
    generated_at: str
    detected: bool = False
    options: list[str] = Field(default_factory=list)
    dimensions: list[ComparisonDimension] = Field(default_factory=list)
    cells: list[ComparisonCell] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    recommendation_id: str
    option: str | None = None
    stance: Literal["recommend", "conditional", "defer", "avoid"] = "defer"
    summary: str
    rationale: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    confidence_label: ConfidenceLabel = "unknown"
    conditions: list[str] = Field(default_factory=list)


class DecisionMemo(BaseModel):
    thread_id: str
    question: str
    generated_at: str
    detected: bool = False
    context: str = ""
    decision_to_make: str = ""
    options: list[str] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    rationale: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    reversibility: str = "Not assessed from available artifacts."
    cost_complexity: str = "Not assessed from available artifacts."
    confidence_label: ConfidenceLabel = "unknown"
    next_validation_steps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class OpenQuestion(BaseModel):
    question_id: str
    text: str
    reason: str = ""
    related_finding_ids: list[str] = Field(default_factory=list)
    requires_primary_source: bool = False
    requires_human_review: bool = True


class UncertaintyBoundary(BaseModel):
    thread_id: str
    question: str
    generated_at: str
    known: list[str] = Field(default_factory=list)
    likely: list[str] = Field(default_factory=list)
    uncertain: list[str] = Field(default_factory=list)
    not_verified: list[str] = Field(default_factory=list)
    freshness_dependent: list[str] = Field(default_factory=list)
    requires_human_review: list[str] = Field(default_factory=list)
    requires_primary_sources: list[str] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReportAssemblyPlan(BaseModel):
    thread_id: str
    generated_at: str
    profile: ReportProfile
    sections: list[str] = Field(default_factory=list)
    source_artifacts: list[str] = Field(default_factory=list)
    safe_to_replace_report: bool = False
    gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SynthesisOutput(BaseModel):
    thread_id: str
    generated_at: str
    findings: list[ResearchFinding] = Field(default_factory=list)
    clusters: list[FindingCluster] = Field(default_factory=list)
    argument_map: ArgumentMap
    comparison_matrix: ComparisonMatrix
    decision_memo: DecisionMemo
    uncertainty_boundaries: UncertaintyBoundary
    report_assembly_plan: ReportAssemblyPlan
    assembled_report_markdown: str = ""
    warnings: list[str] = Field(default_factory=list)
