from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal[
    "official_docs",
    "source_code_repository",
    "release_notes",
    "academic_paper",
    "government_or_policy",
    "legal_or_regulatory",
    "company_announcement",
    "benchmark_report",
    "tutorial_or_blog",
    "forum_discussion",
    "dataset",
    "unknown",
]

QueryIntent = Literal[
    "broad_overview",
    "primary_source",
    "official_documentation",
    "recent_current",
    "comparison",
    "risk_failure_mode",
    "benchmark_evaluation",
    "regulatory_legal",
    "academic_literature",
    "implementation",
]

ProviderType = Literal["mock", "static", "disabled"]
CandidateDecision = Literal["undecided", "selected", "rejected", "duplicate", "skipped"]


class SearchQuery(BaseModel):
    query_id: str
    text: str = Field(..., min_length=3)
    intent: QueryIntent
    target_source_types: list[SourceType] = Field(default_factory=list)
    freshness_required: bool = False
    rationale: str = ""


class SearchQueryPlan(BaseModel):
    question: str
    queries: list[SearchQuery] = Field(default_factory=list)
    freshness_required: bool = False
    comparative: bool = False
    technical: bool = False
    legal_or_policy: bool = False
    academic: bool = False
    entities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SourceDiscoverySettings(BaseModel):
    discovery_enabled: bool = False
    max_queries: int = Field(default=8, ge=0, le=20)
    max_candidates_per_query: int = Field(default=5, ge=0, le=20)
    max_selected_sources: int = Field(default=3, ge=0, le=20)
    require_primary_source_when_possible: bool = True
    allow_secondary_sources: bool = True
    allow_forums: bool = False
    freshness_required: bool | None = None
    provider: ProviderType = "disabled"
    static_results: list[dict[str, Any]] = Field(default_factory=list)


class SearchProviderConfig(BaseModel):
    provider: ProviderType = "disabled"
    enabled: bool = False
    name: str | None = None
    reason: str | None = None
    static_results: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceAcquisitionPlan(BaseModel):
    question: str
    required_source_types: list[SourceType] = Field(default_factory=list)
    preferred_source_types: list[SourceType] = Field(default_factory=list)
    optional_source_types: list[SourceType] = Field(default_factory=list)
    query_plan: SearchQueryPlan
    provider_config: SearchProviderConfig
    user_urls: list[str] = Field(default_factory=list)
    settings: SourceDiscoverySettings = Field(default_factory=SourceDiscoverySettings)
    rationale: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SourceCandidateScore(BaseModel):
    candidate_id: str
    total_score: float = Field(default=0.0, ge=0.0, le=1.0)
    components: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SourceCandidate(BaseModel):
    candidate_id: str
    url: str
    title: str = ""
    snippet: str = ""
    domain: str = ""
    provider: str = ""
    query: str = ""
    query_id: str | None = None
    query_intent: QueryIntent | None = None
    source_type_hint: SourceType = "unknown"
    primary_source_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_hint: str = "unknown"
    authority_hint: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_group_id: str | None = None
    ranking_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ranking_reasons: list[str] = Field(default_factory=list)
    decision: CandidateDecision = "undecided"
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchProviderResult(BaseModel):
    provider: str
    query_id: str
    query: str
    ok: bool = True
    disabled_reason: str | None = None
    candidates: list[SourceCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceAcquisitionDecision(BaseModel):
    candidate_id: str
    url: str
    title: str = ""
    domain: str = ""
    source_type_hint: SourceType = "unknown"
    decision: CandidateDecision
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    rank: int | None = None
    reason: str = ""
    duplicate_group_id: str | None = None
    warnings: list[str] = Field(default_factory=list)


class SourceDiscoverySummary(BaseModel):
    discovery_enabled: bool = False
    provider: str = "disabled"
    question: str
    query_count: int = 0
    provider_result_count: int = 0
    candidate_count: int = 0
    selected_count: int = 0
    selected_urls: list[str] = Field(default_factory=list)
    skipped_reason: str | None = None
    required_source_types: list[SourceType] = Field(default_factory=list)
    coverage_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SourceDiscoveryBatch(BaseModel):
    request: "SourceDiscoveryRequest"
    plan: SourceAcquisitionPlan
    provider_results: list[SearchProviderResult] = Field(default_factory=list)
    candidates: list[SourceCandidate] = Field(default_factory=list)
    decisions: list[SourceAcquisitionDecision] = Field(default_factory=list)
    selected_candidates: list[SourceCandidate] = Field(default_factory=list)
    summary: SourceDiscoverySummary


class SourceDiscoveryRequest(BaseModel):
    question: str = Field(..., min_length=5)
    user_urls: list[str] = Field(default_factory=list)
    thread_id: str | None = None
    settings: SourceDiscoverySettings = Field(default_factory=SourceDiscoverySettings)
    persist: bool = False


try:
    SourceDiscoveryBatch.update_forward_refs()
except Exception:
    pass


def model_to_plain(value: BaseModel) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")  # type: ignore[attr-defined]
    return value.dict()
