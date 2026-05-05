from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.source_identity import ChunkIdentity, DocumentIdentity

RetrievalQueryType = Literal[
    "main",
    "subquestion",
    "entity",
    "comparison",
    "risk",
    "freshness",
    "citation_verification",
]

ContextPackType = Literal[
    "agent_context_pack",
    "evidence_context_pack",
    "synthesis_context_pack",
    "verification_context_pack",
]


class RetrievalDocument(BaseModel):
    document_id: str
    document_identity: DocumentIdentity | None = None
    source_id: str
    url: str
    final_url: str | None = None
    title: str | None = None
    domain: str | None = None
    local_path: str | None = None
    content_hash: str
    text_hash: str
    word_count: int = 0
    char_count: int = 0
    source_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_readiness_score: float | None = Field(default=None, ge=0.0, le=1.0)
    primary_source_likelihood: float | None = Field(default=None, ge=0.0, le=1.0)
    source_role: str | None = None
    freshness_status: str | None = None
    fetched_at: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalChunk(BaseModel):
    chunk_id: str
    chunk_identity: ChunkIdentity | None = None
    source_id: str
    url: str
    title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    start_offset: int
    end_offset: int
    content_hash: str
    entities: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    source_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_readiness_score: float | None = Field(default=None, ge=0.0, le=1.0)
    token_count: int = 0
    document_id: str | None = None
    domain: str | None = None
    source_role: str | None = None
    freshness_status: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalIndex(BaseModel):
    index_id: str
    thread_id: str | None = None
    generated_at: str
    documents: list[RetrievalDocument] = Field(default_factory=list)
    chunks: list[RetrievalChunk] = Field(default_factory=list)
    chunk_count: int = 0
    token_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    query_id: str
    text: str
    query_type: RetrievalQueryType = "main"
    source: str = "question"
    entities: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    phrases: list[str] = Field(default_factory=list)
    freshness_required: bool = False
    primary_source_preferred: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalScore(BaseModel):
    total_score: float = 0.0
    lexical_score: float = 0.0
    phrase_score: float = 0.0
    entity_overlap_score: float = 0.0
    numeric_date_overlap_score: float = 0.0
    heading_score: float = 0.0
    source_quality_score: float = 0.0
    citation_readiness_score: float = 0.0
    freshness_score: float = 0.0
    embedding_score: float | None = None
    diversity_penalty: float = 0.0
    duplication_penalty: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    matched_entities: list[str] = Field(default_factory=list)
    matched_values: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    query_id: str
    rank: int
    chunk: RetrievalChunk
    score: RetrievalScore


class HybridRankingConfig(BaseModel):
    top_k: int = Field(default=12, ge=1, le=100)
    candidate_k: int = Field(default=80, ge=1, le=500)
    lexical_weight: float = 1.0
    phrase_weight: float = 0.25
    entity_weight: float = 0.2
    numeric_date_weight: float = 0.2
    heading_weight: float = 0.12
    source_quality_weight: float = 0.12
    citation_readiness_weight: float = 0.12
    freshness_weight: float = 0.12
    embedding_weight: float = 0.0
    diversity_penalty_weight: float = 0.08
    duplication_penalty_weight: float = 0.15
    max_chunks_per_source: int = Field(default=3, ge=1, le=20)
    min_score: float = 0.0
    use_embeddings: bool = False


class ContextPackItem(BaseModel):
    chunk_id: str
    source_id: str
    url: str
    title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    score: float
    relevance_reason: str
    citation_hint: str
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalCoverageSummary(BaseModel):
    query_count: int = 0
    result_count: int = 0
    selected_chunk_count: int = 0
    source_count: int = 0
    covered_sources: list[str] = Field(default_factory=list)
    missing_queries: list[str] = Field(default_factory=list)
    covered_entities: list[str] = Field(default_factory=list)
    missing_entities: list[str] = Field(default_factory=list)
    covered_dates: list[str] = Field(default_factory=list)
    covered_numbers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ContextPack(BaseModel):
    pack_id: str
    pack_type: ContextPackType
    question: str
    generated_at: str
    items: list[ContextPackItem] = Field(default_factory=list)
    coverage_summary: RetrievalCoverageSummary = Field(default_factory=RetrievalCoverageSummary)
    warnings: list[str] = Field(default_factory=list)
    max_chars: int = 12000
    total_chars: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPackBuildResult(BaseModel):
    thread_id: str | None = None
    question: str
    generated_at: str
    index: RetrievalIndex
    queries: list[RetrievalQuery] = Field(default_factory=list)
    results: list[RetrievalResult] = Field(default_factory=list)
    packs: dict[str, ContextPack] = Field(default_factory=dict)
    coverage_summary: RetrievalCoverageSummary = Field(default_factory=RetrievalCoverageSummary)
    warnings: list[str] = Field(default_factory=list)


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
