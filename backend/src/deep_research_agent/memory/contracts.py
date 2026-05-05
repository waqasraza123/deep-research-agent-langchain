from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from deep_research_agent.source_identity import SourceIdentity


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    FRAMEWORK_LIBRARY = "framework_library"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    PERCENTAGE = "percentage"
    NUMERIC_VALUE = "numeric_value"
    TECHNICAL_TERM = "technical_term"
    LEGAL_POLICY_TERM = "legal_policy_term"


class ExtractedEntity(BaseModel):
    name: str
    entity_type: EntityType
    confidence: float = Field(ge=0.0, le=1.0)
    mentions: int = Field(default=1, ge=1)
    evidence: list[str] = Field(default_factory=list)


class ExtractedTopic(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    topics: list[ExtractedTopic] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ArtifactReference(BaseModel):
    thread_id: str
    path: str
    artifact_type: str = "artifact"
    size_bytes: int | None = None


class MemoryRecord(BaseModel):
    memory_id: str
    thread_id: str
    question: str
    normalized_question: str
    source_url: str
    source_id: str | None = None
    source_identity: SourceIdentity | None = None
    normalized_url: str
    canonical_url: str | None = None
    source_title: str | None = None
    source_domain: str | None = None
    content_hash: str
    extracted_text_hash: str
    source_type: str | None = None
    first_seen_at: str
    last_seen_at: str
    run_count: int = Field(default=1, ge=1)
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    topics: list[ExtractedTopic] = Field(default_factory=list)
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactReference] = Field(default_factory=list)


class SourceReuseDecision(BaseModel):
    reuse_allowed: bool
    reuse_reason: str
    freshness_warning: str | None = None
    previous_thread_ids: list[str] = Field(default_factory=list)
    previous_artifacts: list[ArtifactReference] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    matched_memory_ids: list[str] = Field(default_factory=list)


class MemoryContext(BaseModel):
    question: str
    normalized_question: str
    similar_previous_questions: list[MemoryRecord] = Field(default_factory=list)
    previously_useful_sources: list[MemoryRecord] = Field(default_factory=list)
    known_entities: list[ExtractedEntity] = Field(default_factory=list)
    known_topics: list[ExtractedTopic] = Field(default_factory=list)
    stale_warnings: list[str] = Field(default_factory=list)
    suggested_source_reuse_candidates: list[SourceReuseDecision] = Field(default_factory=list)
    prior_artifact_links: list[ArtifactReference] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class MemoryGraphNode(BaseModel):
    id: str
    type: str
    label: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryGraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float = Field(default=1.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryGraph(BaseModel):
    generated_at: str
    nodes: list[MemoryGraphNode] = Field(default_factory=list)
    edges: list[MemoryGraphEdge] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
