from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from deep_research_agent.source_identity import ChunkIdentity, DocumentIdentity


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def model_to_plain(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


class DocumentExtractionWarning(BaseModel):
    code: str
    message: str
    severity: str = "warning"
    source_id: str | None = None
    location: str | None = None


class DocumentMetadata(BaseModel):
    author: str | None = None
    published_at: str | None = None
    modified_at: str | None = None
    description: str | None = None
    keywords: list[str] = Field(default_factory=list)
    html_headings: list[str] = Field(default_factory=list)
    extraction_strategy: str | None = None
    content_type: str | None = None
    status_code: int | None = None
    canonical_url: str | None = None
    local_path: str | None = None
    word_count: int = 0
    char_count: int = 0
    raw_char_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    section_id: str
    heading: str
    level: int
    start_offset: int
    end_offset: int
    text: str
    parent_section_id: str | None = None
    path: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0


class DocumentTableCell(BaseModel):
    row_index: int
    column_index: int
    text: str
    is_header: bool = False
    row_span: int = 1
    col_span: int = 1


class DocumentTable(BaseModel):
    table_id: str
    source_id: str
    section_id: str | None = None
    caption: str | None = None
    start_offset: int
    end_offset: int
    rows: list[list[str]] = Field(default_factory=list)
    cells: list[DocumentTableCell] = Field(default_factory=list)
    readable_text: str
    table_kind: str
    confidence_score: float
    warnings: list[str] = Field(default_factory=list)


class DocumentCitation(BaseModel):
    citation_id: str
    source_id: str
    citation_type: str
    text: str
    normalized_value: str | None = None
    start_offset: int
    end_offset: int
    context: str
    confidence_score: float


class DocumentFootnote(BaseModel):
    footnote_id: str
    source_id: str
    marker: str
    text: str
    start_offset: int
    end_offset: int
    context: str = ""
    confidence_score: float = 0.7


class DocumentContentFeature(BaseModel):
    name: str
    present: bool
    confidence_score: float
    evidence: list[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    chunk_id: str
    chunk_identity: ChunkIdentity | None = None
    source_id: str
    document_id: str | None = None
    section_id: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    text: str
    start_offset: int
    end_offset: int
    content_hash: str
    ordinal: int
    approx_tokens: int
    context_before: str = ""
    context_after: str = ""
    detected_entities: list[str] = Field(default_factory=list)
    detected_numbers: list[str] = Field(default_factory=list)
    detected_dates: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)


class DocumentNormalizationResult(BaseModel):
    source_id: str
    raw_text: str
    normalized_text: str
    source_type: str
    content_hash: str
    warnings: list[DocumentExtractionWarning] = Field(default_factory=list)
    transformations: list[str] = Field(default_factory=list)


class DocumentProfile(BaseModel):
    document_id: str | None = None
    document_identity: DocumentIdentity | None = None
    source_id: str
    url: str
    title: str | None = None
    domain: str | None = None
    source_type: str = "unknown"
    language_hint: str | None = None
    extracted_at: datetime = Field(default_factory=now_utc)
    content_hash: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    sections: list[DocumentSection] = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)
    tables: list[DocumentTable] = Field(default_factory=list)
    citations: list[DocumentCitation] = Field(default_factory=list)
    footnotes: list[DocumentFootnote] = Field(default_factory=list)
    features: list[DocumentContentFeature] = Field(default_factory=list)
    warnings: list[DocumentExtractionWarning] = Field(default_factory=list)
    quality_summary: dict[str, Any] = Field(default_factory=dict)


class DocumentIntelligenceBatch(BaseModel):
    thread_id: str | None = None
    generated_at: datetime = Field(default_factory=now_utc)
    profiles: list[DocumentProfile] = Field(default_factory=list)
    warnings: list[DocumentExtractionWarning] = Field(default_factory=list)
