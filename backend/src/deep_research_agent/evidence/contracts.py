from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ClaimType = Literal[
    "factual",
    "comparative",
    "numeric",
    "date_sensitive",
    "causal",
    "recommendation",
    "unsupported_broad",
]

SupportLevel = Literal["source_backed", "strong", "moderate", "weak", "unsupported", "contradicted"]

ContradictionSeverity = Literal["low", "medium", "high"]


class EvidenceSource(BaseModel):
    source_id: str
    url: str | None = None
    final_url: str | None = None
    title: str | None = None
    domain: str | None = None
    local_path: str | None = None
    ok: bool = True
    fetched_at: str | None = None
    word_count: int | None = None
    char_count: int | None = None
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceQuote(BaseModel):
    quote_id: str
    source_id: str
    text: str
    start_char: int | None = None
    end_char: int | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class ClaimCitation(BaseModel):
    source_id: str
    quote_id: str | None = None
    url: str | None = None
    title: str | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    matched_text: str = ""
    overlap_terms: list[str] = Field(default_factory=list)
    value_matches: list[str] = Field(default_factory=list)


class ClaimConfidence(BaseModel):
    claim_id: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    support_level: SupportLevel = "unsupported"
    factors: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    freshness_warning: str | None = None


class ExtractedClaim(BaseModel):
    claim_id: str
    text: str
    normalized_text: str
    claim_type: ClaimType = "factual"
    origin: Literal["notes", "report", "source"] = "report"
    origin_ref: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    citations: list[ClaimCitation] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    support_level: SupportLevel = "unsupported"
    contradiction_ids: list[str] = Field(default_factory=list)
    needs_human_review: bool = True
    notes: list[str] = Field(default_factory=list)


class ContradictionGroup(BaseModel):
    contradiction_id: str
    claim_ids: list[str]
    contradiction_type: str
    severity: ContradictionSeverity = "medium"
    explanation: str
    values: list[str] = Field(default_factory=list)


class UnsupportedClaim(BaseModel):
    claim_id: str
    text: str
    origin: str
    support_level: SupportLevel
    reason: str
    needs_human_review: bool = True


class EvidenceCoverageReport(BaseModel):
    generated_at: str
    source_count: int
    total_claims: int
    generated_claims: int
    source_claims: int
    supported_claims: int
    partially_supported_claims: int
    unsupported_claims: int
    contradicted_claims: int
    contradiction_count: int
    average_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    by_claim_type: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class EvidenceLedger(BaseModel):
    ledger_id: str
    thread_id: str
    generated_at: str
    sources: list[EvidenceSource] = Field(default_factory=list)
    quotes: list[EvidenceQuote] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    contradictions: list[ContradictionGroup] = Field(default_factory=list)
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)
    coverage: EvidenceCoverageReport
    citation_map: dict[str, list[ClaimCitation]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
