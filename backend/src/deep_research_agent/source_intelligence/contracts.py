from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CrawlSettings:
    follow_links: bool = False
    max_links_per_source: int = 0
    max_depth: int = 1
    global_link_budget: int = 20


@dataclass(frozen=True)
class LinkCandidate:
    url: str
    normalized_url: str
    parent_url: str
    anchor_text: str = ""
    crawl_depth: int = 1
    source_format: str = "html"


@dataclass(frozen=True)
class PrioritizedLink:
    candidate: LinkCandidate
    score: float
    reasons: tuple[str, ...] = ()
    skip_reason: str | None = None

    @property
    def should_skip(self) -> bool:
        return self.skip_reason is not None


@dataclass(frozen=True)
class QualityScore:
    extraction_quality: float
    content_length_score: float
    citation_usefulness_score: float
    freshness_signal: float
    authority_hint: float
    final_quality_score: float
    warnings: tuple[str, ...] = ()
    freshness_matches: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceRecord:
    url: str
    normalized_url: str
    source_kind: str
    parent_url: str | None = None
    crawl_depth: int = 0
    ok: bool = False
    skipped: bool = False
    skip_reason: str | None = None
    duplicate_of: str | None = None
    final_url: str | None = None
    canonical_url: str | None = None
    title: str | None = None
    content_type: str | None = None
    status_code: int | None = None
    truncated: bool = False
    fetched_at: str | None = None
    local_path: str | None = None
    strategy: str | None = None
    word_count: int = 0
    char_count: int = 0
    document_kind: str | None = None
    priority_score: float | None = None
    priority_reasons: tuple[str, ...] = ()
    discovered_anchor_text: str = ""
    quality_score: QualityScore | None = None
    source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.quality_score is not None:
            data["quality_score"] = self.quality_score.to_dict()
            data["final_quality_score"] = self.quality_score.final_quality_score
        else:
            data["quality_score"] = None
            data["final_quality_score"] = None
        return data


@dataclass
class CrawlBudgetStats:
    root_count: int = 0
    discovered_count: int = 0
    fetched_count: int = 0
    skipped_count: int = 0
    global_link_budget: int = 0
    global_link_budget_used: int = 0
    max_links_per_source: int = 0
    max_depth: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrawlResult:
    root_urls: list[str]
    sources: list[SourceRecord] = field(default_factory=list)
    discovered_links: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, str]] = field(default_factory=list)
    budget: CrawlBudgetStats = field(default_factory=CrawlBudgetStats)

    def usable_sources(self) -> list[SourceRecord]:
        return [s for s in self.sources if s.ok and not s.skipped]

    def to_sources_json(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.sources]
