from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlsplit

from deep_research_agent.source_intelligence.dedupe import content_hash, normalize_url

from .contracts import ArtifactReference, MemoryRecord, SourceReuseDecision
from .repository import MemoryRepository, source_domain
from .topic_index import tokenize


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _age_days(value: str) -> int | None:
    dt = _parse_iso(value)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - dt).days)


def _freshness_warning(records: list[MemoryRecord], stale_after_days: int) -> str | None:
    if not records:
        return None
    ages = [_age_days(record.last_seen_at) for record in records]
    ages = [age for age in ages if age is not None]
    if not ages:
        return "Previous memory has unknown freshness."
    oldest = max(ages)
    if oldest > stale_after_days:
        return f"Previous source memory is stale: oldest match is {oldest} days old."
    return None


def _decision(
    *,
    records: list[MemoryRecord],
    reason: str,
    base_confidence: float,
    stale_after_days: int,
) -> SourceReuseDecision:
    warning = _freshness_warning(records, stale_after_days)
    confidence = base_confidence
    if warning:
        confidence = min(confidence, 0.55)
    artifacts: list[ArtifactReference] = []
    for record in records:
        artifacts.extend(record.artifacts)
    thread_ids = sorted({record.thread_id for record in records})
    return SourceReuseDecision(
        reuse_allowed=bool(records) and warning is None and confidence >= 0.6,
        reuse_reason=reason if records else "no_previous_memory_match",
        freshness_warning=warning,
        previous_thread_ids=thread_ids,
        previous_artifacts=list({(a.thread_id, a.path): a for a in artifacts}.values()),
        confidence_score=round(confidence if records else 0.0, 3),
        matched_memory_ids=[record.memory_id for record in records],
    )


class SourceCache:
    def __init__(self, repository: MemoryRepository, *, stale_after_days: int = 30) -> None:
        self.repository = repository
        self.stale_after_days = max(1, int(stale_after_days))

    def evaluate(
        self,
        *,
        url: str,
        canonical_url: str | None = None,
        text: str | None = None,
        title: str | None = None,
    ) -> SourceReuseDecision:
        exact = self.repository.find_by_exact_url(url)
        if exact:
            return _decision(
                records=exact,
                reason="exact_url_match",
                base_confidence=0.98,
                stale_after_days=self.stale_after_days,
            )

        normalized = normalize_url(url)
        normalized_matches = self.repository.find_by_normalized_url(normalized)
        if normalized_matches:
            return _decision(
                records=normalized_matches,
                reason="normalized_url_match",
                base_confidence=0.93,
                stale_after_days=self.stale_after_days,
            )

        if canonical_url:
            canonical = normalize_url(canonical_url)
            canonical_matches = self.repository.find_by_canonical_url(canonical)
            if canonical_matches:
                return _decision(
                    records=canonical_matches,
                    reason="canonical_url_match",
                    base_confidence=0.9,
                    stale_after_days=self.stale_after_days,
                )

        if text:
            digest = content_hash(text)
            hash_matches = self.repository.find_by_content_hash(digest)
            if hash_matches:
                return _decision(
                    records=hash_matches,
                    reason="content_hash_match",
                    base_confidence=0.88,
                    stale_after_days=self.stale_after_days,
                )

        title_terms = tokenize(title or "")
        domain = source_domain(url) or urlsplit(normalized).hostname or ""
        domain_title_matches = self.repository.find_by_domain_title(domain, title_terms)
        if domain_title_matches:
            return _decision(
                records=domain_title_matches,
                reason="domain_title_similarity_match",
                base_confidence=0.68,
                stale_after_days=self.stale_after_days,
            )

        return SourceReuseDecision(
            reuse_allowed=False,
            reuse_reason="no_previous_memory_match",
            freshness_warning=None,
            confidence_score=0.0,
        )
