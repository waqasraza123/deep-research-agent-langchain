from __future__ import annotations

from deep_research_agent.source_intelligence.dedupe import normalize_url

from .contracts import ExtractedEntity, ExtractedTopic, MemoryContext, MemoryRecord
from .entity_extractor import extract_entities_and_topics
from .repository import MemoryRepository, normalize_question
from .source_cache import SourceCache


def _dedupe_records(records: list[MemoryRecord], *, limit: int) -> list[MemoryRecord]:
    seen: set[str] = set()
    out: list[MemoryRecord] = []
    for record in records:
        if record.memory_id in seen:
            continue
        seen.add(record.memory_id)
        out.append(record)
        if len(out) >= limit:
            break
    return out


def _merge_entities(
    records: list[MemoryRecord],
    current: list[ExtractedEntity],
) -> list[ExtractedEntity]:
    by_key: dict[tuple[str, str], ExtractedEntity] = {}
    for entity in current:
        by_key[(entity.entity_type.value, entity.name.lower())] = entity
    for record in records:
        for entity in record.entities:
            key = (entity.entity_type.value, entity.name.lower())
            existing = by_key.get(key)
            if existing is None or entity.mentions > existing.mentions:
                by_key[key] = entity
    return sorted(
        by_key.values(),
        key=lambda item: (item.confidence, item.mentions, item.name),
        reverse=True,
    )[:40]


def _merge_topics(
    records: list[MemoryRecord],
    current: list[ExtractedTopic],
) -> list[ExtractedTopic]:
    by_name: dict[str, ExtractedTopic] = {topic.name.lower(): topic for topic in current}
    for record in records:
        for topic in record.topics:
            old = by_name.get(topic.name.lower())
            if old is None or topic.score > old.score:
                by_name[topic.name.lower()] = topic
    return sorted(by_name.values(), key=lambda item: (item.score, item.name), reverse=True)[:30]


class MemoryRetriever:
    def __init__(
        self,
        repository: MemoryRepository,
        source_cache: SourceCache | None = None,
    ) -> None:
        self.repository = repository
        self.source_cache = source_cache or SourceCache(repository)

    def retrieve(self, *, question: str, urls: list[str]) -> MemoryContext:
        extraction = extract_entities_and_topics(question=question, text="", title=None, url=None)
        query_records = self.repository.search(question, limit=12)
        source_decisions = [
            self.source_cache.evaluate(url=normalize_url(url) if url else url)
            for url in urls
            if url
        ]
        source_records: list[MemoryRecord] = []
        for decision in source_decisions:
            for memory_id in decision.matched_memory_ids:
                source_records.extend(
                    record
                    for record in self.repository.list_records(limit=2000)
                    if record.memory_id == memory_id
                )
        source_records = _dedupe_records(source_records, limit=12)
        combined_records = _dedupe_records(query_records + source_records, limit=20)

        stale_warnings = [
            decision.freshness_warning
            for decision in source_decisions
            if decision.freshness_warning
        ]
        prior_artifacts = []
        for record in combined_records:
            prior_artifacts.extend(record.artifacts)
        confidence = 0.0
        if combined_records:
            confidence = min(0.92, 0.25 + len(combined_records) * 0.05)
        if any(decision.reuse_allowed for decision in source_decisions):
            confidence = min(0.98, confidence + 0.2)

        return MemoryContext(
            question=question,
            normalized_question=normalize_question(question),
            similar_previous_questions=query_records[:8],
            previously_useful_sources=source_records[:8],
            known_entities=_merge_entities(combined_records, extraction.entities),
            known_topics=_merge_topics(combined_records, extraction.topics),
            stale_warnings=list(dict.fromkeys(stale_warnings)),
            suggested_source_reuse_candidates=source_decisions,
            prior_artifact_links=list({(a.thread_id, a.path): a for a in prior_artifacts}.values()),
            confidence_score=round(confidence, 3),
        )
