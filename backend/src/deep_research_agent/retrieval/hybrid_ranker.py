from __future__ import annotations

from collections import Counter

from .contracts import (
    HybridRankingConfig,
    RetrievalIndex,
    RetrievalQuery,
    RetrievalResult,
    RetrievalScore,
)
from .embeddings import DisabledEmbeddingProvider, EmbeddingProvider, cosine_similarity
from .lexical import (
    LexicalIndex,
    group_duplicate_texts,
    overlap_score,
    phrase_match_score,
    token_set,
)


def _heading_score(
    query: RetrievalQuery, section_path: list[str], title: str | None
) -> tuple[float, list[str]]:
    haystack = " ".join([*(section_path or []), title or ""])
    query_terms = token_set(query.text)
    heading_terms = token_set(haystack)
    score, overlap = overlap_score(query_terms, heading_terms)
    return min(1.0, score), overlap


def _freshness_score(status: str | None, required: bool) -> tuple[float, str | None]:
    if not required:
        return 0.0, None
    mapping = {
        "current": 1.0,
        "recent": 0.75,
        "unknown": 0.25,
        "possibly_stale": -0.25,
        "stale": -0.5,
    }
    value = mapping.get((status or "unknown").lower(), 0.0)
    if value > 0:
        return value, f"freshness status `{status}`"
    if value < 0:
        return value, f"freshness warning `{status}`"
    return value, None


def rank_retrieval_results(
    index: RetrievalIndex,
    query: RetrievalQuery,
    *,
    config: HybridRankingConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[RetrievalResult]:
    config = config or HybridRankingConfig()
    provider = embedding_provider or DisabledEmbeddingProvider()
    lexical_index = LexicalIndex(index.chunks)
    lexical_hits = lexical_index.search(query.text, limit=config.candidate_k)
    if not lexical_hits and index.chunks:
        lexical_hits = [
            type("FallbackHit", (), {"chunk": chunk, "score": 0.0, "matched_terms": []})
            for chunk in index.chunks[: config.candidate_k]
        ]

    duplicate_groups = group_duplicate_texts([hit.chunk for hit in lexical_hits])
    duplicate_chunk_ids = {
        chunk_id
        for group in duplicate_groups.values()
        for chunk_id in sorted(group)[1:]
    }

    embedding_scores: dict[str, float] = {}
    if config.use_embeddings and config.embedding_weight > 0 and provider.enabled:
        query_vec = provider.embed_texts([query.text])[0]
        chunk_vecs = provider.embed_texts([hit.chunk.text for hit in lexical_hits])
        for hit, vec in zip(lexical_hits, chunk_vecs, strict=True):
            embedding_scores[hit.chunk.chunk_id] = max(0.0, cosine_similarity(query_vec, vec))

    prelim: list[RetrievalResult] = []
    for hit in lexical_hits:
        chunk = hit.chunk
        phrase_score, matched_phrases = phrase_match_score(chunk.text, query.phrases)
        entity_score, matched_entities = overlap_score(query.entities, chunk.entities)
        date_score, matched_dates = overlap_score(query.dates, chunk.dates)
        number_score, matched_numbers = overlap_score(query.numbers, chunk.numbers)
        value_score = max(date_score, number_score)
        heading_score, heading_matches = _heading_score(query, chunk.section_path, chunk.title)
        quality = chunk.source_quality_score or 0.0
        citation = chunk.citation_readiness_score or 0.0
        freshness, freshness_reason = _freshness_score(
            chunk.freshness_status, query.freshness_required
        )
        embedding = embedding_scores.get(chunk.chunk_id)
        duplication_penalty = 1.0 if chunk.chunk_id in duplicate_chunk_ids else 0.0

        total = (
            config.lexical_weight * hit.score
            + config.phrase_weight * phrase_score
            + config.entity_weight * entity_score
            + config.numeric_date_weight * value_score
            + config.heading_weight * heading_score
            + config.source_quality_weight * quality
            + config.citation_readiness_weight * citation
            + config.freshness_weight * freshness
            + (config.embedding_weight * (embedding or 0.0))
            - config.duplication_penalty_weight * duplication_penalty
        )

        reasons: list[str] = []
        if hit.matched_terms:
            reasons.append("matched terms: " + ", ".join(hit.matched_terms[:8]))
        if matched_phrases:
            reasons.append("matched phrase: " + "; ".join(matched_phrases[:3]))
        if matched_entities:
            reasons.append("entity overlap: " + ", ".join(matched_entities[:6]))
        values = [*matched_dates, *matched_numbers]
        if values:
            reasons.append("value/date overlap: " + ", ".join(values[:6]))
        if heading_matches:
            reasons.append("heading/title overlap: " + ", ".join(heading_matches[:6]))
        if quality > 0:
            reasons.append(f"source quality {quality:.2f}")
        if citation > 0:
            reasons.append(f"citation readiness {citation:.2f}")
        if freshness_reason:
            reasons.append(freshness_reason)
        if duplication_penalty:
            reasons.append("near-duplicate text penalty")

        prelim.append(
            RetrievalResult(
                query_id=query.query_id,
                rank=0,
                chunk=chunk,
                score=RetrievalScore(
                    total_score=total,
                    lexical_score=hit.score,
                    phrase_score=phrase_score,
                    entity_overlap_score=entity_score,
                    numeric_date_overlap_score=value_score,
                    heading_score=heading_score,
                    source_quality_score=quality,
                    citation_readiness_score=citation,
                    freshness_score=freshness,
                    embedding_score=embedding,
                    duplication_penalty=duplication_penalty,
                    reasons=reasons,
                    matched_terms=list(hit.matched_terms),
                    matched_entities=matched_entities,
                    matched_values=values,
                ),
            )
        )

    prelim.sort(key=lambda result: result.score.total_score, reverse=True)
    source_counts: Counter[str] = Counter()
    selected: list[RetrievalResult] = []
    overflow: list[RetrievalResult] = []
    for result in prelim:
        count = source_counts[result.chunk.source_id]
        if count >= config.max_chunks_per_source:
            result.score.diversity_penalty = float(count - config.max_chunks_per_source + 1)
            result.score.total_score -= (
                config.diversity_penalty_weight * result.score.diversity_penalty
            )
            result.score.reasons.append("source diversity penalty")
            overflow.append(result)
            continue
        if result.score.total_score >= config.min_score:
            source_counts[result.chunk.source_id] += 1
            selected.append(result)
        if len(selected) >= config.top_k:
            break
    if len(selected) < config.top_k:
        overflow.sort(key=lambda result: result.score.total_score, reverse=True)
        selected.extend(overflow[: config.top_k - len(selected)])
    selected.sort(key=lambda result: result.score.total_score, reverse=True)
    for idx, result in enumerate(selected[: config.top_k], start=1):
        result.rank = idx
    return selected[: config.top_k]


def rank_all_queries(
    index: RetrievalIndex,
    queries: list[RetrievalQuery],
    *,
    config: HybridRankingConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    for query in queries:
        results.extend(
            rank_retrieval_results(
                index,
                query,
                config=config,
                embedding_provider=embedding_provider,
            )
        )
    return results
