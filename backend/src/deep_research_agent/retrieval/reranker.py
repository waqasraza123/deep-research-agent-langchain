from __future__ import annotations

from collections import Counter

from .contracts import RetrievalResult


def diversify_results(
    results: list[RetrievalResult],
    *,
    max_per_source: int = 3,
    limit: int = 20,
) -> list[RetrievalResult]:
    source_counts: Counter[str] = Counter()
    unique_chunks: set[str] = set()
    selected: list[RetrievalResult] = []
    deferred: list[RetrievalResult] = []
    for result in sorted(results, key=lambda item: item.score.total_score, reverse=True):
        if result.chunk.chunk_id in unique_chunks:
            continue
        unique_chunks.add(result.chunk.chunk_id)
        if source_counts[result.chunk.source_id] >= max_per_source:
            deferred.append(result)
            continue
        source_counts[result.chunk.source_id] += 1
        selected.append(result)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected.extend(deferred[: limit - len(selected)])
    return selected[:limit]
