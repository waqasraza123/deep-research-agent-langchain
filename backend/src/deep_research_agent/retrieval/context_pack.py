from __future__ import annotations

import hashlib
from collections import Counter

from .contracts import (
    ContextPack,
    ContextPackBuildResult,
    ContextPackItem,
    ContextPackType,
    RetrievalCoverageSummary,
    RetrievalIndex,
    RetrievalQuery,
    RetrievalResult,
)
from .indexer import now_iso_utc
from .reranker import diversify_results

PACK_LIMITS: dict[ContextPackType, int] = {
    "agent_context_pack": 11000,
    "evidence_context_pack": 15000,
    "synthesis_context_pack": 18000,
    "verification_context_pack": 9000,
}


def _pack_id(pack_type: str, question: str, result_count: int) -> str:
    digest = hashlib.sha1(f"{pack_type}|{question}|{result_count}".encode("utf-8")).hexdigest()
    return f"{pack_type}-{digest[:10]}"


def citation_hint(result: RetrievalResult) -> str:
    chunk = result.chunk
    title = chunk.title or chunk.domain or chunk.url
    section = " > ".join(chunk.section_path) if chunk.section_path else "source body"
    return f"Cite {chunk.source_id} ({title}), section `{section}`, URL: {chunk.url}"


def relevance_reason(result: RetrievalResult) -> str:
    if result.score.reasons:
        return "; ".join(result.score.reasons[:3])
    return f"Selected with retrieval score {result.score.total_score:.3f}"


def build_coverage_summary(
    *,
    queries: list[RetrievalQuery],
    results: list[RetrievalResult],
    selected: list[RetrievalResult],
    index: RetrievalIndex,
) -> RetrievalCoverageSummary:
    result_query_ids = {result.query_id for result in results}
    all_entities = sorted({entity for query in queries for entity in query.entities})
    selected_entities = sorted({entity for result in selected for entity in result.chunk.entities})
    selected_entity_norm = {entity.lower() for entity in selected_entities}
    covered_entities = [entity for entity in all_entities if entity.lower() in selected_entity_norm]
    missing_entities = [
        entity for entity in all_entities if entity.lower() not in selected_entity_norm
    ]
    covered_dates = sorted({date for result in selected for date in result.chunk.dates})
    covered_numbers = sorted({number for result in selected for number in result.chunk.numbers})
    covered_sources = sorted({result.chunk.source_id for result in selected})
    warnings: list[str] = []
    if not selected:
        warnings.append("No retrieval chunks selected.")
    if len(covered_sources) < min(2, len(index.documents)):
        warnings.append("Low source diversity in selected context.")
    return RetrievalCoverageSummary(
        query_count=len(queries),
        result_count=len(results),
        selected_chunk_count=len(selected),
        source_count=len(index.documents),
        covered_sources=covered_sources,
        missing_queries=[
            query.query_id for query in queries if query.query_id not in result_query_ids
        ],
        covered_entities=covered_entities,
        missing_entities=missing_entities,
        covered_dates=covered_dates,
        covered_numbers=covered_numbers,
        warnings=warnings,
    )


def _item_from_result(result: RetrievalResult, *, max_item_chars: int) -> ContextPackItem:
    text = result.chunk.text.strip()
    if len(text) > max_item_chars:
        text = text[:max_item_chars].rsplit(" ", 1)[0].rstrip() + "\n[chunk truncated]"
    warnings = list(result.chunk.warnings)
    if (result.chunk.citation_readiness_score or 0.0) < 0.35:
        warnings.append("Citation readiness is weak; verify before quoting.")
    if result.chunk.freshness_status in {"stale", "possibly_stale"}:
        warnings.append(f"Freshness warning: {result.chunk.freshness_status}.")
    if result.score.duplication_penalty:
        warnings.append("Near-duplicate chunk; use only if unique detail is needed.")
    return ContextPackItem(
        chunk_id=result.chunk.chunk_id,
        source_id=result.chunk.source_id,
        url=result.chunk.url,
        title=result.chunk.title,
        section_path=result.chunk.section_path,
        text=text,
        score=result.score.total_score,
        relevance_reason=relevance_reason(result),
        citation_hint=citation_hint(result),
        warnings=warnings,
        metadata={
            "score_reasons": result.score.reasons,
            "matched_terms": result.score.matched_terms,
            "matched_entities": result.score.matched_entities,
            "matched_values": result.score.matched_values,
            "domain": result.chunk.domain,
            "source_role": result.chunk.source_role,
        },
    )


def build_context_pack(
    *,
    pack_type: ContextPackType,
    question: str,
    queries: list[RetrievalQuery],
    results: list[RetrievalResult],
    index: RetrievalIndex,
    max_chars: int | None = None,
) -> ContextPack:
    max_chars = max_chars or PACK_LIMITS[pack_type]
    if pack_type == "verification_context_pack":
        candidates = [
            result
            for result in results
            if result.chunk.citation_readiness_score is None
            or result.chunk.citation_readiness_score >= 0.3
        ]
    elif pack_type == "evidence_context_pack":
        candidates = sorted(
            results,
            key=lambda result: (
                result.chunk.citation_readiness_score or 0.0,
                result.score.total_score,
            ),
            reverse=True,
        )
    elif pack_type == "synthesis_context_pack":
        candidates = sorted(
            results,
            key=lambda result: (
                result.score.total_score,
                result.chunk.source_quality_score or 0.0,
            ),
            reverse=True,
        )
    else:
        candidates = results

    diversified = diversify_results(
        candidates,
        max_per_source=2 if pack_type == "agent_context_pack" else 4,
        limit=24,
    )
    items: list[ContextPackItem] = []
    total = 0
    max_item_chars = 1800 if pack_type == "agent_context_pack" else 2400
    for result in diversified:
        item = _item_from_result(result, max_item_chars=max_item_chars)
        projected = total + len(item.text)
        if projected > max_chars and items:
            continue
        items.append(item)
        total += len(item.text)
        if total >= max_chars:
            break

    selected_by_id = {item.chunk_id for item in items}
    selected_results = [result for result in diversified if result.chunk.chunk_id in selected_by_id]
    coverage = build_coverage_summary(
        queries=queries,
        results=results,
        selected=selected_results,
        index=index,
    )
    warnings = list(coverage.warnings)
    if len(items) < 2 and len(index.chunks) >= 2:
        warnings.append("Context pack has very few chunks; source extraction may be sparse.")
    source_counts = Counter(item.source_id for item in items)
    if source_counts and max(source_counts.values()) > 3:
        warnings.append("One source dominates this context pack.")

    return ContextPack(
        pack_id=_pack_id(pack_type, question, len(items)),
        pack_type=pack_type,
        question=question,
        generated_at=now_iso_utc(),
        items=items,
        coverage_summary=coverage,
        warnings=warnings,
        max_chars=max_chars,
        total_chars=total,
        metadata={
            "source_counts": dict(source_counts),
            "pack_strategy": pack_type,
        },
    )


def build_all_context_packs(
    *,
    thread_id: str | None,
    question: str,
    index: RetrievalIndex,
    queries: list[RetrievalQuery],
    results: list[RetrievalResult],
    context_pack_max_chars: int | None = None,
) -> ContextPackBuildResult:
    packs = {
        pack_type: build_context_pack(
            pack_type=pack_type,
            question=question,
            queries=queries,
            results=results,
            index=index,
            max_chars=min(PACK_LIMITS[pack_type], context_pack_max_chars)
            if context_pack_max_chars
            else None,
        )
        for pack_type in PACK_LIMITS
    }
    selected_results = [
        result
        for result in results
        if result.chunk.chunk_id
        in {item.chunk_id for pack in packs.values() for item in pack.items}
    ]
    coverage = build_coverage_summary(
        queries=queries,
        results=results,
        selected=selected_results,
        index=index,
    )
    warnings = list(index.warnings)
    warnings.extend(coverage.warnings)
    return ContextPackBuildResult(
        thread_id=thread_id,
        question=question,
        generated_at=now_iso_utc(),
        index=index,
        queries=queries,
        results=results,
        packs=packs,
        coverage_summary=coverage,
        warnings=warnings,
    )


def render_agent_context_block(pack: ContextPack) -> str:
    lines = [
        "Local retrieval context pack (citation-ready excerpts from fetched sources):",
        f"- Pack: {pack.pack_id}",
        f"- Chunks: {len(pack.items)}",
        f"- Sources: {', '.join(pack.coverage_summary.covered_sources) or 'none'}",
        "",
    ]
    for idx, item in enumerate(pack.items, start=1):
        section = " > ".join(item.section_path) if item.section_path else "source body"
        lines.extend(
            [
                f"[R{idx}] {item.source_id} | {item.title or item.url}",
                f"URL: {item.url}",
                f"Section: {section}",
                f"Why relevant: {item.relevance_reason}",
                f"Citation hint: {item.citation_hint}",
                "Excerpt:",
                item.text.strip(),
                "",
            ]
        )
        if item.warnings:
            lines.append("Warnings: " + "; ".join(item.warnings[:4]))
            lines.append("")
    if pack.warnings:
        lines.append("Pack warnings: " + "; ".join(pack.warnings[:6]))
    return "\n".join(lines).strip()
