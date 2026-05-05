from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

from deep_research_agent.source_identity import source_domain
from deep_research_agent.tools import FetchResult, fetch_document

from .contracts import CrawlBudgetStats, CrawlResult, CrawlSettings, PrioritizedLink, SourceRecord
from .dedupe import DedupeIndex, normalize_url
from .link_extractor import extraction_attempts_from_fetch_result, normalize_and_validate_link
from .prioritizer import prioritize_links
from .quality import score_source
from .source_graph import write_source_graph_artifacts, write_sources_manifest

FetchDocument = Callable[..., FetchResult]
FetchStartCallback = Callable[[str, str], None]
FetchCompleteCallback = Callable[[SourceRecord], None]


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _source_record_from_fetch(
    *,
    result: FetchResult,
    normalized_url: str,
    source_kind: str,
    local_path: str,
    parent_url: str | None,
    crawl_depth: int,
    priority: PrioritizedLink | None,
    fetched_at: str,
) -> SourceRecord:
    return SourceRecord(
        url=result.url,
        normalized_url=normalized_url,
        source_kind=source_kind,
        parent_url=parent_url,
        crawl_depth=crawl_depth,
        ok=result.ok,
        final_url=result.final_url,
        canonical_url=result.canonical_url,
        title=result.title,
        content_type=result.content_type,
        status_code=result.status_code,
        truncated=result.truncated,
        fetched_at=fetched_at,
        local_path=local_path,
        strategy=result.strategy,
        word_count=result.word_count,
        char_count=result.char_count,
        document_kind=result.kind,
        priority_score=priority.score if priority else None,
        priority_reasons=priority.reasons if priority else (),
        discovered_anchor_text=priority.candidate.anchor_text if priority else "",
        quality_score=score_source(result),
        source_domain=source_domain(result.final_url or result.url),
        content_hash=hashlib.sha1(
            " ".join((result.extracted_text or "").strip().lower().split()).encode("utf-8")
        ).hexdigest(),
    )


def _fetch_and_store(
    *,
    url: str,
    thread_id: str,
    sources_dir: Path,
    timeout_s: float,
    max_chars: int,
    fetcher: FetchDocument,
    dedupe: DedupeIndex,
    source_kind: str,
    parent_url: str | None,
    crawl_depth: int,
    priority: PrioritizedLink | None,
    require_extracted_links: bool = False,
    on_fetch_start: FetchStartCallback | None = None,
    on_fetch_complete: FetchCompleteCallback | None = None,
) -> tuple[SourceRecord, FetchResult | None]:
    normalized = normalize_url(url)
    pre = dedupe.precheck_url(url, normalized)
    if pre.is_duplicate:
        record = SourceRecord(
            url=url,
            normalized_url=normalized,
            source_kind=source_kind,
            parent_url=parent_url,
            crawl_depth=crawl_depth,
            skipped=True,
            skip_reason=pre.reason,
            duplicate_of=pre.duplicate_of,
            priority_score=priority.score if priority else None,
            priority_reasons=priority.reasons if priority else (),
            discovered_anchor_text=priority.candidate.anchor_text if priority else "",
        )
        if on_fetch_complete:
            on_fetch_complete(record)
        return record, None

    url_hash = _sha1(url)
    txt_path = sources_dir / f"{url_hash}.txt"
    meta_path = sources_dir / f"{url_hash}.json"
    local_path = f"runs/{thread_id}/sources/{url_hash}.txt"

    result: FetchResult | None = None
    fetched_at = _now_iso_utc()
    if not require_extracted_links and txt_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("ok") is True and meta.get("url") == url:
                text = txt_path.read_text(encoding="utf-8", errors="ignore")
                result = FetchResult(
                    ok=True,
                    url=meta.get("url") or url,
                    final_url=meta.get("final_url") or url,
                    status_code=int(meta.get("status_code") or 200),
                    content_type=meta.get("content_type") or "",
                    extracted_text=text,
                    title=meta.get("title"),
                    truncated=bool(meta.get("truncated")),
                    strategy=meta.get("strategy") or "direct",
                    word_count=int(meta.get("word_count") or 0),
                    char_count=int(meta.get("char_count") or len(text)),
                    kind=meta.get("document_kind") or meta.get("kind") or "unknown",
                    canonical_url=meta.get("canonical_url"),
                )
                fetched_at = meta.get("fetched_at") or fetched_at
        except Exception:
            result = None

    if result is None:
        try:
            if on_fetch_start:
                on_fetch_start(url, source_kind)
            result = fetcher(
                url,
                timeout_s=timeout_s,
                max_chars=max_chars,
                min_words=160,
                min_chars=1200,
            )
            txt_path.write_text(result.extracted_text, encoding="utf-8")
        except Exception as exc:
            txt_path.write_text(
                f"Fetch failed: {type(exc).__name__}: {exc}\nURL: {url}\n",
                encoding="utf-8",
            )
            record = SourceRecord(
                url=url,
                normalized_url=normalized,
                source_kind=source_kind,
                parent_url=parent_url,
                crawl_depth=crawl_depth,
                ok=False,
                skipped=False,
                skip_reason=None,
                fetched_at=fetched_at,
                local_path=local_path,
                strategy="error",
            )
            meta_path.write_text(json.dumps(record.to_dict(), ensure_ascii=False), encoding="utf-8")
            if on_fetch_complete:
                on_fetch_complete(record)
            return record, None

    record = _source_record_from_fetch(
        result=result,
        normalized_url=normalized,
        source_kind=source_kind,
        local_path=local_path,
        parent_url=parent_url,
        crawl_depth=crawl_depth,
        priority=priority,
        fetched_at=fetched_at,
    )

    if result.ok:
        dup = dedupe.check_fetched(
            url=url,
            normalized_url=normalized,
            canonical_url=result.canonical_url,
            title=result.title,
            text=result.extracted_text,
        )
        if dup.is_duplicate:
            record.skipped = True
            record.skip_reason = dup.reason
            record.duplicate_of = dup.duplicate_of
        else:
            source_id = f"S{dedupe.registered_count + 1}"
            record.source_id = source_id
            dedupe.register(
                source_id,
                url=url,
                normalized_url=normalized,
                canonical_url=result.canonical_url,
                title=result.title,
                text=result.extracted_text,
            )

    meta_path.write_text(json.dumps(record.to_dict(), ensure_ascii=False), encoding="utf-8")
    if on_fetch_complete:
        on_fetch_complete(record)
    return record, result


def _skipped_link_record(
    *,
    url: str,
    parent_url: str | None,
    reason: str,
    crawl_depth: int,
    priority: PrioritizedLink | None = None,
) -> SourceRecord:
    return SourceRecord(
        url=url,
        normalized_url=normalize_url(url) if url.startswith(("http://", "https://")) else url,
        source_kind="discovered",
        parent_url=parent_url,
        crawl_depth=crawl_depth,
        skipped=True,
        skip_reason=reason,
        priority_score=priority.score if priority else None,
        priority_reasons=priority.reasons if priority else (),
        discovered_anchor_text=priority.candidate.anchor_text if priority else "",
    )


def crawl_sources(
    *,
    question: str,
    root_urls: list[str],
    thread_id: str,
    thread_dir: Path,
    timeout_s: float,
    max_chars: int,
    follow_links: bool,
    max_links_per_source: int,
    fetcher: FetchDocument = fetch_document,
    max_depth: int = 1,
    global_link_budget: int | None = None,
    on_fetch_start: FetchStartCallback | None = None,
    on_fetch_complete: FetchCompleteCallback | None = None,
) -> CrawlResult:
    sources_dir = (thread_dir / "sources").resolve()
    sources_dir.mkdir(parents=True, exist_ok=True)

    max_links_per_source = max(0, min(int(max_links_per_source), 10))
    max_depth = max(0, min(int(max_depth), 1))
    root_urls = [u for u in root_urls if u]
    if global_link_budget is None:
        global_link_budget = min(20, len(root_urls) * max_links_per_source)
    global_link_budget = max(0, min(int(global_link_budget), 20))

    settings = CrawlSettings(
        follow_links=bool(follow_links),
        max_links_per_source=max_links_per_source,
        max_depth=max_depth,
        global_link_budget=global_link_budget,
    )
    result = CrawlResult(root_urls=root_urls)
    result.budget = CrawlBudgetStats(
        root_count=len(root_urls),
        global_link_budget=settings.global_link_budget,
        max_links_per_source=settings.max_links_per_source,
        max_depth=settings.max_depth,
    )

    dedupe = DedupeIndex()
    fetched_results: list[tuple[SourceRecord, FetchResult | None]] = []

    for root_url in root_urls:
        normalized, reason = normalize_and_validate_link(root_url, root_url)
        if reason or normalized is None:
            result.sources.append(
                _skipped_link_record(
                    url=root_url,
                    parent_url=None,
                    reason=reason or "invalid_url",
                    crawl_depth=0,
                )
            )
            continue
        record, fetch_result = _fetch_and_store(
            url=normalized,
            thread_id=thread_id,
            sources_dir=sources_dir,
            timeout_s=timeout_s,
            max_chars=max_chars,
            fetcher=fetcher,
            dedupe=dedupe,
            source_kind="root",
            parent_url=None,
            crawl_depth=0,
            priority=None,
            require_extracted_links=settings.follow_links,
            on_fetch_start=on_fetch_start,
            on_fetch_complete=on_fetch_complete,
        )
        result.sources.append(record)
        fetched_results.append((record, fetch_result))

    if settings.follow_links and settings.max_links_per_source > 0 and settings.max_depth >= 1:
        for root_record, fetch_result in fetched_results:
            if fetch_result is None or not fetch_result.ok or root_record.skipped:
                continue
            attempts = extraction_attempts_from_fetch_result(fetch_result, root_record.url)
            candidates = []
            for attempt in attempts:
                if attempt.candidate is not None:
                    candidates.append(attempt.candidate)
                    continue
                skipped_url = attempt.href or root_record.url
                result.discovered_links.append(
                    {
                        "url": skipped_url,
                        "normalized_url": skipped_url,
                        "parent_url": root_record.url,
                        "anchor_text": attempt.anchor_text,
                        "crawl_depth": 1,
                        "priority_score": None,
                        "priority_reasons": [],
                        "skip_reason": attempt.skip_reason or "invalid_link",
                    }
                )
                result.sources.append(
                    _skipped_link_record(
                        url=skipped_url,
                        parent_url=root_record.url,
                        reason=attempt.skip_reason or "invalid_link",
                        crawl_depth=1,
                    )
                )

            prioritized = prioritize_links(candidates, root_url=root_record.url, question=question)
            per_source_used = 0
            for priority in prioritized:
                candidate = priority.candidate
                discovered = {
                    "url": candidate.url,
                    "normalized_url": candidate.normalized_url,
                    "parent_url": candidate.parent_url,
                    "anchor_text": candidate.anchor_text,
                    "crawl_depth": candidate.crawl_depth,
                    "priority_score": priority.score,
                    "priority_reasons": list(priority.reasons),
                    "skip_reason": priority.skip_reason,
                }
                result.discovered_links.append(discovered)

                if priority.skip_reason:
                    result.sources.append(
                        _skipped_link_record(
                            url=candidate.url,
                            parent_url=candidate.parent_url,
                            reason=priority.skip_reason,
                            crawl_depth=candidate.crawl_depth,
                            priority=priority,
                        )
                    )
                    continue
                if per_source_used >= settings.max_links_per_source:
                    result.sources.append(
                        _skipped_link_record(
                            url=candidate.url,
                            parent_url=candidate.parent_url,
                            reason="max_links_per_source",
                            crawl_depth=candidate.crawl_depth,
                            priority=priority,
                        )
                    )
                    continue
                if result.budget.global_link_budget_used >= settings.global_link_budget:
                    result.sources.append(
                        _skipped_link_record(
                            url=candidate.url,
                            parent_url=candidate.parent_url,
                            reason="global_crawl_budget",
                            crawl_depth=candidate.crawl_depth,
                            priority=priority,
                        )
                    )
                    continue

                record, _child_result = _fetch_and_store(
                    url=candidate.url,
                    thread_id=thread_id,
                    sources_dir=sources_dir,
                    timeout_s=timeout_s,
                    max_chars=max_chars,
                    fetcher=fetcher,
                    dedupe=dedupe,
                    source_kind="discovered",
                    parent_url=candidate.parent_url,
                    crawl_depth=candidate.crawl_depth,
                    priority=priority,
                    on_fetch_start=on_fetch_start,
                    on_fetch_complete=on_fetch_complete,
                )
                result.sources.append(record)
                result.edges.append(
                    {"parent_url": candidate.parent_url, "child_url": candidate.url}
                )
                per_source_used += 1
                result.budget.global_link_budget_used += 1

    result.budget.discovered_count = len(result.discovered_links)
    result.budget.fetched_count = len([s for s in result.sources if s.local_path])
    result.budget.skipped_count = len([s for s in result.sources if s.skipped])

    write_source_graph_artifacts(thread_dir, result)
    write_sources_manifest(thread_dir / "sources.json", result.sources)
    return result
