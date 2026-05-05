from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import CrawlResult, SourceRecord


def source_graph_payload(result: CrawlResult) -> dict[str, Any]:
    fetched = [s.to_dict() for s in result.sources if s.local_path]
    skipped = [s.to_dict() for s in result.sources if s.skipped]
    return {
        "root_urls": result.root_urls,
        "discovered_links": result.discovered_links,
        "fetched_links": fetched,
        "skipped_links": skipped,
        "edges": result.edges,
        "quality_scores": {
            s.url: s.quality_score.to_dict()
            for s in result.sources
            if s.quality_score is not None
        },
        "dedupe_relationships": [
            {"url": s.url, "duplicate_of": s.duplicate_of, "reason": s.skip_reason}
            for s in result.sources
            if s.duplicate_of
        ],
        "budget": result.budget.to_dict(),
    }


def render_source_graph_markdown(result: CrawlResult) -> str:
    lines = [
        "# Source Graph",
        "",
        "## Crawl Budget",
        "",
        f"- Roots: {result.budget.root_count}",
        f"- Discovered links: {result.budget.discovered_count}",
        f"- Fetched links: {result.budget.fetched_count}",
        f"- Skipped links: {result.budget.skipped_count}",
        f"- Global expansion budget: {result.budget.global_link_budget_used}/"
        f"{result.budget.global_link_budget}",
        f"- Max links per source: {result.budget.max_links_per_source}",
        f"- Max depth: {result.budget.max_depth}",
        "",
        "## Roots",
        "",
    ]
    for url in result.root_urls:
        lines.append(f"- {url}")

    lines.extend(["", "## Fetched Sources", ""])
    for source in result.sources:
        if not source.local_path:
            continue
        label = source.source_id or "skipped"
        score = (
            f"{source.quality_score.final_quality_score:.3f}"
            if source.quality_score is not None
            else "n/a"
        )
        parent = f" parent={source.parent_url}" if source.parent_url else ""
        status = "skipped" if source.skipped else "usable"
        lines.append(
            f"- {label}: {source.url} ({source.source_kind}, depth {source.crawl_depth}, "
            f"{status}, quality {score}){parent}"
        )
        if source.skip_reason:
            lines.append(f"  - Skip reason: {source.skip_reason}")

    lines.extend(["", "## Skipped Links", ""])
    skipped = [s for s in result.sources if s.skipped]
    if not skipped:
        lines.append("- None")
    else:
        for source in skipped:
            reason = source.skip_reason or "skipped"
            dup = f", duplicate_of={source.duplicate_of}" if source.duplicate_of else ""
            lines.append(f"- {source.url}: {reason}{dup}")

    lines.extend(["", "## Parent Child Relationships", ""])
    if not result.edges:
        lines.append("- None")
    else:
        for edge in result.edges:
            lines.append(f"- {edge['parent_url']} -> {edge['child_url']}")

    return "\n".join(lines).rstrip() + "\n"


def write_source_graph_artifacts(thread_dir: Path, result: CrawlResult) -> None:
    (thread_dir / "source_graph.json").write_text(
        json.dumps(source_graph_payload(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (thread_dir / "source_graph.md").write_text(
        render_source_graph_markdown(result),
        encoding="utf-8",
    )


def write_sources_manifest(path: Path, records: list[SourceRecord]) -> None:
    existing_by_url: dict[str, dict[str, Any]] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for item in existing:
                    if isinstance(item, dict) and isinstance(item.get("url"), str):
                        existing_by_url[item["url"]] = item
        except Exception:
            existing_by_url = {}

    merged: list[dict[str, Any]] = []
    for record in records:
        data = record.to_dict()
        old = existing_by_url.get(record.url, {})
        if old:
            old.update(data)
            data = old
        merged.append(data)

    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
