from __future__ import annotations

import json
from pathlib import Path

from .context_pack import render_agent_context_block
from .contracts import ContextPackBuildResult, model_to_plain

RETRIEVAL_ARTIFACTS = (
    "retrieval_index.json",
    "retrieval_queries.json",
    "retrieval_results.json",
    "context_packs.json",
    "context_packs.md",
    "retrieval_coverage.md",
)


def _json_text(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write_retrieval_artifacts(run_dir: Path, result: ContextPackBuildResult) -> list[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "retrieval_index.json": _json_text(model_to_plain(result.index)),
        "retrieval_queries.json": _json_text(
            {"thread_id": result.thread_id, "queries": model_to_plain(result.queries)}
        ),
        "retrieval_results.json": _json_text(
            {"thread_id": result.thread_id, "results": model_to_plain(result.results)}
        ),
        "context_packs.json": _json_text(
            {"thread_id": result.thread_id, "packs": model_to_plain(result.packs)}
        ),
        "context_packs.md": render_context_packs_md(result),
        "retrieval_coverage.md": render_retrieval_coverage_md(result),
    }
    for rel_path, content in files.items():
        (run_dir / rel_path).write_text(content, encoding="utf-8")
    return sorted(files)


def render_context_packs_md(result: ContextPackBuildResult) -> str:
    lines = [
        "# Context Packs",
        "",
        f"- Thread: `{result.thread_id}`",
        f"- Question: {result.question}",
        f"- Generated: `{result.generated_at}`",
        "",
    ]
    for pack_name, pack in result.packs.items():
        lines.extend(
            [
                f"## {pack_name}",
                "",
                f"- Pack ID: `{pack.pack_id}`",
                f"- Items: {len(pack.items)}",
                f"- Total chars: {pack.total_chars}/{pack.max_chars}",
                f"- Sources: {', '.join(pack.coverage_summary.covered_sources) or 'None'}",
                "",
            ]
        )
        if pack.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in pack.warnings)
            lines.append("")
        for idx, item in enumerate(pack.items, start=1):
            section = " > ".join(item.section_path) if item.section_path else "source body"
            excerpt = item.text.strip().replace("\n", " ")
            if len(excerpt) > 500:
                excerpt = excerpt[:500].rsplit(" ", 1)[0] + "..."
            lines.extend(
                [
                    f"### {idx}. {item.source_id} `{item.chunk_id}`",
                    "",
                    f"- Title: {item.title or 'Untitled'}",
                    f"- URL: {item.url}",
                    f"- Section: {section}",
                    f"- Score: {item.score:.3f}",
                    f"- Reason: {item.relevance_reason}",
                    f"- Citation: {item.citation_hint}",
                    "",
                    excerpt,
                    "",
                ]
            )
    agent_pack = result.packs.get("agent_context_pack")
    if agent_pack is not None:
        lines.extend(
            [
                "## Agent Instruction Block",
                "",
                "```text",
                render_agent_context_block(agent_pack),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_retrieval_coverage_md(result: ContextPackBuildResult) -> str:
    coverage = result.coverage_summary
    lines = [
        "# Retrieval Coverage",
        "",
        f"- Thread: `{result.thread_id}`",
        f"- Generated: `{result.generated_at}`",
        f"- Documents indexed: {len(result.index.documents)}",
        f"- Chunks indexed: {result.index.chunk_count}",
        f"- Queries: {coverage.query_count}",
        f"- Results: {coverage.result_count}",
        f"- Selected chunks: {coverage.selected_chunk_count}",
        f"- Covered sources: {', '.join(coverage.covered_sources) or 'None'}",
        "",
        "## Missing Queries",
        "",
    ]
    if coverage.missing_queries:
        lines.extend(f"- `{query_id}`" for query_id in coverage.missing_queries)
    else:
        lines.append("- None")
    lines.extend(["", "## Entity Coverage", ""])
    lines.append("- Covered: " + (", ".join(coverage.covered_entities) or "None"))
    lines.append("- Missing: " + (", ".join(coverage.missing_entities) or "None"))
    lines.extend(["", "## Values", ""])
    lines.append("- Dates: " + (", ".join(coverage.covered_dates[:20]) or "None"))
    lines.append("- Numbers: " + (", ".join(coverage.covered_numbers[:20]) or "None"))
    lines.extend(["", "## Warnings", ""])
    warnings = [*result.warnings, *coverage.warnings]
    if warnings:
        lines.extend(f"- {warning}" for warning in dict.fromkeys(warnings))
    else:
        lines.append("- None")
    return "\n".join(lines).rstrip() + "\n"
