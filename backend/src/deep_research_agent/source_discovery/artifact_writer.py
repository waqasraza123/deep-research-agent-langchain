from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import SourceDiscoveryBatch, model_to_plain

DISCOVERY_ARTIFACTS = (
    "source_acquisition_plan.json",
    "source_acquisition_plan.md",
    "search_queries.json",
    "source_candidates.json",
    "source_selection.json",
    "source_discovery_summary.md",
    "source_discovery.md",
)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def write_source_discovery_artifacts(thread_dir: Path, batch: SourceDiscoveryBatch) -> list[str]:
    files = {
        "source_acquisition_plan.json": _json(model_to_plain(batch.plan)),
        "source_acquisition_plan.md": _plan_markdown(batch),
        "search_queries.json": _json(
            [
                q.model_dump(mode="json") if hasattr(q, "model_dump") else q.dict()
                for q in batch.plan.query_plan.queries
            ]
        ),
        "source_candidates.json": _json(
            [
                c.model_dump(mode="json") if hasattr(c, "model_dump") else c.dict()
                for c in batch.candidates
            ]
        ),
        "source_selection.json": _json(
            {
                "selected_candidates": [
                    c.model_dump(mode="json") if hasattr(c, "model_dump") else c.dict()
                    for c in batch.selected_candidates
                ],
                "decisions": [
                    d.model_dump(mode="json") if hasattr(d, "model_dump") else d.dict()
                    for d in batch.decisions
                ],
            }
        ),
        "source_discovery_summary.md": _summary_markdown(batch),
        "source_discovery.md": _summary_markdown(batch),
    }
    written: list[str] = []
    for rel_path, content in files.items():
        path = thread_dir / rel_path
        path.write_text(content, encoding="utf-8")
        written.append(rel_path)
    return written


def _plan_markdown(batch: SourceDiscoveryBatch) -> str:
    plan = batch.plan
    lines = [
        "# Source Acquisition Plan",
        "",
        f"Question: {plan.question}",
        "",
        f"- Discovery enabled: `{plan.settings.discovery_enabled}`",
        f"- Provider: `{plan.provider_config.provider}`",
        f"- Max queries: `{plan.settings.max_queries}`",
        f"- Max candidates per query: `{plan.settings.max_candidates_per_query}`",
        f"- Max selected sources: `{plan.settings.max_selected_sources}`",
        "",
        "## Source Types",
        "",
        f"- Required: {', '.join(plan.required_source_types) or 'none'}",
        f"- Preferred: {', '.join(plan.preferred_source_types) or 'none'}",
        f"- Optional: {', '.join(plan.optional_source_types) or 'none'}",
        "",
        "## Rationale",
        "",
    ]
    lines.extend(f"- {item}" for item in plan.rationale)
    if plan.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in plan.warnings)
    lines.extend(["", "## Queries", ""])
    for query in plan.query_plan.queries:
        lines.append(f"- `{query.query_id}` {query.intent}: {query.text}")
    return "\n".join(lines).strip() + "\n"


def _summary_markdown(batch: SourceDiscoveryBatch) -> str:
    summary = batch.summary
    lines = [
        "# Source Discovery Summary",
        "",
        f"- Discovery enabled: `{summary.discovery_enabled}`",
        f"- Provider: `{summary.provider}`",
        f"- Queries: `{summary.query_count}`",
        f"- Candidates: `{summary.candidate_count}`",
        f"- Selected sources: `{summary.selected_count}`",
    ]
    if summary.skipped_reason:
        lines.append(f"- Skipped reason: {summary.skipped_reason}")
    lines.extend(["", "## Selected Sources", ""])
    if batch.selected_candidates:
        for candidate in batch.selected_candidates:
            lines.append(
                f"- {candidate.title or candidate.url} ({candidate.url}) - "
                f"{candidate.source_type_hint}, score {candidate.ranking_score:.2f}, "
                "automatically discovered"
            )
    else:
        lines.append("- No sources were automatically selected.")
    lines.extend(["", "## Coverage Notes", ""])
    if summary.coverage_notes:
        lines.extend(f"- {item}" for item in summary.coverage_notes)
    else:
        lines.append("- No coverage notes.")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in summary.warnings)
    return "\n".join(lines).strip() + "\n"
