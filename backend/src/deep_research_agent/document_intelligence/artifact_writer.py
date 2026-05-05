from __future__ import annotations

import json
from pathlib import Path

from .contracts import DocumentIntelligenceBatch, model_to_plain

DOCUMENT_PROFILE_JSON = "document_profiles.json"
DOCUMENT_PROFILE_MD = "document_profiles.md"
DOCUMENT_CHUNKS_JSONL = "document_chunks.jsonl"
DOCUMENT_TABLES_JSON = "document_tables.json"
DOCUMENT_CITATIONS_JSON = "document_citations.json"
DOCUMENT_WARNINGS_MD = "document_warnings.md"

DOCUMENT_INTELLIGENCE_ARTIFACTS = (
    DOCUMENT_PROFILE_JSON,
    DOCUMENT_PROFILE_MD,
    DOCUMENT_CHUNKS_JSONL,
    DOCUMENT_TABLES_JSON,
    DOCUMENT_CITATIONS_JSON,
    DOCUMENT_WARNINGS_MD,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_profiles_markdown(batch: DocumentIntelligenceBatch) -> str:
    lines = ["# Document Profiles", ""]
    if not batch.profiles:
        lines.append("No document profiles were generated.")
        return "\n".join(lines).rstrip() + "\n"
    for profile in batch.profiles:
        lines.extend(
            [
                f"## {profile.source_id}: {profile.title or profile.url or 'Untitled source'}",
                "",
                f"- URL: {profile.url}",
                f"- Domain: {profile.domain or 'unknown'}",
                f"- Type: {profile.source_type}",
                f"- Content hash: `{profile.content_hash}`",
                f"- Sections: {len(profile.sections)}",
                f"- Chunks: {len(profile.chunks)}",
                f"- Tables: {len(profile.tables)}",
                f"- Citations: {len(profile.citations)}",
                "",
            ]
        )
        present = [feature.name for feature in profile.features if feature.present]
        if present:
            lines.append("- Features: " + ", ".join(present))
            lines.append("")
        if profile.quality_summary:
            lines.append("### Quality")
            lines.append("")
            for key, value in profile.quality_summary.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
        if profile.sections:
            lines.append("### Sections")
            lines.append("")
            for section in profile.sections[:12]:
                indent = "  " * max(0, section.level - 1)
                lines.append(
                    f"- {indent}{' > '.join(section.path)} "
                    f"({section.start_offset}-{section.end_offset}, "
                    f"confidence {section.confidence_score:.2f})"
                )
            if len(profile.sections) > 12:
                lines.append(f"- ... {len(profile.sections) - 12} more sections")
            lines.append("")
        if profile.tables:
            lines.append("### Tables")
            lines.append("")
            for table in profile.tables[:5]:
                lines.append(
                    f"- {table.table_id} ({table.table_kind}, "
                    f"confidence {table.confidence_score:.2f})"
                )
            lines.append("")
        if profile.warnings:
            lines.append("### Warnings")
            lines.append("")
            for warning in profile.warnings[:10]:
                lines.append(f"- {warning.severity}: {warning.code} - {warning.message}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_warnings_markdown(batch: DocumentIntelligenceBatch) -> str:
    lines = ["# Document Intelligence Warnings", ""]
    warnings = list(batch.warnings)
    for profile in batch.profiles:
        warnings.extend(profile.warnings)
    if not warnings:
        lines.append("No document intelligence warnings.")
        return "\n".join(lines).rstrip() + "\n"
    for warning in warnings:
        source = f" ({warning.source_id})" if warning.source_id else ""
        loc = f" at {warning.location}" if warning.location else ""
        lines.append(f"- {warning.severity}: {warning.code}{source}{loc} - {warning.message}")
    return "\n".join(lines).rstrip() + "\n"


def write_document_intelligence_artifacts(
    thread_dir: Path, batch: DocumentIntelligenceBatch
) -> list[str]:
    thread_dir.mkdir(parents=True, exist_ok=True)
    _write_json(thread_dir / DOCUMENT_PROFILE_JSON, model_to_plain(batch))
    (thread_dir / DOCUMENT_PROFILE_MD).write_text(render_profiles_markdown(batch), encoding="utf-8")
    with (thread_dir / DOCUMENT_CHUNKS_JSONL).open("w", encoding="utf-8") as f:
        for profile in batch.profiles:
            for chunk in profile.chunks:
                f.write(json.dumps(model_to_plain(chunk), ensure_ascii=False) + "\n")
    _write_json(
        thread_dir / DOCUMENT_TABLES_JSON,
        {
            "thread_id": batch.thread_id,
            "tables": [
                model_to_plain(table) for profile in batch.profiles for table in profile.tables
            ],
        },
    )
    _write_json(
        thread_dir / DOCUMENT_CITATIONS_JSON,
        {
            "thread_id": batch.thread_id,
            "citations": [
                model_to_plain(citation)
                for profile in batch.profiles
                for citation in profile.citations
            ],
            "footnotes": [
                model_to_plain(footnote)
                for profile in batch.profiles
                for footnote in profile.footnotes
            ],
        },
    )
    (thread_dir / DOCUMENT_WARNINGS_MD).write_text(
        render_warnings_markdown(batch), encoding="utf-8"
    )
    return list(DOCUMENT_INTELLIGENCE_ARTIFACTS)
