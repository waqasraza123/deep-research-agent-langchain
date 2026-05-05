from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from deep_research_agent.source_identity import document_identity_from_source

from .artifact_writer import (
    DOCUMENT_INTELLIGENCE_ARTIFACTS,
    write_document_intelligence_artifacts,
)
from .chunker import ChunkingConfig, chunk_document
from .citation_extractor import extract_citations
from .content_features import detect_content_features
from .contracts import (
    DocumentChunk,
    DocumentCitation,
    DocumentContentFeature,
    DocumentExtractionWarning,
    DocumentFootnote,
    DocumentIntelligenceBatch,
    DocumentMetadata,
    DocumentNormalizationResult,
    DocumentProfile,
    DocumentSection,
    DocumentTable,
    DocumentTableCell,
    model_to_plain,
)
from .metadata_extractor import metadata_from_source, source_basics
from .normalizer import normalize_text
from .sectioner import section_document
from .table_extractor import extract_tables

__all__ = [
    "DOCUMENT_INTELLIGENCE_ARTIFACTS",
    "ChunkingConfig",
    "DocumentChunk",
    "DocumentCitation",
    "DocumentContentFeature",
    "DocumentExtractionWarning",
    "DocumentFootnote",
    "DocumentIntelligenceBatch",
    "DocumentMetadata",
    "DocumentNormalizationResult",
    "DocumentProfile",
    "DocumentSection",
    "DocumentTable",
    "DocumentTableCell",
    "build_document_context_block",
    "build_document_intelligence_batch",
    "model_to_plain",
    "profile_document",
    "write_document_intelligence_artifacts",
]


def _language_hint(text: str) -> str | None:
    if not text.strip():
        return None
    ascii_letters = len(re.findall(r"[A-Za-z]", text))
    if ascii_letters / max(1, len(text)) > 0.45:
        return "en"
    return None


def _assign_table_sections(tables: list[DocumentTable], sections: list[DocumentSection]) -> None:
    for table in tables:
        containing = [
            section
            for section in sections
            if section.start_offset <= table.start_offset and section.end_offset >= table.end_offset
        ]
        if containing:
            containing.sort(
                key=lambda section: (section.level, -(section.end_offset - section.start_offset))
            )
            table.section_id = containing[-1].section_id


def _quality_summary(profile: DocumentProfile, normalized_text: str) -> dict[str, Any]:
    present = {feature.name for feature in profile.features if feature.present}
    return {
        "normalized_char_count": len(normalized_text),
        "section_count": len(profile.sections),
        "chunk_count": len(profile.chunks),
        "table_count": len(profile.tables),
        "citation_count": len(profile.citations),
        "warning_count": len(profile.warnings),
        "has_error_or_empty_extraction": "has_error_or_empty_extraction" in present,
        "structure_score": round(
            min(
                1.0,
                (0.25 if profile.sections else 0.0)
                + (0.25 if profile.chunks else 0.0)
                + (0.2 if profile.tables else 0.0)
                + (0.2 if profile.citations else 0.0)
                + (0.1 if normalized_text.strip() else 0.0),
            ),
            3,
        ),
    }


def profile_document(
    *,
    raw_text: str,
    source: dict[str, Any] | None = None,
    source_id: str | None = None,
    url: str = "",
    title: str | None = None,
    source_type: str = "unknown",
    chunking: ChunkingConfig | None = None,
) -> DocumentProfile:
    source = dict(source or {})
    if source_id:
        source["source_id"] = source_id
    if url:
        source["url"] = url
    if title:
        source["title"] = title
    if source_type:
        source["document_kind"] = source_type

    resolved_source_id, resolved_url, resolved_title, domain, resolved_type = source_basics(source)
    normalization = normalize_text(
        raw_text,
        source_id=resolved_source_id,
        source_type=resolved_type,
    )
    normalized_text = normalization.normalized_text
    document_identity = document_identity_from_source(
        source,
        content_hash=normalization.content_hash,
    )
    metadata = metadata_from_source(source, raw_text=raw_text, normalized_text=normalized_text)
    sections = section_document(
        normalized_text,
        source_id=resolved_source_id,
        html_headings=metadata.html_headings,
    )
    tables = extract_tables(
        normalized_text, source_id=resolved_source_id, source_type=resolved_type
    )
    _assign_table_sections(tables, sections)
    citations, footnotes = extract_citations(normalized_text, source_id=resolved_source_id)
    features = detect_content_features(
        normalized_text,
        has_tables=bool(tables),
        has_references=bool(citations or footnotes),
    )
    chunks = chunk_document(
        normalized_text,
        source_id=resolved_source_id,
        sections=sections,
        tables=tables,
        config=chunking,
        document_id=document_identity.document_id,
    )
    warnings = list(normalization.warnings)
    if len(normalized_text) < 80:
        warnings.append(
            DocumentExtractionWarning(
                code="short_normalized_text",
                message="Normalized document text is very short; retrieval quality may be weak.",
                source_id=resolved_source_id,
            )
        )
    profile = DocumentProfile(
        document_id=document_identity.document_id,
        document_identity=document_identity,
        source_id=resolved_source_id,
        url=resolved_url,
        title=resolved_title,
        domain=domain,
        source_type=resolved_type,
        language_hint=_language_hint(normalized_text),
        content_hash=normalization.content_hash,
        metadata=metadata,
        sections=sections,
        chunks=chunks,
        tables=tables,
        citations=citations,
        footnotes=footnotes,
        features=features,
        warnings=warnings,
    )
    profile.quality_summary = _quality_summary(profile, normalized_text)
    return profile


def _safe_source_path(thread_dir: Path, local_path: str | None) -> Path | None:
    if not local_path:
        return None
    rel = str(local_path)
    if "runs/" in rel:
        parts = rel.split("runs/", 1)[-1].split("/", 1)
        rel = parts[1] if len(parts) == 2 else ""
    if not rel or rel.startswith("/") or ".." in rel or "\\" in rel:
        return None
    candidate = (thread_dir / rel).resolve()
    root = thread_dir.resolve()
    if not str(candidate).startswith(str(root)):
        return None
    return candidate


def build_document_intelligence_batch(
    *,
    thread_dir: Path,
    sources: list[dict[str, Any]],
    thread_id: str | None = None,
    chunking: ChunkingConfig | None = None,
) -> DocumentIntelligenceBatch:
    profiles: list[DocumentProfile] = []
    warnings: list[DocumentExtractionWarning] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        if source.get("ok") is not True or source.get("skipped"):
            continue
        source_id, _, _, _, _ = source_basics(source)
        source_path = _safe_source_path(thread_dir, source.get("local_path"))
        if source_path is None or not source_path.exists() or source_path.is_dir():
            warnings.append(
                DocumentExtractionWarning(
                    code="missing_source_text",
                    message="Source metadata did not point to a readable local text artifact.",
                    source_id=source_id,
                    severity="error",
                )
            )
            continue
        raw_text = source_path.read_text(encoding="utf-8", errors="ignore")
        profiles.append(profile_document(raw_text=raw_text, source=source, chunking=chunking))
    return DocumentIntelligenceBatch(thread_id=thread_id, profiles=profiles, warnings=warnings)


def batch_from_sources_manifest(
    *,
    thread_dir: Path,
    manifest_path: Path,
    thread_id: str | None = None,
    chunking: ChunkingConfig | None = None,
) -> DocumentIntelligenceBatch:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        data = []
    sources = [item for item in data if isinstance(item, dict)]
    return build_document_intelligence_batch(
        thread_dir=thread_dir,
        sources=sources,
        thread_id=thread_id,
        chunking=chunking,
    )


def build_document_context_block(batch: DocumentIntelligenceBatch, *, max_chunks: int = 8) -> str:
    if not batch.profiles:
        return ""
    lines = ["Document intelligence context (derived from fetched source text):"]
    chunks: list[tuple[DocumentProfile, Any]] = []
    for profile in batch.profiles:
        present = [feature.name for feature in profile.features if feature.present]
        lines.append(
            f"- {profile.source_id}: {profile.title or profile.url} "
            f"({profile.source_type}, sections={len(profile.sections)}, "
            f"tables={len(profile.tables)}, citations={len(profile.citations)})"
        )
        if present:
            lines.append(f"  Features: {', '.join(present[:8])}")
        if profile.warnings:
            lines.append(
                "  Extraction warnings: " + "; ".join(w.message for w in profile.warnings[:3])
            )
        for table in profile.tables[:2]:
            preview = table.readable_text[:900].strip()
            if preview:
                lines.append(f"  Table {table.table_id}:\n{preview}")
        for chunk in profile.chunks:
            chunks.append((profile, chunk))
    chunks.sort(
        key=lambda item: (
            len(item[1].detected_numbers) + len(item[1].detected_dates),
            -item[1].approx_tokens,
        ),
        reverse=True,
    )
    if chunks:
        lines.append("")
        lines.append("Important source-local chunks:")
        for profile, chunk in chunks[:max_chunks]:
            heading = " > ".join(chunk.heading_path) or "Document"
            text = re.sub(r"\s+", " ", chunk.text).strip()
            if len(text) > 900:
                text = text[:900].rsplit(" ", 1)[0] + "..."
            lines.append(f"- {profile.source_id} / {heading}: {text}")
    return "\n".join(lines)
