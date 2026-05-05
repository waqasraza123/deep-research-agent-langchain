from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .contracts import DocumentChunk, DocumentSection, DocumentTable
from .normalizer import content_hash


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 3200
    overlap_chars: int = 300
    context_window_chars: int = 320


DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}(?:-\d{2}-\d{2})?\b|"
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)
NUMBER_RE = re.compile(
    r"(?<!\w)(?:[$€£]\s?)?\d[\d,]*(?:\.\d+)?\s?(?:%|percent|USD|users|requests|tokens|ms|s)?\b",
    re.I,
)
ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&.'-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9&.'-]+|[A-Z]{2,})){0,3}\b"
)


def _chunk_id(source_id: str, section_id: str | None, start: int, text: str) -> str:
    digest = hashlib.sha1(
        f"{source_id}|{section_id}|{start}|{content_hash(text)}".encode("utf-8")
    ).hexdigest()[:14]
    return f"{source_id}-chk-{digest}"


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _find_split(text: str, lo: int, hi: int) -> int:
    window = text[lo:hi]
    for pattern in (r"\n\n", r"\.\s+", r";\s+", r",\s+"):
        matches = list(re.finditer(pattern, window))
        if matches:
            pos = lo + matches[-1].end()
            if pos > lo:
                return pos
    return hi


def _adjust_for_tables(
    start: int, end: int, table_ranges: list[tuple[int, int]]
) -> tuple[int, int]:
    for t_start, t_end in table_ranges:
        if start < t_start < end < t_end:
            end = t_start
        elif t_start < start < t_end < end:
            start = t_end
        elif start <= t_start and t_end <= end:
            continue
    return start, end


def _simple_entities(text: str) -> list[str]:
    out: list[str] = []
    for match in ENTITY_RE.finditer(text):
        value = match.group(0).strip()
        if len(value) < 3 or value.lower() in {"the", "and", "for"}:
            continue
        if value not in out:
            out.append(value)
        if len(out) >= 24:
            break
    return out


def _simple_matches(pattern: re.Pattern[str], text: str, *, limit: int = 24) -> list[str]:
    out: list[str] = []
    for match in pattern.finditer(text):
        value = match.group(0).strip()
        if value and value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def chunk_document(
    text: str,
    *,
    source_id: str,
    sections: list[DocumentSection],
    tables: list[DocumentTable],
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    cfg = config or ChunkingConfig()
    table_ranges = [(table.start_offset, table.end_offset) for table in tables]
    chunks: list[DocumentChunk] = []
    ordinal = 0
    section_iter = sections or [
        DocumentSection(
            section_id=None,  # type: ignore[arg-type]
            heading="Document",
            level=1,
            start_offset=0,
            end_offset=len(text),
            text=text,
            path=["Document"],
            confidence_score=0.3,
        )
    ]

    for section in section_iter:
        section_start = max(0, section.start_offset)
        section_end = min(len(text), section.end_offset)
        cursor = section_start
        while cursor < section_end:
            desired_end = min(section_end, cursor + cfg.max_chars)
            if desired_end < section_end:
                desired_end = _find_split(text, cursor + max(400, cfg.max_chars // 3), desired_end)
            start, end = _adjust_for_tables(cursor, desired_end, table_ranges)
            if end <= start:
                end = min(section_end, cursor + cfg.max_chars)
            chunk_text = text[start:end].strip()
            if chunk_text:
                ordinal += 1
                related_tables = [
                    table.table_id
                    for table in tables
                    if table.start_offset < end and table.end_offset > start
                ]
                chunks.append(
                    DocumentChunk(
                        chunk_id=_chunk_id(source_id, section.section_id, start, chunk_text),
                        source_id=source_id,
                        section_id=section.section_id,
                        heading_path=section.path,
                        text=chunk_text,
                        start_offset=start,
                        end_offset=end,
                        content_hash=content_hash(chunk_text),
                        ordinal=ordinal,
                        approx_tokens=_approx_tokens(chunk_text),
                        context_before=text[
                            max(0, start - cfg.context_window_chars) : start
                        ].strip(),
                        context_after=text[
                            end : min(len(text), end + cfg.context_window_chars)
                        ].strip(),
                        detected_entities=_simple_entities(chunk_text),
                        detected_numbers=_simple_matches(NUMBER_RE, chunk_text),
                        detected_dates=_simple_matches(DATE_RE, chunk_text),
                        table_ids=related_tables,
                    )
                )
            if end >= section_end:
                break
            cursor = max(end - cfg.overlap_chars, cursor + 1)

    return chunks
