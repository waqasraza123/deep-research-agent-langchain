from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from deep_research_agent.source_identity import (
    ChunkIdentity,
    document_identity_from_source,
    source_domain,
    source_identity_from_dict,
    stable_chunk_id,
)
from deep_research_agent.source_intelligence.dedupe import content_hash

from .contracts import RetrievalChunk, RetrievalDocument, RetrievalIndex
from .errors import RetrievalIndexError
from .lexical import tokenize

DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
    r"\d{4})\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"(?<!\w)(?:[$€£])?\d+(?:,\d{3})*(?:\.\d+)?%?(?!\w)")
ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&.\-]+(?:\s+[A-Z][A-Za-z0-9&.\-]+){0,4})\b"
)


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha1_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def extract_dates(text: str) -> list[str]:
    return sorted({match.group(0).strip() for match in DATE_RE.finditer(text or "")})


def extract_numbers(text: str) -> list[str]:
    return sorted({match.group(0).strip() for match in NUMBER_RE.finditer(text or "")})


def extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for match in ENTITY_RE.finditer(text or ""):
        value = " ".join(match.group(0).split())
        if len(value) < 3 or value.lower() in {"the", "this"}:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(value)
        if len(entities) >= 40:
            break
    return entities


def _read_json(path: Path) -> Any:
    try:
        if not path.exists() or path.is_dir():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _audit_maps(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    audit_payload = _read_json(run_dir / "source_audit.json")
    audits: dict[str, dict[str, Any]] = {}
    warnings: dict[str, list[str]] = {}
    if isinstance(audit_payload, dict):
        for item in audit_payload.get("audits", []):
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id") or "")
            if not source_id:
                continue
            audits[source_id] = item
            warning_msgs = []
            for warning in item.get("warnings", []) or []:
                if isinstance(warning, dict):
                    warning_msgs.append(str(warning.get("message") or warning.get("code") or ""))
                elif warning:
                    warning_msgs.append(str(warning))
            warnings[source_id] = [msg for msg in warning_msgs if msg]
    return audits, warnings


def _source_quality(source: dict[str, Any], audit: dict[str, Any] | None) -> float | None:
    if audit and audit.get("final_source_score") is not None:
        return _bounded_float(audit.get("final_source_score"))
    quality_score = source.get("final_quality_score")
    if quality_score is None and isinstance(source.get("quality_score"), dict):
        quality_score = source["quality_score"].get("final_quality_score")
    if quality_score is None:
        quality_score = source.get("priority_score")
    return _bounded_float(quality_score)


def _citation_score(audit: dict[str, Any] | None) -> float | None:
    if not audit:
        return None
    citation = audit.get("citation_readiness_score")
    if isinstance(citation, dict):
        return _bounded_float(citation.get("score"))
    return _bounded_float(citation)


def _bounded_float(value: Any) -> float | None:
    try:
        f = float(value)
    except Exception:
        return None
    return max(0.0, min(1.0, f))


def _source_role(audit: dict[str, Any] | None) -> str | None:
    if not audit:
        return None
    authority = audit.get("authority_score")
    if isinstance(authority, dict) and authority.get("source_role"):
        return str(authority.get("source_role"))
    return None


def _primary_likelihood(audit: dict[str, Any] | None) -> float | None:
    if not audit:
        return None
    primary = audit.get("primary_source_likelihood")
    if isinstance(primary, dict):
        return _bounded_float(primary.get("likelihood"))
    return None


def _freshness_status(audit: dict[str, Any] | None) -> str | None:
    if not audit:
        return None
    freshness = audit.get("freshness_score")
    if isinstance(freshness, dict):
        return freshness.get("status")
    return None


def _resolve_local_path(run_dir: Path, local_path: str | None) -> Path | None:
    if not local_path:
        return None
    rel = local_path
    marker = f"runs/{run_dir.name}/"
    if marker in rel:
        rel = rel.split(marker, 1)[-1]
    elif rel.startswith("runs/"):
        parts = rel.split("/", 2)
        rel = parts[2] if len(parts) >= 3 else ""
    if not rel or rel.startswith("/") or ".." in rel or "\\" in rel:
        return None
    path = (run_dir / rel).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError:
        return None
    return path


def _section_heading(line: str) -> str | None:
    clean = line.strip()
    if not clean:
        return None
    if clean.startswith("#"):
        return clean.lstrip("#").strip()[:140] or None
    if len(clean) <= 90 and not clean.endswith((".", ",", ";", ":")):
        words = clean.split()
        if 2 <= len(words) <= 12:
            uppercase_words = sum(1 for word in words if word[:1].isupper())
            if uppercase_words >= max(1, len(words) // 2):
                return clean
    return None


def _paragraph_spans(text: str) -> list[tuple[int, int, str, list[str]]]:
    spans: list[tuple[int, int, str, list[str]]] = []
    section_path: list[str] = []
    offset = 0
    blocks = re.split(r"(\n\s*\n)", text)
    cursor = 0
    for block in blocks:
        start = cursor
        cursor += len(block)
        if not block or not block.strip() or re.fullmatch(r"\n\s*\n", block):
            continue
        clean = block.strip()
        first_line = clean.splitlines()[0].strip()
        heading = _section_heading(first_line)
        if heading and len(clean) <= len(first_line) + 3:
            section_path = [heading]
            continue
        if heading and len(clean.splitlines()) > 1:
            section_path = [heading]
            clean = "\n".join(clean.splitlines()[1:]).strip()
            offset = block.find(clean)
            start = start + max(0, offset)
        if clean:
            spans.append((start, start + len(clean), clean, list(section_path)))
    return spans


def _chunk_source_text(
    *,
    document: RetrievalDocument,
    text: str,
    target_chars: int,
    overlap_chars: int,
) -> list[RetrievalChunk]:
    paragraph_spans = _paragraph_spans(text)
    chunks: list[RetrievalChunk] = []
    current_texts: list[str] = []
    current_start: int | None = None
    current_end = 0
    current_section: list[str] = []

    def flush() -> None:
        nonlocal current_texts, current_start, current_end, current_section
        if current_start is None or not current_texts:
            return
        chunk_text = "\n\n".join(part.strip() for part in current_texts if part.strip()).strip()
        if not chunk_text:
            current_texts = []
            current_start = None
            return
        digest = content_hash(chunk_text)
        ordinal = len(chunks) + 1
        chunk_id = stable_chunk_id(
            document_id=document.document_id,
            source_id=document.source_id,
            content_hash=digest,
            start_offset=current_start,
            end_offset=current_end,
            ordinal=ordinal,
        )
        chunks.append(
            RetrievalChunk(
                chunk_id=chunk_id,
                chunk_identity=ChunkIdentity(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    source_id=document.source_id,
                    content_hash=digest,
                    start_offset=current_start,
                    end_offset=current_end,
                    ordinal=ordinal,
                ),
                document_id=document.document_id,
                source_id=document.source_id,
                url=document.final_url or document.url,
                title=document.title,
                section_path=current_section,
                text=chunk_text,
                start_offset=current_start,
                end_offset=current_end,
                content_hash=digest,
                entities=extract_entities(chunk_text),
                dates=extract_dates(chunk_text),
                numbers=extract_numbers(chunk_text),
                source_quality_score=document.source_quality_score,
                citation_readiness_score=document.citation_readiness_score,
                token_count=len(tokenize(chunk_text)),
                domain=document.domain,
                source_role=document.source_role,
                freshness_status=document.freshness_status,
                warnings=list(document.warnings),
                metadata={"document_id": document.document_id},
            )
        )
        if overlap_chars > 0 and len(chunk_text) > overlap_chars:
            tail = chunk_text[-overlap_chars:]
            current_texts = [tail]
            current_start = max(current_start, current_end - len(tail))
        else:
            current_texts = []
            current_start = None
        current_end = 0

    for start, end, paragraph, section in paragraph_spans:
        if current_start is None:
            current_start = start
            current_section = section
        projected = sum(len(part) for part in current_texts) + len(paragraph) + 2
        if projected > target_chars and current_texts:
            flush()
            if current_start is None:
                current_start = start
                current_section = section
        if section:
            current_section = section
        current_texts.append(paragraph)
        current_end = end
        if len(paragraph) >= target_chars:
            flush()
    flush()
    return chunks


def build_retrieval_index(
    run_dir: Path,
    *,
    thread_id: str | None = None,
    chunk_chars: int = 1600,
    overlap_chars: int = 180,
) -> RetrievalIndex:
    manifest_path = run_dir / "sources.json"
    manifest = _read_json(manifest_path)
    if manifest is None:
        raise RetrievalIndexError("sources.json not found or invalid")
    if not isinstance(manifest, list):
        raise RetrievalIndexError("sources.json must contain a list")

    audits, audit_warnings = _audit_maps(run_dir)
    documents: list[RetrievalDocument] = []
    chunks: list[RetrievalChunk] = []
    warnings: list[str] = []

    for item in manifest:
        if not isinstance(item, dict):
            continue
        if item.get("ok") is not True or item.get("skipped") is True:
            continue
        path = _resolve_local_path(run_dir, item.get("local_path"))
        if path is None or not path.exists() or path.is_dir():
            warnings.append(f"Source text unavailable for {item.get('url') or 'unknown source'}.")
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            warnings.append(f"Could not read {path.name}: {type(exc).__name__}: {exc}")
            continue
        if not text.strip():
            warnings.append(f"Source text empty for {item.get('url') or path.name}.")
            continue

        identity = source_identity_from_dict({**item, "content_hash": item.get("content_hash")})
        source_id = identity.source_id
        text_hash = content_hash(text)
        document_identity = document_identity_from_source(item, content_hash=text_hash)
        audit = audits.get(source_id)
        url = str(item.get("url") or identity.url or "")
        final_url = item.get("final_url") or url
        domain = (
            item.get("source_domain")
            or source_domain(final_url)
            or urlsplit(final_url).hostname
        )
        doc = RetrievalDocument(
            document_id=document_identity.document_id,
            document_identity=document_identity,
            source_id=source_id,
            url=url,
            final_url=final_url,
            title=item.get("title") or identity.title,
            domain=domain,
            local_path=str(item.get("local_path") or ""),
            content_hash=str(item.get("content_hash") or text_hash),
            text_hash=text_hash,
            word_count=int(item.get("word_count") or len(tokenize(text))),
            char_count=int(item.get("char_count") or len(text)),
            source_quality_score=_source_quality(item, audit),
            citation_readiness_score=_citation_score(audit),
            primary_source_likelihood=_primary_likelihood(audit),
            source_role=_source_role(audit),
            freshness_status=_freshness_status(audit),
            fetched_at=item.get("fetched_at"),
            warnings=audit_warnings.get(source_id, []),
            metadata={
                "document_kind": item.get("document_kind"),
                "content_type": item.get("content_type"),
                "recommended_usage": audit.get("recommended_usage") if audit else None,
            },
        )
        documents.append(doc)
        chunks.extend(
            _chunk_source_text(
                document=doc,
                text=text,
                target_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )
        )

    index = RetrievalIndex(
        index_id=sha1_text(f"{thread_id or run_dir.name}|{len(documents)}|{len(chunks)}")[:16],
        thread_id=thread_id or run_dir.name,
        generated_at=now_iso_utc(),
        documents=documents,
        chunks=chunks,
        chunk_count=len(chunks),
        token_count=sum(chunk.token_count for chunk in chunks),
        warnings=warnings,
        metadata={"chunk_chars": chunk_chars, "overlap_chars": overlap_chars},
    )
    if not documents:
        index.warnings.append("No usable source documents were available for retrieval indexing.")
    return index
