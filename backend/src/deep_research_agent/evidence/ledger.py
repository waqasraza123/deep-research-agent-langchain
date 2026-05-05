from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from deep_research_agent.artifacts import now_iso_utc

from .citation_mapper import SourceDocument, map_claim_citations
from .claim_extractor import ClaimInput, extract_claims
from .confidence import score_claims
from .contracts import (
    EvidenceCoverageReport,
    EvidenceLedger,
    EvidenceSource,
    ExtractedClaim,
)
from .contradiction import detect_contradictions


def build_evidence_ledger(run_dir: Path, *, thread_id: str) -> EvidenceLedger:
    generated_at = now_iso_utc()
    notes_text = _read_text(run_dir / "notes.md")
    report_text = _read_text(run_dir / "report.md")
    source_documents = load_source_documents(run_dir)

    inputs = [
        ClaimInput(origin="notes", text=notes_text, origin_ref="notes.md"),
        ClaimInput(origin="report", text=report_text, origin_ref="report.md"),
    ]
    inputs.extend(
        ClaimInput(
            origin="source",
            text=doc.text,
            origin_ref=doc.source.source_id,
            source_ids=(doc.source.source_id,),
        )
        for doc in source_documents
    )

    claims = extract_claims(inputs)
    citation_map, quotes = map_claim_citations(claims, source_documents)

    for claim in claims:
        claim.citations = citation_map.get(claim.claim_id, [])
        if claim.origin != "source":
            claim.source_ids = sorted({citation.source_id for citation in claim.citations})

    contradictions = detect_contradictions(claims)
    _attach_contradictions(claims, contradictions)
    claims, _confidences, unsupported_claims = score_claims(
        claims,
        [doc.source for doc in source_documents],
    )
    coverage = build_coverage_report(
        generated_at=generated_at,
        sources=[doc.source for doc in source_documents],
        claims=claims,
        contradiction_count=len(contradictions),
    )

    ledger_id = "EL-" + hashlib.sha1(f"{thread_id}:{generated_at}".encode("utf-8")).hexdigest()[:12]
    return EvidenceLedger(
        ledger_id=ledger_id,
        thread_id=thread_id,
        generated_at=generated_at,
        sources=[doc.source for doc in source_documents],
        quotes=quotes,
        claims=claims,
        contradictions=contradictions,
        unsupported_claims=unsupported_claims,
        coverage=coverage,
        citation_map=citation_map,
        metadata={
            "notes_chars": len(notes_text),
            "report_chars": len(report_text),
            "source_documents": len(source_documents),
            "method": "deterministic_offline_heuristics",
        },
    )


def load_source_documents(run_dir: Path) -> list[SourceDocument]:
    sources_meta = _load_sources_metadata(run_dir)
    source_documents: list[SourceDocument] = []
    used_paths: set[Path] = set()

    for idx, meta in enumerate(sources_meta, start=1):
        source_id = str(meta.get("id") or meta.get("source_id") or f"S{idx}")
        source = _source_from_metadata(source_id, meta)
        text_path = _resolve_source_text_path(run_dir, source.local_path)
        text = _read_text(text_path) if text_path else ""
        if text_path:
            used_paths.add(text_path)
        source_documents.append(SourceDocument(source=source, text=text))

    next_idx = len(source_documents) + 1
    for meta_path in sorted((run_dir / "sources").glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        local_path = str(meta.get("local_path") or "")
        text_path = _resolve_source_text_path(run_dir, local_path)
        if text_path in used_paths:
            continue
        source_id = str(meta.get("id") or meta.get("source_id") or f"S{next_idx}")
        next_idx += 1
        source = _source_from_metadata(source_id, meta)
        text = _read_text(text_path) if text_path else ""
        if text_path:
            used_paths.add(text_path)
        source_documents.append(SourceDocument(source=source, text=text))

    if not source_documents:
        for txt_path in sorted((run_dir / "sources").glob("*.txt")):
            source_id = f"S{next_idx}"
            next_idx += 1
            source_documents.append(
                SourceDocument(
                    source=EvidenceSource(
                        source_id=source_id,
                        local_path=f"runs/{run_dir.name}/sources/{txt_path.name}",
                        domain=None,
                        quality_score=_quality_score({"ok": True}, _read_text(txt_path)),
                    ),
                    text=_read_text(txt_path),
                )
            )

    return source_documents


def build_coverage_report(
    *,
    generated_at: str,
    sources: list[EvidenceSource],
    claims: list[ExtractedClaim],
    contradiction_count: int,
) -> EvidenceCoverageReport:
    total = len(claims)
    generated_claims = len([claim for claim in claims if claim.origin != "source"])
    source_claims = total - generated_claims
    supported = len(
        [claim for claim in claims if claim.support_level in {"source_backed", "strong"}]
    )
    partial = len([claim for claim in claims if claim.support_level in {"moderate", "weak"}])
    unsupported = len([claim for claim in claims if claim.support_level == "unsupported"])
    contradicted = len([claim for claim in claims if claim.support_level == "contradicted"])
    average = sum(claim.confidence_score for claim in claims) / total if total else 0.0

    by_type: dict[str, int] = {}
    for claim in claims:
        by_type[claim.claim_type] = by_type.get(claim.claim_type, 0) + 1

    warnings: list[str] = []
    if not sources:
        warnings.append("No source documents were available for evidence mapping.")
    if unsupported:
        warnings.append(f"{unsupported} claim(s) require evidence review.")
    if contradiction_count:
        warnings.append(f"{contradiction_count} possible contradiction group(s) detected.")

    return EvidenceCoverageReport(
        generated_at=generated_at,
        source_count=len(sources),
        total_claims=total,
        generated_claims=generated_claims,
        source_claims=source_claims,
        supported_claims=supported,
        partially_supported_claims=partial,
        unsupported_claims=unsupported,
        contradicted_claims=contradicted,
        contradiction_count=contradiction_count,
        average_confidence=round(average, 3),
        by_claim_type=by_type,
        warnings=warnings,
    )


def _load_sources_metadata(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "sources.json"
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        items = raw.get("sources")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _source_from_metadata(source_id: str, meta: dict[str, Any]) -> EvidenceSource:
    url = _optional_str(meta.get("url"))
    final_url = _optional_str(meta.get("final_url"))
    title = _optional_str(meta.get("title"))
    local_path = _optional_str(meta.get("local_path"))
    domain = _domain(final_url or url)
    return EvidenceSource(
        source_id=source_id,
        url=url,
        final_url=final_url,
        title=title,
        domain=domain,
        local_path=local_path,
        ok=bool(meta.get("ok", True)),
        fetched_at=_optional_str(meta.get("fetched_at")),
        word_count=_optional_int(meta.get("word_count")),
        char_count=_optional_int(meta.get("char_count")),
        quality_score=_quality_score(meta, ""),
        metadata=meta,
    )


def _quality_score(meta: dict[str, Any], text: str) -> float:
    score = 0.35
    if meta.get("ok", True):
        score += 0.18
    if meta.get("title"):
        score += 0.08
    if meta.get("final_url") or meta.get("url"):
        score += 0.08
    word_count = _optional_int(meta.get("word_count")) or len(text.split())
    if word_count >= 1000:
        score += 0.18
    elif word_count >= 250:
        score += 0.10
    if meta.get("truncated"):
        score -= 0.06
    return round(max(0.0, min(1.0, score)), 3)


def _resolve_source_text_path(run_dir: Path, local_path: str | None) -> Path | None:
    if not local_path:
        return None
    if local_path.startswith("/") or ".." in local_path or "\\" in local_path:
        return None
    marker = f"runs/{run_dir.name}/"
    rel = local_path.split(marker, 1)[-1] if marker in local_path else local_path
    if rel.startswith("/") or ".." in rel or "\\" in rel:
        return None
    candidate = (run_dir / rel).resolve()
    if not str(candidate).startswith(str(run_dir.resolve())):
        return None
    return candidate


def _attach_contradictions(claims: list[ExtractedClaim], contradictions) -> None:
    by_id = {claim.claim_id: claim for claim in claims}
    for group in contradictions:
        for claim_id in group.claim_ids:
            claim = by_id.get(claim_id)
            if claim and group.contradiction_id not in claim.contradiction_ids:
                claim.contradiction_ids.append(group.contradiction_id)


def _read_text(path: Path | None, *, max_chars: int = 120_000) -> str:
    if path is None or not path.exists() or path.is_dir():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:max_chars]


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    host = urlparse(url).hostname
    return host.lower() if host else None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None
