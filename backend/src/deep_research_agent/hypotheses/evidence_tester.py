from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from deep_research_agent.evidence.citation_mapper import SourceDocument, content_terms
from deep_research_agent.evidence.claim_extractor import (
    ClaimInput,
    extract_claims,
    extract_values,
    normalize_claim_text,
    split_sentences,
)
from deep_research_agent.evidence.ledger import load_source_documents

from .contracts import (
    HypothesisEvidence,
    HypothesisSet,
    HypothesisStatus,
    HypothesisTestResult,
    ResearchHypothesis,
)
from .generator import HypothesisBuildInput

_NEGATION_RE = re.compile(
    (
        r"\b(no|not|never|without|cannot|can't|does not|do not|is not|are not|"
        r"unsupported|incompatible)\b"
    ),
    re.IGNORECASE,
)
_POSITIVE_RE = re.compile(
    r"\b(supports|supported|allows|enables|can|is compatible|stronger|better|recommended)\b",
    re.IGNORECASE,
)
_OPPOSITES = (
    ("better", "worse"),
    ("stronger", "weaker"),
    ("more", "less"),
    ("higher", "lower"),
    ("faster", "slower"),
    ("supported", "unsupported"),
    ("compatible", "incompatible"),
    ("increase", "decrease"),
    ("improves", "worsens"),
)


def test_hypotheses(
    hypothesis_set: HypothesisSet,
    build_input: HypothesisBuildInput,
    run_dir: Path,
) -> HypothesisSet:
    source_docs = load_source_documents(run_dir)
    source_audit = _source_audit_by_id(build_input.source_audit)
    claims = _load_claims(build_input)
    evidence_by_id: dict[str, HypothesisEvidence] = {}
    results: list[HypothesisTestResult] = []

    for hypothesis in hypothesis_set.hypotheses:
        result = test_hypothesis(
            hypothesis,
            source_docs=source_docs,
            claims=claims,
            source_audit=source_audit,
        )
        results.append(result)
        hypothesis.status = result.status
        ids: list[str] = []
        for evidence in [
            *result.supporting_evidence,
            *result.opposing_evidence,
            *result.neutral_evidence,
        ]:
            evidence_by_id[evidence.evidence_id] = evidence
            ids.append(evidence.evidence_id)
        hypothesis.evidence_ids = sorted(dict.fromkeys(ids))
        hypothesis.unresolved_questions = result.unresolved

    hypothesis_set.test_results = results
    return hypothesis_set


def test_hypothesis(
    hypothesis: ResearchHypothesis,
    *,
    source_docs: list[SourceDocument],
    claims: list[dict[str, Any]],
    source_audit: dict[str, dict[str, Any]],
) -> HypothesisTestResult:
    support: list[HypothesisEvidence] = []
    oppose: list[HypothesisEvidence] = []
    neutral: list[HypothesisEvidence] = []

    for claim in claims[:300]:
        evidence = _score_text(
            hypothesis,
            str(claim.get("text") or ""),
            source_id=_first_source_id(claim),
            claim_id=str(claim.get("claim_id") or ""),
            artifact_path="evidence_ledger.json",
            source_audit=source_audit,
        )
        if evidence is None:
            continue
        _append_by_stance(evidence, support, oppose, neutral)

    for doc in source_docs:
        for sentence in split_sentences(doc.text)[:260]:
            evidence = _score_text(
                hypothesis,
                sentence,
                source_id=doc.source.source_id,
                artifact_path=doc.source.local_path,
                url=doc.source.final_url or doc.source.url,
                title=doc.source.title,
                fallback_quality=doc.source.quality_score,
                source_audit=source_audit,
            )
            if evidence is None:
                continue
            _append_by_stance(evidence, support, oppose, neutral)

    support = _dedupe_evidence(support)[:6]
    oppose = _dedupe_evidence(oppose)[:6]
    neutral = _dedupe_evidence(neutral)[:4]

    support_score = _aggregate_score(support)
    opposition_score = _aggregate_score(oppose)
    net = max(0.0, min(1.0, support_score - (opposition_score * 0.72)))
    supporting_source_ids = sorted({ev.source_id for ev in support if ev.source_id})
    opposing_source_ids = sorted({ev.source_id for ev in oppose if ev.source_id})
    all_source_ids = sorted({*supporting_source_ids, *opposing_source_ids})
    primary_count = len({ev.source_id for ev in support if ev.source_id and ev.primary_source})
    citation_count = len(
        {
            ev.source_id
            for ev in support
            if ev.source_id and (ev.citation_readiness or 0) >= 0.65
        }
    )
    contradiction_ids = _claim_contradictions(claims, support, oppose)
    status, warnings, unresolved = _status_for_result(
        support=support,
        oppose=oppose,
        support_score=support_score,
        opposition_score=opposition_score,
        source_docs=source_docs,
        contradiction_ids=contradiction_ids,
    )

    return HypothesisTestResult(
        result_id=_result_id(hypothesis.hypothesis_id),
        hypothesis_id=hypothesis.hypothesis_id,
        status=status,
        support_score=round(support_score, 3),
        opposition_score=round(opposition_score, 3),
        net_evidence_score=round(net, 3),
        supporting_evidence=support,
        opposing_evidence=oppose,
        neutral_evidence=neutral,
        supporting_source_ids=supporting_source_ids,
        opposing_source_ids=opposing_source_ids,
        source_diversity=len(all_source_ids),
        primary_source_count=primary_count,
        citation_ready_count=citation_count,
        contradiction_ids=contradiction_ids,
        unresolved=unresolved,
        warnings=warnings,
    )


def evidence_by_id(hypothesis_set: HypothesisSet) -> dict[str, HypothesisEvidence]:
    out: dict[str, HypothesisEvidence] = {}
    for result in hypothesis_set.test_results:
        for item in [
            *result.supporting_evidence,
            *result.opposing_evidence,
            *result.neutral_evidence,
        ]:
            out[item.evidence_id] = item
    return out


def _score_text(
    hypothesis: ResearchHypothesis,
    text: str,
    *,
    source_id: str | None = None,
    claim_id: str | None = None,
    artifact_path: str | None = None,
    url: str | None = None,
    title: str | None = None,
    fallback_quality: float | None = None,
    source_audit: dict[str, dict[str, Any]],
) -> HypothesisEvidence | None:
    clean = " ".join(text.split())
    if len(clean) < 24:
        return None
    hyp_norm = hypothesis.normalized_text
    text_norm = normalize_claim_text(clean)
    hyp_terms = content_terms(hyp_norm)
    text_terms = content_terms(text_norm)
    values = {value.lower(): value for value in extract_values(hypothesis.text)}
    value_matches = [original for normalized, original in values.items() if normalized in text_norm]
    overlap_terms = sorted(hyp_terms & text_terms)
    phrase_score = _phrase_score(hyp_norm, text_norm)
    keyword_score = len(overlap_terms) / max(len(hyp_terms), 1)
    entity_matches = sorted(_entities(hypothesis.text) & _entities(clean))
    entity_score = len(entity_matches) / max(len(_entities(hypothesis.text)), 1)
    value_score = len(value_matches) / max(len(values), 1) if values else 0.0
    audit = source_audit.get(source_id or "", {})
    quality = _float_or_none(audit.get("final_source_score")) or fallback_quality
    citation = _nested_score(audit, "citation_readiness_score")
    primary = (_nested_score(audit, "primary_source_likelihood") or 0.0) >= 0.58
    freshness_status = _nested_value(audit, "freshness_score", "status")
    quality_score = quality if quality is not None else 0.45
    citation_score = citation if citation is not None else 0.35

    score = min(
        1.0,
        (phrase_score * 0.30)
        + (keyword_score * 0.32)
        + (entity_score * 0.16)
        + (value_score * 0.14)
        + (quality_score * 0.05)
        + (citation_score * 0.03),
    )
    if value_matches and keyword_score >= 0.18:
        score = min(1.0, score + 0.08)
    if phrase_score >= 0.72:
        score = max(score, 0.68)
    if score < 0.18:
        return None

    stance = _stance(hyp_norm, text_norm, score)
    if stance == "neutral" and score < 0.34:
        return None

    signals = []
    if phrase_score >= 0.35:
        signals.append(f"exact phrase overlap {phrase_score:.2f}")
    if keyword_score >= 0.25:
        signals.append(f"keyword overlap {keyword_score:.2f}")
    if entity_matches:
        signals.append("entity overlap")
    if value_matches:
        signals.append("numeric/date overlap")
    if quality is not None:
        signals.append(f"source quality {quality:.2f}")
    if citation is not None:
        signals.append(f"citation readiness {citation:.2f}")
    if primary:
        signals.append("primary source signal")
    if freshness_status:
        signals.append(f"freshness {freshness_status}")

    return HypothesisEvidence(
        evidence_id=_evidence_id(hypothesis.hypothesis_id, source_id, claim_id, clean),
        hypothesis_id=hypothesis.hypothesis_id,
        source_id=source_id,
        claim_id=claim_id,
        artifact_path=artifact_path,
        url=url or str(audit.get("url") or ""),
        title=title or audit.get("title"),
        stance=stance,
        score=round(score, 3),
        matched_text=clean[:900],
        overlap_terms=overlap_terms[:18],
        matched_entities=entity_matches[:12],
        matched_values=value_matches,
        signals=signals,
        source_quality=quality,
        citation_readiness=citation,
        primary_source=primary,
        freshness_status=freshness_status,
    )


def _stance(hyp_norm: str, text_norm: str, score: float) -> str:
    if _opposes(hyp_norm, text_norm):
        return "oppose"
    if score >= 0.34:
        return "support"
    return "neutral"


def _opposes(left: str, right: str) -> bool:
    if _NEGATION_RE.search(left) != _NEGATION_RE.search(right) and _predicate_overlap(left, right):
        return True
    if _POSITIVE_RE.search(left) and _NEGATION_RE.search(right) and _predicate_overlap(left, right):
        return True
    for positive, negative in _OPPOSITES:
        if _contains(left, positive) and _contains(right, negative):
            return True
        if _contains(left, negative) and _contains(right, positive):
            return True
    return False


def _predicate_overlap(left: str, right: str) -> bool:
    left_terms = content_terms(left)
    right_terms = content_terms(right)
    if not left_terms or not right_terms:
        return False
    return len(left_terms & right_terms) / max(min(len(left_terms), len(right_terms)), 1) >= 0.38


def _phrase_score(left: str, right: str) -> float:
    words = [term for term in re.findall(r"\b[a-z0-9'-]+\b", left) if len(term) >= 4]
    if len(words) < 3:
        return 0.0
    best = 0
    for start in range(len(words)):
        for end in range(start + 3, min(len(words), start + 10) + 1):
            phrase = " ".join(words[start:end])
            if phrase in right:
                best = max(best, end - start)
    return min(1.0, best / len(words))


def _aggregate_score(items: list[HypothesisEvidence]) -> float:
    if not items:
        return 0.0
    top = sorted((item.score for item in items), reverse=True)[:4]
    score = top[0] if top else 0.0
    if len(top) > 1:
        score += sum(top[1:]) * 0.18
    return max(0.0, min(1.0, score))


def _status_for_result(
    *,
    support: list[HypothesisEvidence],
    oppose: list[HypothesisEvidence],
    support_score: float,
    opposition_score: float,
    source_docs: list[SourceDocument],
    contradiction_ids: list[str],
) -> tuple[HypothesisStatus, list[str], list[str]]:
    warnings: list[str] = []
    unresolved: list[str] = []
    if not source_docs:
        warnings.append("No source documents were available for hypothesis testing.")
        unresolved.append(
            "Add source documents or an evidence ledger before relying on this hypothesis."
        )
        return HypothesisStatus.NEEDS_MORE_EVIDENCE, warnings, unresolved
    if contradiction_ids:
        warnings.append("Evidence ledger contains contradiction groups linked to this hypothesis.")
    if opposition_score >= 0.42 and opposition_score >= support_score * 0.72:
        return HypothesisStatus.CONTRADICTED, warnings, unresolved
    if not support and not oppose:
        unresolved.append("No deterministic evidence matched this hypothesis.")
        return HypothesisStatus.UNSUPPORTED, warnings, unresolved
    if support_score >= 0.68 and len({ev.source_id for ev in support if ev.source_id}) >= 2:
        return HypothesisStatus.SUPPORTED, warnings, unresolved
    if support_score >= 0.42:
        if len({ev.source_id for ev in support if ev.source_id}) < 2:
            unresolved.append("Only one supporting source matched the hypothesis.")
        return HypothesisStatus.PARTIALLY_SUPPORTED, warnings, unresolved
    if support_score > 0:
        unresolved.append("Evidence matched weakly and needs corroboration.")
        return HypothesisStatus.INCONCLUSIVE, warnings, unresolved
    return HypothesisStatus.UNSUPPORTED, warnings, unresolved


def _load_claims(build_input: HypothesisBuildInput) -> list[dict[str, Any]]:
    ledger = build_input.evidence_ledger or {}
    claims = ledger.get("claims") if isinstance(ledger, dict) else None
    if isinstance(claims, list):
        return [item for item in claims if isinstance(item, dict)]
    extracted = extract_claims(
        [
            ClaimInput(origin="notes", text=build_input.notes_text, origin_ref="notes.md"),
            ClaimInput(origin="report", text=build_input.report_text, origin_ref="report.md"),
        ]
    )
    return [
        {
            "claim_id": claim.claim_id,
            "text": claim.text,
            "source_ids": claim.source_ids,
            "contradiction_ids": claim.contradiction_ids,
        }
        for claim in extracted
    ]


def _claim_contradictions(
    claims: list[dict[str, Any]],
    support: list[HypothesisEvidence],
    oppose: list[HypothesisEvidence],
) -> list[str]:
    claim_ids = {ev.claim_id for ev in [*support, *oppose] if ev.claim_id}
    out: list[str] = []
    for claim in claims:
        if claim.get("claim_id") not in claim_ids:
            continue
        for cid in claim.get("contradiction_ids") or []:
            if cid not in out:
                out.append(str(cid))
    return out


def _source_audit_by_id(source_audit: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not source_audit:
        return {}
    audits = source_audit.get("audits")
    if not isinstance(audits, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in audits:
        if isinstance(item, dict) and item.get("source_id"):
            out[str(item["source_id"])] = item
    return out


def _append_by_stance(
    evidence: HypothesisEvidence,
    support: list[HypothesisEvidence],
    oppose: list[HypothesisEvidence],
    neutral: list[HypothesisEvidence],
) -> None:
    if evidence.stance == "support":
        support.append(evidence)
    elif evidence.stance == "oppose":
        oppose.append(evidence)
    else:
        neutral.append(evidence)


def _dedupe_evidence(items: list[HypothesisEvidence]) -> list[HypothesisEvidence]:
    seen: set[tuple[str | None, str]] = set()
    out: list[HypothesisEvidence] = []
    for item in sorted(items, key=lambda ev: ev.score, reverse=True):
        key = (item.source_id or item.claim_id, normalize_claim_text(item.matched_text)[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _first_source_id(claim: dict[str, Any]) -> str | None:
    source_ids = claim.get("source_ids")
    if isinstance(source_ids, list) and source_ids:
        return str(source_ids[0])
    citations = claim.get("citations")
    if isinstance(citations, list) and citations and isinstance(citations[0], dict):
        return str(citations[0].get("source_id") or "") or None
    return None


def _nested_score(item: dict[str, Any], key: str) -> float | None:
    value = item.get(key)
    if isinstance(value, dict):
        return _float_or_none(value.get("score") or value.get("likelihood"))
    return _float_or_none(value)


def _nested_value(item: dict[str, Any], key: str, nested_key: str) -> str | None:
    value = item.get(key)
    if isinstance(value, dict) and value.get(nested_key) is not None:
        return str(value[nested_key])
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return max(0.0, min(float(value), 1.0))
    except Exception:
        return None


def _entities(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", text))


def _contains(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", text))


def _evidence_id(
    hypothesis_id: str, source_id: str | None, claim_id: str | None, text: str
) -> str:
    digest = hashlib.sha1(
        f"{hypothesis_id}:{source_id or ''}:{claim_id or ''}:{text[:220]}".encode("utf-8")
    ).hexdigest()[:12]
    return f"HE-{digest}"


def _result_id(hypothesis_id: str) -> str:
    digest = hashlib.sha1(hypothesis_id.encode("utf-8")).hexdigest()[:10]
    return f"HT-{digest}"
