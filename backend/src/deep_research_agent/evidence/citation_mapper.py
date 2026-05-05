from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .claim_extractor import extract_values, normalize_claim_text, split_sentences, tokenize
from .contracts import ClaimCitation, EvidenceQuote, EvidenceSource, ExtractedClaim

_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "because",
    "before",
    "being",
    "between",
    "could",
    "does",
    "from",
    "have",
    "into",
    "more",
    "most",
    "only",
    "over",
    "should",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "through",
    "under",
    "using",
    "were",
    "with",
    "would",
}


@dataclass(frozen=True)
class SourceDocument:
    source: EvidenceSource
    text: str


def content_terms(text: str) -> set[str]:
    return {term for term in tokenize(text) if term not in _STOPWORDS and len(term) >= 4}


def map_claim_citations(
    claims: list[ExtractedClaim],
    sources: list[SourceDocument],
    *,
    threshold: float = 0.34,
    max_candidates: int = 3,
) -> tuple[dict[str, list[ClaimCitation]], list[EvidenceQuote]]:
    citation_map: dict[str, list[ClaimCitation]] = {}
    quotes: list[EvidenceQuote] = []

    for claim in claims:
        if claim.origin == "source" and claim.source_ids:
            citation = _self_citation(claim, sources)
            citation_map[claim.claim_id] = [citation] if citation else []
            if citation and citation.quote_id:
                quotes.append(
                    EvidenceQuote(
                        quote_id=citation.quote_id,
                        source_id=citation.source_id,
                        text=citation.matched_text,
                        score=citation.score,
                        reason=citation.reason,
                    )
                )
            continue

        candidates: list[ClaimCitation] = []
        for source in sources:
            candidates.extend(_score_source(claim, source))
        candidates.sort(key=lambda item: item.score, reverse=True)
        selected = [
            candidate for candidate in candidates if candidate.score >= threshold
        ][:max_candidates]
        citation_map[claim.claim_id] = selected
        for citation in selected:
            if citation.quote_id:
                quotes.append(
                    EvidenceQuote(
                        quote_id=citation.quote_id,
                        source_id=citation.source_id,
                        text=citation.matched_text,
                        score=citation.score,
                        reason=citation.reason,
                    )
                )

    return citation_map, _dedupe_quotes(quotes)


def _self_citation(claim: ExtractedClaim, sources: list[SourceDocument]) -> ClaimCitation | None:
    source_id = claim.source_ids[0]
    source = next((item for item in sources if item.source.source_id == source_id), None)
    if source is None:
        return None
    matched = claim.text[:500]
    quote_id = _quote_id(source_id, matched)
    return ClaimCitation(
        source_id=source_id,
        quote_id=quote_id,
        url=source.source.final_url or source.source.url,
        title=source.source.title,
        score=1.0,
        reason="Claim was extracted directly from this source.",
        matched_text=matched,
        overlap_terms=sorted(content_terms(claim.text))[:12],
        value_matches=extract_values(claim.text),
    )


def _score_source(claim: ExtractedClaim, source: SourceDocument) -> list[ClaimCitation]:
    if not source.text.strip():
        return []

    claim_norm = normalize_claim_text(claim.text)
    claim_terms = content_terms(claim_norm)
    claim_values = {value.lower(): value for value in extract_values(claim.text)}
    if not claim_terms and not claim_values:
        return []

    sentences = split_sentences(source.text)
    if not sentences:
        sentences = [source.text[:800]]

    title_domain_terms = content_terms(
        " ".join([source.source.title or "", source.source.domain or ""])
    )
    out: list[ClaimCitation] = []

    for sentence in sentences[:240]:
        sentence_norm = normalize_claim_text(sentence)
        sentence_terms = content_terms(sentence_norm)
        if not sentence_terms and not claim_values:
            continue

        overlap = sorted(claim_terms & sentence_terms)
        value_matches = [
            original for normalized, original in claim_values.items() if normalized in sentence_norm
        ]
        exact_phrase = _longest_exact_phrase_score(claim_norm, sentence_norm)
        keyword_score = len(overlap) / max(len(claim_terms), 1)
        entity_score = _entity_overlap_score(claim.text, sentence)
        value_score = len(value_matches) / max(len(claim_values), 1) if claim_values else 0.0
        title_hint = min(len(claim_terms & title_domain_terms) * 0.04, 0.12)

        score = min(
            1.0,
            (exact_phrase * 0.34)
            + (keyword_score * 0.30)
            + (entity_score * 0.18)
            + (value_score * 0.16)
            + title_hint,
        )
        if value_matches and keyword_score >= 0.18:
            score = min(1.0, score + 0.08)
        if exact_phrase >= 0.75:
            score = max(score, 0.72)

        if score <= 0.05:
            continue

        reasons = _score_reasons(
            exact_phrase=exact_phrase,
            keyword_score=keyword_score,
            entity_score=entity_score,
            value_matches=value_matches,
            title_hint=title_hint,
        )
        out.append(
            ClaimCitation(
                source_id=source.source.source_id,
                quote_id=_quote_id(source.source.source_id, sentence),
                url=source.source.final_url or source.source.url,
                title=source.source.title,
                score=round(score, 3),
                reason="; ".join(reasons),
                matched_text=sentence[:700],
                overlap_terms=overlap[:16],
                value_matches=value_matches,
            )
        )

    out.sort(key=lambda item: item.score, reverse=True)
    return out[:5]


def _longest_exact_phrase_score(claim_norm: str, sentence_norm: str) -> float:
    claim_words = [w for w in tokenize(claim_norm) if w not in _STOPWORDS]
    if len(claim_words) < 3:
        return 0.0

    best = 0
    for start in range(len(claim_words)):
        for end in range(start + 3, min(len(claim_words), start + 12) + 1):
            phrase = " ".join(claim_words[start:end])
            if phrase in sentence_norm:
                best = max(best, end - start)
    return min(1.0, best / max(len(claim_words), 1))


def _entity_overlap_score(left: str, right: str) -> float:
    left_entities = set(re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", left))
    right_entities = set(re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", right))
    if not left_entities:
        return 0.0
    return len(left_entities & right_entities) / len(left_entities)


def _score_reasons(
    *,
    exact_phrase: float,
    keyword_score: float,
    entity_score: float,
    value_matches: list[str],
    title_hint: float,
) -> list[str]:
    reasons: list[str] = []
    if exact_phrase >= 0.25:
        reasons.append("exact phrase overlap")
    if keyword_score >= 0.2:
        reasons.append("keyword overlap")
    if entity_score > 0:
        reasons.append("entity overlap")
    if value_matches:
        reasons.append("numeric/date value overlap")
    if title_hint:
        reasons.append("title/domain hint")
    return reasons or ["weak lexical overlap"]


def _quote_id(source_id: str, text: str) -> str:
    digest = hashlib.sha1(f"{source_id}:{text}".encode("utf-8")).hexdigest()[:10]
    return f"Q-{digest}"


def _dedupe_quotes(quotes: list[EvidenceQuote]) -> list[EvidenceQuote]:
    seen: set[str] = set()
    out: list[EvidenceQuote] = []
    for quote in quotes:
        if quote.quote_id in seen:
            continue
        seen.add(quote.quote_id)
        out.append(quote)
    return out
