from __future__ import annotations

import re

from .citation_mapper import content_terms
from .claim_extractor import extract_values
from .contracts import ClaimConfidence, EvidenceSource, ExtractedClaim, UnsupportedClaim

_STRONG_LANGUAGE_RE = re.compile(
    r"\b(always|never|proves|guarantees|undeniably|clearly|definitely|all|none|best|must)\b",
    re.IGNORECASE,
)
_FRESHNESS_RE = re.compile(
    r"\b(current|currently|latest|recent|today|now|as of|this year|newest)\b",
    re.IGNORECASE,
)


def score_claims(
    claims: list[ExtractedClaim],
    sources: list[EvidenceSource],
    *,
    citation_threshold: float = 0.34,
) -> tuple[list[ExtractedClaim], list[ClaimConfidence], list[UnsupportedClaim]]:
    source_quality = {source.source_id: source.quality_score for source in sources}
    generated_reference_terms = _reference_terms_by_origin(claims)
    confidences: list[ClaimConfidence] = []
    unsupported: list[UnsupportedClaim] = []

    for claim in claims:
        if claim.origin == "source":
            confidence = ClaimConfidence(
                claim_id=claim.claim_id,
                score=0.78,
                support_level="source_backed",
                factors=["Extracted directly from source text."],
            )
            claim.confidence_score = confidence.score
            claim.support_level = confidence.support_level
            claim.needs_human_review = False
            confidences.append(confidence)
            continue

        confidence = _score_generated_claim(
            claim,
            source_quality,
            generated_reference_terms,
            citation_threshold=citation_threshold,
        )
        claim.confidence_score = confidence.score
        claim.support_level = confidence.support_level
        claim.needs_human_review = confidence.support_level in {
            "weak",
            "unsupported",
            "contradicted",
        }
        claim.notes.extend(confidence.factors)
        claim.notes.extend(f"Penalty: {penalty}" for penalty in confidence.penalties)
        if confidence.freshness_warning:
            claim.notes.append(confidence.freshness_warning)
        confidences.append(confidence)

        unsupported_reason = unsupported_reason_for_claim(claim, confidence)
        if unsupported_reason:
            unsupported.append(
                UnsupportedClaim(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    origin=claim.origin,
                    support_level=claim.support_level,
                    reason=unsupported_reason,
                    needs_human_review=True,
                )
            )

    return claims, confidences, unsupported


def unsupported_reason_for_claim(claim: ExtractedClaim, confidence: ClaimConfidence) -> str | None:
    if claim.origin == "source":
        return None
    if confidence.support_level == "unsupported":
        return "No source citation candidate passed the support threshold."
    if confidence.support_level == "contradicted":
        return "Claim is part of a possible contradiction group."
    if _STRONG_LANGUAGE_RE.search(claim.text) and confidence.score < 0.72:
        return "Strong language is backed only by weak or partial evidence."
    if extract_values(claim.text):
        value_matches = {
            value.lower()
            for citation in claim.citations
            for value in citation.value_matches
            if citation.score >= 0.34
        }
        missing = [
            value for value in extract_values(claim.text) if value.lower() not in value_matches
        ]
        if missing:
            return (
                "Claim contains numeric/date values with no matching source evidence: "
                + ", ".join(missing)
            )
    if (
        claim.origin == "report"
        and claim.support_level == "weak"
        and any("not found in notes or source claims" in note for note in claim.notes)
    ):
        return (
            "Claim appears only in report text and lacks corroboration from notes or source "
            "claims."
        )
    return None


def _score_generated_claim(
    claim: ExtractedClaim,
    source_quality: dict[str, float],
    reference_terms_by_origin: dict[str, set[str]],
    *,
    citation_threshold: float,
) -> ClaimConfidence:
    factors: list[str] = []
    penalties: list[str] = []
    good_citations = [
        citation for citation in claim.citations if citation.score >= citation_threshold
    ]
    source_ids = {citation.source_id for citation in good_citations}

    score = 0.18
    if good_citations:
        citation_strength = max(citation.score for citation in good_citations)
        avg_quality = sum(
            source_quality.get(source_id, 0.5) for source_id in source_ids
        ) / len(source_ids)
        score += min(0.28, 0.10 * len(source_ids))
        score += citation_strength * 0.34
        score += avg_quality * 0.12
        factors.append(f"{len(source_ids)} supporting source(s).")
        factors.append(f"Best citation score {citation_strength:.2f}.")
    else:
        penalties.append("No citation candidate above threshold.")
        score -= 0.16

    values = extract_values(claim.text)
    if values:
        value_matches = {
            value.lower()
            for citation in good_citations
            for value in citation.value_matches
        }
        if all(value.lower() in value_matches for value in values):
            score += 0.08
            factors.append("Numeric/date values are present in supporting citations.")
        else:
            score -= 0.12
            penalties.append("Numeric/date values are missing from supporting citations.")

    if claim.contradiction_ids:
        score -= 0.30
        penalties.append("Possible contradiction detected.")

    if _STRONG_LANGUAGE_RE.search(claim.text) and len(good_citations) < 2:
        score -= 0.12
        penalties.append("Strong language has fewer than two supporting sources.")

    freshness_warning = None
    if claim.claim_type == "date_sensitive" or _FRESHNESS_RE.search(claim.text):
        score -= 0.04
        freshness_warning = (
            "Freshness warning: date-sensitive claim should be reviewed against current "
            "sources."
        )
        penalties.append("Date-sensitive or freshness-sensitive language.")

    if claim.origin == "report" and not _claim_has_reference_support(
        claim, reference_terms_by_origin
    ):
        score -= 0.08
        penalties.append("Report claim was not found in notes or source claims.")
        claim.notes.append("Report-only claim: not found in notes or source claims.")

    score = max(0.0, min(1.0, score))
    support_level = _support_level(
        score,
        bool(good_citations),
        bool(claim.contradiction_ids),
    )

    return ClaimConfidence(
        claim_id=claim.claim_id,
        score=round(score, 3),
        support_level=support_level,
        factors=factors,
        penalties=penalties,
        freshness_warning=freshness_warning,
    )


def _support_level(score: float, has_citation: bool, contradicted: bool) -> str:
    if contradicted:
        return "contradicted"
    if not has_citation:
        return "unsupported"
    if score >= 0.76:
        return "strong"
    if score >= 0.56:
        return "moderate"
    return "weak"


def _reference_terms_by_origin(claims: list[ExtractedClaim]) -> dict[str, set[str]]:
    notes_terms: set[str] = set()
    source_terms: set[str] = set()
    for claim in claims:
        if claim.origin == "notes":
            notes_terms |= content_terms(claim.normalized_text)
        elif claim.origin == "source":
            source_terms |= content_terms(claim.normalized_text)
    return {"notes": notes_terms, "source": source_terms}


def _claim_has_reference_support(
    claim: ExtractedClaim, reference_terms_by_origin: dict[str, set[str]]
) -> bool:
    claim_terms = content_terms(claim.normalized_text)
    if not claim_terms:
        return False
    reference_terms = reference_terms_by_origin["notes"] | reference_terms_by_origin["source"]
    return len(claim_terms & reference_terms) / max(len(claim_terms), 1) >= 0.45
