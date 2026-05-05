from __future__ import annotations

import hashlib
import re

from .citation_mapper import content_terms
from .claim_extractor import extract_values
from .contracts import ContradictionGroup, ExtractedClaim

_NEGATION_RE = re.compile(
    r"\b(no|not|never|without|cannot|can't|does not|do not|did not|is not|are not)\b"
)
_SUPPORT_POSITIVE_RE = re.compile(r"\b(supports|supported|allows|enables|is compatible with|can)\b")
_SUPPORT_NEGATIVE_RE = re.compile(
    r"\b(does not support|do not support|unsupported|incompatible|cannot|can't)\b"
)
_OPEN_SOURCE_RE = re.compile(r"\b(open[- ]source|oss)\b")
_PROPRIETARY_RE = re.compile(r"\b(proprietary|closed[- ]source|commercial license)\b")
_COMPARATIVE_OPPOSITES = (
    ("higher", "lower"),
    ("more", "less"),
    ("more", "fewer"),
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("faster", "slower"),
    ("better", "worse"),
    ("larger", "smaller"),
    ("above", "below"),
)


def detect_contradictions(claims: list[ExtractedClaim]) -> list[ContradictionGroup]:
    groups: list[ContradictionGroup] = []
    eligible = [claim for claim in claims if len(content_terms(claim.normalized_text)) >= 3]

    for idx, left in enumerate(eligible):
        for right in eligible[idx + 1 :]:
            if left.claim_id == right.claim_id:
                continue
            if not _same_topic(left, right):
                continue
            contradiction = _detect_pair(left, right)
            if contradiction is None:
                continue
            groups.append(contradiction)

    return _dedupe_groups(groups)


def _detect_pair(left: ExtractedClaim, right: ExtractedClaim) -> ContradictionGroup | None:
    left_norm = left.normalized_text
    right_norm = right.normalized_text
    left_values = {v.lower() for v in extract_values(left.text)}
    right_values = {v.lower() for v in extract_values(right.text)}

    if (
        left_values
        and right_values
        and left_values != right_values
        and _value_claims_conflict(left, right)
    ):
        return _group(
            left,
            right,
            contradiction_type="conflicting_values",
            severity="high",
            explanation=(
                "Claims discuss the same topic but contain different numeric or date "
                "values."
            ),
            values=sorted(left_values | right_values),
        )

    if _has_yes_no_conflict(left_norm, right_norm):
        return _group(
            left,
            right,
            contradiction_type="yes_no_conflict",
            severity="high",
            explanation="One claim affirms support/availability while the other denies it.",
        )

    if _OPEN_SOURCE_RE.search(left_norm) and _PROPRIETARY_RE.search(right_norm):
        return _group(
            left,
            right,
            contradiction_type="licensing_conflict",
            severity="medium",
            explanation=(
                "One claim describes the subject as open source while another describes "
                "it as proprietary."
            ),
        )
    if _PROPRIETARY_RE.search(left_norm) and _OPEN_SOURCE_RE.search(right_norm):
        return _group(
            left,
            right,
            contradiction_type="licensing_conflict",
            severity="medium",
            explanation=(
                "One claim describes the subject as proprietary while another describes "
                "it as open source."
            ),
        )

    for positive, negative in _COMPARATIVE_OPPOSITES:
        if _contains_word(left_norm, positive) and _contains_word(right_norm, negative):
            return _group(
                left,
                right,
                contradiction_type="opposite_comparison",
                severity="medium",
                explanation=(
                    f"Claims use opposite comparative language: {positive!r} vs "
                    f"{negative!r}."
                ),
            )
        if _contains_word(left_norm, negative) and _contains_word(right_norm, positive):
            return _group(
                left,
                right,
                contradiction_type="opposite_comparison",
                severity="medium",
                explanation=(
                    f"Claims use opposite comparative language: {negative!r} vs "
                    f"{positive!r}."
                ),
            )

    if _NEGATION_RE.search(left_norm) != _NEGATION_RE.search(
        right_norm
    ) and _predicate_overlap(left, right):
        return _group(
            left,
            right,
            contradiction_type="negation_conflict",
            severity="medium",
            explanation="Claims share a predicate but one contains explicit negation.",
        )

    return None


def _same_topic(left: ExtractedClaim, right: ExtractedClaim) -> bool:
    left_terms = content_terms(left.normalized_text)
    right_terms = content_terms(right.normalized_text)
    if not left_terms or not right_terms:
        return False
    overlap = len(left_terms & right_terms) / max(min(len(left_terms), len(right_terms)), 1)
    if overlap >= 0.42:
        return True
    left_entities = _entities(left.text)
    right_entities = _entities(right.text)
    return bool(left_entities and left_entities & right_entities and overlap >= 0.25)


def _value_claims_conflict(left: ExtractedClaim, right: ExtractedClaim) -> bool:
    if left.claim_type not in {"numeric", "date_sensitive"} and right.claim_type not in {
        "numeric",
        "date_sensitive",
    }:
        return False
    return _predicate_overlap(left, right) or _same_topic(left, right)


def _predicate_overlap(left: ExtractedClaim, right: ExtractedClaim) -> bool:
    left_terms = _terms_without_values(left.normalized_text)
    right_terms = _terms_without_values(right.normalized_text)
    if not left_terms or not right_terms:
        return False
    return len(left_terms & right_terms) / max(min(len(left_terms), len(right_terms)), 1) >= 0.45


def _terms_without_values(text: str) -> set[str]:
    clean = text
    for value in extract_values(text):
        clean = clean.replace(value.lower(), " ")
    return content_terms(clean)


def _has_yes_no_conflict(left_norm: str, right_norm: str) -> bool:
    left_positive = bool(_SUPPORT_POSITIVE_RE.search(left_norm))
    left_negative = bool(_SUPPORT_NEGATIVE_RE.search(left_norm))
    right_positive = bool(_SUPPORT_POSITIVE_RE.search(right_norm))
    right_negative = bool(_SUPPORT_NEGATIVE_RE.search(right_norm))
    return (left_positive and right_negative) or (left_negative and right_positive)


def _contains_word(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", text))


def _entities(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", text))


def _group(
    left: ExtractedClaim,
    right: ExtractedClaim,
    *,
    contradiction_type: str,
    severity: str,
    explanation: str,
    values: list[str] | None = None,
) -> ContradictionGroup:
    key = "|".join(sorted([left.claim_id, right.claim_id]))
    digest = hashlib.sha1(f"{contradiction_type}:{key}".encode("utf-8")).hexdigest()[:10]
    return ContradictionGroup(
        contradiction_id=f"K-{digest}",
        claim_ids=[left.claim_id, right.claim_id],
        contradiction_type=contradiction_type,
        severity=severity,  # type: ignore[arg-type]
        explanation=explanation,
        values=values or [],
    )


def _dedupe_groups(groups: list[ContradictionGroup]) -> list[ContradictionGroup]:
    seen: set[str] = set()
    out: list[ContradictionGroup] = []
    for group in groups:
        key = "|".join(sorted(group.claim_ids)) + ":" + group.contradiction_type
        if key in seen:
            continue
        seen.add(key)
        out.append(group)
    return out
