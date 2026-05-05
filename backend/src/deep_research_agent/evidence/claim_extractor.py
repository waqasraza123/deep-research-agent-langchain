from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .contracts import ClaimType, ExtractedClaim

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|"
    "december|jan\\.|feb\\.|mar\\.|apr\\.|jun\\.|jul\\.|aug\\.|sep\\.|sept\\.|oct\\.|nov\\.|dec\\."
)
_DATE_RE = re.compile(
    rf"\b(?:20\d{{2}}|19\d{{2}}|{_MONTHS}|q[1-4]\s+20\d{{2}}|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}})\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])(?:[$€£])?\d+(?:[.,]\d+)*(?:\.\d+)?\s*(?:%|percent|x|k|m|bn|billion|million)?(?=$|\W)",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[(?:S\d+(?:,\s*S\d+)*)\]", re.IGNORECASE)

_COMPARATIVE_TERMS = {
    "more",
    "less",
    "fewer",
    "higher",
    "lower",
    "better",
    "worse",
    "faster",
    "slower",
    "largest",
    "smallest",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "outperform",
    "underperform",
    "compared",
    "versus",
    "vs",
}
_CAUSAL_TERMS = {
    "because",
    "caused",
    "causes",
    "due to",
    "leads to",
    "led to",
    "results in",
    "resulted in",
    "drives",
    "driven by",
    "therefore",
}
_RECOMMENDATION_TERMS = {
    "should",
    "must",
    "recommend",
    "recommended",
    "best",
    "avoid",
    "prefer",
    "use",
    "consider",
}
_BROAD_TERMS = {
    "always",
    "never",
    "all",
    "none",
    "everyone",
    "nobody",
    "clearly",
    "obviously",
    "undeniably",
    "proves",
    "guarantees",
}
_FACTUAL_VERBS = {
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "uses",
    "contains",
    "includes",
    "supports",
    "provides",
    "offers",
    "requires",
    "released",
    "announced",
    "completed",
    "finished",
}


@dataclass(frozen=True)
class ClaimInput:
    origin: str
    text: str
    origin_ref: str | None = None
    source_ids: tuple[str, ...] = ()


def normalize_claim_text(text: str) -> str:
    clean = _CITATION_RE.sub("", text)
    clean = re.sub(r"\[[^\]]+\]\([^)]+\)", "", clean)
    clean = re.sub(r"`([^`]+)`", r"\1", clean)
    clean = re.sub(r"[*_>#]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" -:\t\r\n")
    return clean.lower()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]{2,}\b", text)]


def extract_values(text: str) -> list[str]:
    values = [m.group(0).strip() for m in _NUMBER_RE.finditer(text)]
    values.extend(m.group(0).strip() for m in _DATE_RE.finditer(text))
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def has_number(text: str) -> bool:
    return bool(_NUMBER_RE.search(text))


def has_date(text: str) -> bool:
    return bool(_DATE_RE.search(text))


def _strip_markdown_noise(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line:
            continue
        if line.startswith("|") and line.endswith("|"):
            continue
        if re.match(r"^#{1,6}\s+", line):
            continue
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def split_sentences(text: str) -> list[str]:
    clean = _strip_markdown_noise(text)
    if not clean:
        return []
    candidates: list[str] = []
    for line in clean.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        candidates.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9$])", line))
    out: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip(" -\t\r\n")
        if not candidate:
            continue
        if len(candidate) > 420:
            out.extend(_split_long_sentence(candidate))
        else:
            out.append(candidate)
    return out


def _split_long_sentence(sentence: str) -> list[str]:
    pieces = re.split(r"\s+(?:;|--|\u2014|\band\b)\s+", sentence)
    out = [p.strip(" -\t\r\n") for p in pieces if 35 <= len(p.strip()) <= 420]
    return out or [sentence[:420].strip()]


def classify_claim(sentence: str) -> ClaimType | None:
    normalized = normalize_claim_text(sentence)
    tokens = set(tokenize(normalized))
    if has_number(sentence):
        return "numeric"
    if has_date(sentence):
        return "date_sensitive"
    if len(tokens) < 4:
        return None
    if any(term in normalized for term in _CAUSAL_TERMS):
        return "causal"
    if tokens & _COMPARATIVE_TERMS or re.search(r"\bthan\b|\bcompared with\b", normalized):
        return "comparative"
    if tokens & _RECOMMENDATION_TERMS:
        return "recommendation"
    if tokens & _BROAD_TERMS:
        return "unsupported_broad"
    if tokens & _FACTUAL_VERBS:
        return "factual"
    return None


def is_candidate_claim(sentence: str) -> bool:
    if len(sentence) < 28:
        return False
    if len(sentence.split()) < 5:
        return False
    if sentence.endswith(":"):
        return False
    return classify_claim(sentence) is not None


def _claim_id(origin: str, normalized: str, ordinal: int) -> str:
    digest = hashlib.sha1(f"{origin}:{ordinal}:{normalized}".encode("utf-8")).hexdigest()[:10]
    return f"C-{digest}"


def extract_claims(
    inputs: list[ClaimInput],
    *,
    max_source_claims_per_source: int = 40,
) -> list[ExtractedClaim]:
    claims: list[ExtractedClaim] = []
    seen: set[tuple[str, str]] = set()
    ordinal = 0

    for item in inputs:
        per_source_count = 0
        for sentence in split_sentences(item.text):
            if not is_candidate_claim(sentence):
                continue
            normalized = normalize_claim_text(sentence)
            key = (item.origin, normalized)
            if key in seen:
                continue
            if item.origin == "source":
                per_source_count += 1
                if per_source_count > max_source_claims_per_source:
                    break
            seen.add(key)
            ordinal += 1
            claim_type = classify_claim(sentence) or "factual"
            notes = _claim_notes(sentence, claim_type)
            claim = ExtractedClaim(
                claim_id=_claim_id(item.origin, normalized, ordinal),
                text=sentence,
                normalized_text=normalized,
                claim_type=claim_type,
                origin=item.origin,  # type: ignore[arg-type]
                origin_ref=item.origin_ref,
                source_ids=list(item.source_ids),
                support_level="source_backed" if item.origin == "source" else "unsupported",
                confidence_score=0.75 if item.origin == "source" else 0.0,
                needs_human_review=item.origin != "source",
                notes=notes,
            )
            claims.append(claim)

    return claims


def _claim_notes(sentence: str, claim_type: ClaimType) -> list[str]:
    notes: list[str] = [f"Detected as {claim_type.replace('_', ' ')} claim."]
    if has_number(sentence):
        notes.append("Contains numeric/statistical value.")
    if has_date(sentence):
        notes.append("Contains date-sensitive value.")
    if _CITATION_RE.search(sentence):
        notes.append("Contains explicit source marker.")
    return notes
