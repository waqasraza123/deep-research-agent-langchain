from __future__ import annotations

import re

from .contracts import DocumentContentFeature

FEATURE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "has_pricing": [
        re.compile(r"[$€£]\s?\d"),
        re.compile(r"\bpricing|price|subscription|per month|per year\b", re.I),
    ],
    "has_api_docs": [re.compile(r"\bAPI\b|endpoint|request body|response body|SDK", re.I)],
    "has_code_examples": [
        re.compile(r"```|\b(?:curl|pip install|npm install|import\s+\w+|def\s+\w+)\b")
    ],
    "has_legal_terms": [
        re.compile(
            r"\bterms of service|privacy policy|liability|indemn|governing law|GDPR|CCPA\b", re.I
        )
    ],
    "has_dates": [
        re.compile(r"\b(?:19|20)\d{2}\b"),
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
            re.I,
        ),
    ],
    "has_statistics": [re.compile(r"\b\d+(?:\.\d+)?\s?%|\b\d+(?:,\d{3})+(?:\.\d+)?\b")],
    "has_comparison": [
        re.compile(
            r"\bversus|vs\.?|compared with|comparison|better than|less than|more than\b", re.I
        )
    ],
    "has_references": [re.compile(r"\breferences|bibliography|doi:|https?://|\[\d+\]\s+", re.I)],
    "has_tables": [re.compile(r"\|.+\||\t|,\s*\w+,\s*\w+")],
    "has_marketing_language": [
        re.compile(
            r"\bbest-in-class|industry-leading|seamless|revolutionary|unlock|"
            r"transform your|world-class\b",
            re.I,
        )
    ],
    "has_tutorial_language": [
        re.compile(r"\bhow to|get started|quickstart|step \d+|tutorial|guide\b", re.I)
    ],
    "has_error_or_empty_extraction": [
        re.compile(
            r"\bfetch failed|unsupported content-type|no text was found|access denied|forbidden\b",
            re.I,
        )
    ],
}


def _evidence(text: str, pattern: re.Pattern[str], *, limit: int = 3) -> list[str]:
    out: list[str] = []
    for match in pattern.finditer(text):
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        if snippet not in out:
            out.append(snippet)
        if len(out) >= limit:
            break
    return out


def detect_content_features(
    text: str,
    *,
    has_tables: bool = False,
    has_references: bool = False,
) -> list[DocumentContentFeature]:
    features: list[DocumentContentFeature] = []
    for name, patterns in FEATURE_PATTERNS.items():
        evidence: list[str] = []
        for pattern in patterns:
            evidence.extend(_evidence(text, pattern, limit=2))
        present = bool(evidence)
        if name == "has_tables" and has_tables:
            present = True
            evidence = evidence or ["Structured table extraction produced at least one table."]
        if name == "has_references" and has_references:
            present = True
            evidence = evidence or ["Citation extraction produced at least one citation."]
        features.append(
            DocumentContentFeature(
                name=name,
                present=present,
                confidence_score=0.82 if present else 0.65,
                evidence=evidence[:4],
            )
        )
    return features
