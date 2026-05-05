from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any
from urllib.parse import urlsplit

from .contracts import RecommendedAction, RiskLevel, SourcePoisoningFinding

WORD_RE = re.compile(r"\b[a-z][a-z0-9\-]{2,}\b", flags=re.IGNORECASE)
CITATION_RE = re.compile(
    r"(?:\[[0-9]{1,3}\]|\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2}\)|"
    r"\bdoi\s*:\s*10\.[^\s]+)",
    flags=re.IGNORECASE,
)
ABSOLUTE_CLAIM_RE = re.compile(
    r"\b(?:always|never|guarantees?|proves?|undeniable|everyone|no\s+evidence\s+against|"
    r"100%|definitive(?:ly)?|must\s+be\s+true|cannot\s+be\s+false)\b",
    flags=re.IGNORECASE,
)
OFFICIAL_CLAIM_RE = re.compile(
    r"\b(?:official|authorized|government-approved|regulator-approved|primary)\s+"
    r"(?:source|website|publication|report|guidance|document)\b",
    flags=re.IGNORECASE,
)
HIDDEN_TEXT_RE = re.compile(
    r"\b(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|"
    r"color\s*:\s*white|hidden\s+keywords?|tiny\s+text)\b",
    flags=re.IGNORECASE,
)
FREE_HOSTS = (
    "blogspot.",
    "wordpress.",
    "medium.com",
    "substack.com",
    "github.io",
    "pages.dev",
    "vercel.app",
    "netlify.app",
    "wixsite.com",
    "weebly.com",
)
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "are",
    "was",
    "were",
    "not",
    "you",
    "your",
    "source",
    "article",
    "report",
}


def _finding_id(source_id: str, category: str, evidence: str) -> str:
    digest = hashlib.sha1(f"{source_id}|{category}|{evidence}".encode("utf-8")).hexdigest()
    return f"SP-{digest[:12]}"


def _domain(url: str | None) -> str:
    if not url:
        return ""
    return (urlsplit(url).hostname or "").lower()


def _related_domain(a: str, b: str) -> bool:
    if not a or not b:
        return True
    return a == b or a.endswith("." + b) or b.endswith("." + a)


def _excerpt(value: str, max_chars: int = 240) -> str:
    clean = " ".join((value or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."


def _finding(
    *,
    source_id: str,
    url: str,
    category: str,
    risk_level: RiskLevel,
    evidence: str,
    explanation: str,
    recommended_action: RecommendedAction = "allow_with_warning",
    metadata: dict[str, Any] | None = None,
) -> SourcePoisoningFinding:
    return SourcePoisoningFinding(
        finding_id=_finding_id(source_id, category, evidence),
        source_id=source_id,
        url=url,
        category=category,
        risk_level=risk_level,
        evidence=_excerpt(evidence),
        explanation=explanation,
        recommended_action=recommended_action,
        metadata=metadata or {},
    )


def detect_source_poisoning(
    text: str,
    *,
    source_id: str,
    url: str = "",
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[SourcePoisoningFinding]:
    metadata = metadata or {}
    source_text = text or ""
    findings: list[SourcePoisoningFinding] = []
    findings.extend(_detect_fake_or_weak_citations(source_text, source_id=source_id, url=url))
    findings.extend(_detect_keyword_stuffing(source_text, source_id=source_id, url=url))
    findings.extend(_detect_duplicate_content(source_text, source_id=source_id, url=url))
    findings.extend(
        _detect_metadata_and_domain_mismatch(
            source_text,
            source_id=source_id,
            url=url,
            title=title,
            metadata=metadata,
        )
    )
    findings.extend(_detect_hidden_text(source_text, source_id=source_id, url=url))
    return findings


def _detect_fake_or_weak_citations(
    text: str, *, source_id: str, url: str
) -> list[SourcePoisoningFinding]:
    out: list[SourcePoisoningFinding] = []
    citations = CITATION_RE.findall(text)
    has_reference_section = bool(
        re.search(r"^\s*(references|bibliography|works cited)\s*$", text, re.I | re.M)
    )
    if len(citations) >= 3 and not has_reference_section:
        out.append(
            _finding(
                source_id=source_id,
                url=url,
                category="fake_or_unverifiable_citations",
                risk_level="medium",
                evidence=", ".join(citations[:8]),
                explanation=(
                    "The source contains several citation-looking markers without a visible "
                    "references section; verify before treating them as real citations."
                ),
            )
        )

    absolute_claims = ABSOLUTE_CLAIM_RE.findall(text)
    link_count = len(re.findall(r"https?://", text))
    if len(absolute_claims) >= 8 and len(citations) + link_count < 2:
        out.append(
            _finding(
                source_id=source_id,
                url=url,
                category="excessive_unsupported_claims",
                risk_level="medium",
                evidence=", ".join(absolute_claims[:12]),
                explanation="The source makes many absolute claims with little citation support.",
            )
        )
    journal_signal = re.search(
        r"\bJournal of (?:Advanced|International|Modern) [A-Z][A-Za-z ]+\b", text
    )
    unverifiable_signal = re.search(
        r"\b(?:forthcoming|unpublished|private study|internal data)\b", text, re.I
    )
    if journal_signal and unverifiable_signal:
        out.append(
            _finding(
                source_id=source_id,
                url=url,
                category="invented_reference_signal",
                risk_level="medium",
                evidence="Invented-looking journal/reference language with unverifiable status.",
                explanation=(
                    "The reference language has hallmarks of unverifiable invented support."
                ),
            )
        )
    return out


def _detect_keyword_stuffing(
    text: str, *, source_id: str, url: str
) -> list[SourcePoisoningFinding]:
    words = [w.lower() for w in WORD_RE.findall(text) if w.lower() not in STOPWORDS]
    if len(words) < 80:
        return []
    counts = Counter(words)
    word, count = counts.most_common(1)[0]
    ratio = count / max(1, len(words))
    comma_runs = re.findall(r"(?:\b[a-z][a-z0-9\-]{2,}\b\s*,\s*){8,}", text, flags=re.I)
    if count >= 14 and ratio >= 0.08:
        return [
            _finding(
                source_id=source_id,
                url=url,
                category="keyword_stuffing",
                risk_level="medium",
                evidence=f"{word} repeated {count} times ({ratio:.1%} of content words)",
                explanation=(
                    "A single non-trivial keyword dominates the text, consistent with SEO spam "
                    "or source poisoning."
                ),
                metadata={"keyword": word, "count": count, "ratio": ratio},
            )
        ]
    if comma_runs:
        return [
            _finding(
                source_id=source_id,
                url=url,
                category="seo_spam",
                risk_level="medium",
                evidence=comma_runs[0],
                explanation="The source contains long comma-separated keyword runs.",
            )
        ]
    return []


def _detect_duplicate_content(
    text: str, *, source_id: str, url: str
) -> list[SourcePoisoningFinding]:
    paragraphs = [
        " ".join(p.split()).lower()
        for p in re.split(r"\n\s*\n", text or "")
        if len(p.split()) >= 8
    ]
    if len(paragraphs) < 4:
        return []
    counts = Counter(paragraphs)
    repeated = [(p, c) for p, c in counts.items() if c >= 3]
    if not repeated:
        return []
    para, count = repeated[0]
    return [
        _finding(
            source_id=source_id,
            url=url,
            category="scraped_duplicate_content",
            risk_level="low",
            evidence=para,
            explanation=(
                f"A substantial paragraph is repeated {count} times, suggesting scraped "
                "or padded content."
            ),
            metadata={"repeat_count": count},
        )
    ]


def _detect_hidden_text(text: str, *, source_id: str, url: str) -> list[SourcePoisoningFinding]:
    match = HIDDEN_TEXT_RE.search(text or "")
    if not match:
        return []
    return [
        _finding(
            source_id=source_id,
            url=url,
            category="hidden_or_tiny_text_indicator",
            risk_level="medium",
            evidence=match.group(0),
            explanation="The extracted text contains a visible indicator of hidden or tiny text.",
        )
    ]


def _detect_metadata_and_domain_mismatch(
    text: str,
    *,
    source_id: str,
    url: str,
    title: str | None,
    metadata: dict[str, Any],
) -> list[SourcePoisoningFinding]:
    out: list[SourcePoisoningFinding] = []
    domain = _domain(metadata.get("final_url") or url)
    original_domain = _domain(metadata.get("url") or url)
    canonical_domain = _domain(metadata.get("canonical_url"))
    if original_domain and domain and not _related_domain(original_domain, domain):
        out.append(
            _finding(
                source_id=source_id,
                url=url,
                category="suspicious_redirect_source_mismatch",
                risk_level="medium",
                evidence=f"{original_domain} -> {domain}",
                explanation="Fetched source redirected to a different registrable-looking domain.",
                metadata={"original_domain": original_domain, "final_domain": domain},
            )
        )
    if canonical_domain and domain and not _related_domain(canonical_domain, domain):
        out.append(
            _finding(
                source_id=source_id,
                url=url,
                category="contradictory_canonical_metadata",
                risk_level="medium",
                evidence=f"final={domain}, canonical={canonical_domain}",
                explanation="Canonical URL domain differs from the fetched final URL domain.",
            )
        )

    if OFFICIAL_CLAIM_RE.search(text) and (
        any(host in domain for host in FREE_HOSTS) or not domain.endswith((".gov", ".edu", ".int"))
    ):
        out.append(
            _finding(
                source_id=source_id,
                url=url,
                category="fake_official_source_claim",
                risk_level="high",
                evidence=f"official-source claim on domain {domain or 'unknown'}",
                explanation=(
                    "The source claims official or primary status, but the domain does not look "
                    "like a government, education, international organization, or known "
                    "official domain."
                ),
                recommended_action="quote_only",
                metadata={"domain": domain},
            )
        )

    title_words = {
        word.lower()
        for word in WORD_RE.findall(title or "")
        if word.lower() not in STOPWORDS and len(word) > 3
    }
    if len(title_words) >= 3:
        body_words = {word.lower() for word in WORD_RE.findall(text[:4000])}
        overlap = title_words & body_words
        if not overlap:
            out.append(
                _finding(
                    source_id=source_id,
                    url=url,
                    category="title_body_topic_mismatch",
                    risk_level="low",
                    evidence=f"title={title}",
                    explanation="The title topic terms do not appear in the extracted body sample.",
                    metadata={"title_terms": sorted(title_words)},
                )
            )
    return out
