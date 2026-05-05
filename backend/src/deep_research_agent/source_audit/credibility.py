from __future__ import annotations

import re

from ._heuristics import clamp, count_pattern, host_domain, text_head, url_path
from .contracts import SourceCredibilityScore

OFFICIAL_DOC_PATTERNS = (
    r"\bofficial\b",
    r"\bdocumentation\b",
    r"\bapi reference\b",
    r"\bdeveloper guide\b",
    r"\breference manual\b",
)
REFERENCE_PATTERNS = (
    r"\breferences\b",
    r"\bcitations?\b",
    r"\bdoi:\b",
    r"\barxiv\b",
    r"https?://",
    r"\bsource:\b",
)
MARKETING_PATTERNS = (
    r"\brevolutionary\b",
    r"\bgame[- ]changing\b",
    r"\bindustry[- ]leading\b",
    r"\bworld'?s best\b",
    r"\bguaranteed\b",
    r"\bunlock\b",
    r"\bseamless\b",
    r"\beffortless\b",
)
SEO_PATTERNS = (
    r"\bbest\b",
    r"\btop\s+\d+\b",
    r"\bultimate guide\b",
    r"\beverything you need to know\b",
    r"\b2026 guide\b",
    r"\bbuy now\b",
    r"\bclick here\b",
)
AUTHOR_DATE_PATTERNS = (
    r"\bby\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\b",
    r"\bauthor\b",
    r"\bpublished\b",
    r"\bupdated\b",
    r"\blast modified\b",
)


def score_credibility(
    *,
    url: str,
    title: str | None,
    text: str,
    metadata: dict | None = None,
) -> SourceCredibilityScore:
    domain = host_domain(url)
    path = url_path(url)
    head = text_head(text)
    joined = "\n".join([title or "", path, head])
    positives: list[str] = []
    negatives: list[str] = []
    score = 0.5
    word_count = int((metadata or {}).get("word_count") or len(re.findall(r"\b\w+\b", text or "")))

    if domain.endswith((".gov", ".edu", ".mil", ".ac.uk")):
        score += 0.24
        positives.append("Government, academic, or institutional domain.")
    standards_hosts = ("ietf.org", "w3.org", "nist.gov", "rfc-editor.org", "iso.org")
    if any(domain == host or domain.endswith("." + host) for host in standards_hosts):
        score += 0.28
        positives.append("Standards body or public specification publisher.")
    docs_paths = ("/docs", "/documentation", "/reference", "/api/")
    if "docs" in domain or any(part in path for part in docs_paths):
        score += 0.18
        positives.append("Official documentation or reference indicator.")
    if any(part in path for part in ("/research", "/paper", "/report", "/whitepaper")):
        score += 0.12
        positives.append("Research, paper, report, or whitepaper path.")
    if count_pattern(joined, OFFICIAL_DOC_PATTERNS) >= 2:
        score += 0.08
        positives.append("Title or content uses documentation/reference language.")
    if count_pattern(joined, AUTHOR_DATE_PATTERNS) >= 1:
        score += 0.07
        positives.append("Author, publication, update, or modification signal found.")
    ref_count = count_pattern(joined, REFERENCE_PATTERNS)
    if ref_count >= 3:
        score += 0.11
        positives.append("Clear references, citations, DOI, or outbound source links.")
    elif ref_count == 0:
        score -= 0.08
        negatives.append("No visible references or citation signals.")

    if word_count >= 1200:
        score += 0.08
        positives.append("Substantial extracted content.")
    elif word_count < 160:
        score -= 0.22
        negatives.append("Thin extracted content.")
    elif word_count < 350:
        score -= 0.1
        negatives.append("Short extracted content.")

    marketing_hits = count_pattern(joined, MARKETING_PATTERNS)
    if marketing_hits >= 2:
        score -= 0.16
        negatives.append("Excessive promotional or marketing language.")
    elif marketing_hits == 1:
        score -= 0.06
        negatives.append("Some promotional language detected.")

    seo_hits = count_pattern(joined, SEO_PATTERNS)
    if seo_hits >= 4:
        score -= 0.2
        negatives.append("Obvious SEO or content-farm language pattern.")
    elif seo_hits >= 2:
        score -= 0.1
        negatives.append("SEO-style phrasing detected.")

    if not title:
        score -= 0.07
        negatives.append("No identifiable title.")
    if not re.search(r"\b(?:by|author|published|updated|last modified|copyright)\b", joined, re.I):
        score -= 0.07
        negatives.append("Anonymous or unattributed content.")
    copied = "copied from" in joined.lower()
    if any(term in domain for term in ("scrape", "mirror", "contentfarm")) or copied:
        score -= 0.18
        negatives.append("Possible scraped, mirrored, or copied content.")

    reasons = positives + negatives
    if not reasons:
        reasons.append("Credibility is based on generic source structure; no strong signals found.")

    return SourceCredibilityScore(
        score=round(clamp(score), 3),
        positive_signals=positives,
        negative_signals=negatives,
        reasons=reasons,
    )
