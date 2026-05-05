from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from .contracts import BalanceAssessment, Severity

_COMPARATIVE_TERMS = (
    "compare",
    "versus",
    " vs ",
    "which",
    "best",
    "better",
    "should",
    "recommend",
    "pros",
    "cons",
    "tradeoff",
    "trade-off",
)
_CONTROVERSIAL_TERMS = (
    "controvers",
    "debate",
    "risk",
    "policy",
    "regulation",
    "legal",
    "ethic",
    "safety",
)
_BOTH_SIDE_TERMS = ("both", "while", "whereas", "compared", "respectively")
_TRADEOFF_TERMS = ("tradeoff", "trade-off", "pros", "cons", "advantage", "disadvantage")
_COUNTER_TERMS = ("counterargument", "critics", "criticism", "opposing", "however", "although")
_LIMITATION_TERMS = ("limitation", "caveat", "uncertain", "depends", "not enough evidence")
_VENDOR_DOMAINS = (
    "aws.amazon.com",
    "azure.microsoft.com",
    "cloud.google.com",
    "openai.com",
    "anthropic.com",
    "stripe.com",
    "vercel.com",
)


def assess_balance(
    *,
    question: str,
    report_text: str,
    sources: list[dict[str, Any]],
) -> BalanceAssessment:
    q = question.lower()
    r = report_text.lower()
    is_sensitive = any(term in q for term in _COMPARATIVE_TERMS + _CONTROVERSIAL_TERMS)
    includes_both = any(term in r for term in _BOTH_SIDE_TERMS)
    includes_tradeoffs = any(term in r for term in _TRADEOFF_TERMS)
    includes_counter = any(term in r for term in _COUNTER_TERMS)
    includes_limitations = any(term in r for term in _LIMITATION_TERMS)
    diversity = _source_diversity_score(sources)
    vendor_warning = _vendor_bias_warning(sources)

    if not is_sensitive:
        score = 0.85 if includes_limitations else 0.75
        reasons = ["Question does not appear strongly comparative or controversial."]
    else:
        components = [
            0.22 if includes_both else 0.0,
            0.20 if includes_tradeoffs else 0.0,
            0.18 if includes_counter else 0.0,
            0.18 if includes_limitations else 0.0,
            0.22 * diversity,
        ]
        score = sum(components)
        reasons = []
        if not includes_both:
            reasons.append("Report does not clearly cover both sides.")
        if not includes_tradeoffs:
            reasons.append("Report does not clearly explain tradeoffs.")
        if not includes_counter:
            reasons.append("Report lacks counterarguments or opposing views.")
        if not includes_limitations:
            reasons.append("Report lacks limitations or caveats.")
        if diversity < 0.5:
            reasons.append("Source diversity is low.")
    if vendor_warning:
        score = max(0.0, score - 0.08)
        reasons.append(vendor_warning)

    return BalanceAssessment(
        score=round(max(0.0, min(1.0, score)), 3),
        severity=_severity_for_score(score),
        is_comparative_or_controversial=is_sensitive,
        includes_both_sides=includes_both,
        includes_tradeoffs=includes_tradeoffs,
        includes_counterarguments=includes_counter,
        includes_limitations=includes_limitations,
        source_diversity_score=diversity,
        vendor_bias_warning=vendor_warning,
        reasons=reasons or ["Balance checks did not detect major issues."],
    )


def _source_diversity_score(sources: list[dict[str, Any]]) -> float:
    domains = {
        _domain(str(source.get("final_url") or source.get("url") or ""))
        for source in sources
        if source.get("ok", True) is not False
    }
    domains.discard("")
    if not domains:
        return 0.0
    if len(domains) == 1:
        return 0.35
    if len(domains) == 2:
        return 0.7
    return 1.0


def _vendor_bias_warning(sources: list[dict[str, Any]]) -> str | None:
    domains = [
        _domain(str(source.get("final_url") or source.get("url") or ""))
        for source in sources
        if source.get("ok", True) is not False
    ]
    domains = [domain for domain in domains if domain]
    if not domains:
        return None
    vendor_count = sum(
        1 for domain in domains if any(vendor in domain for vendor in _VENDOR_DOMAINS)
    )
    if vendor_count and vendor_count == len(domains):
        return "All usable sources appear to be vendor-controlled; add independent sources."
    if vendor_count / len(domains) >= 0.67:
        return "Most usable sources appear vendor-controlled; check for vendor bias."
    return None


def _domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host.lower()


def _severity_for_score(score: float) -> Severity:
    if score < 0.25:
        return "critical"
    if score < 0.45:
        return "high"
    if score < 0.65:
        return "medium"
    if score < 0.8:
        return "low"
    return "info"
