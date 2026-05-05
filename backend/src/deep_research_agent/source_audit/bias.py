from __future__ import annotations

from ._heuristics import clamp, count_pattern, host_domain, text_head, url_path
from .contracts import SourceBiasRisk

PROMOTIONAL_PATTERNS = (
    r"\bbest\b",
    r"\bultimate\b",
    r"\bguaranteed\b",
    r"\brevolutionary\b",
    r"\bgame[- ]changing\b",
    r"\bindustry[- ]leading\b",
    r"\bno\.?\s*1\b",
    r"\bunbeatable\b",
)
AFFILIATE_PATTERNS = (
    r"\baffiliate\b",
    r"\bcommission\b",
    r"\bsponsored\b",
    r"\bpartner links?\b",
    r"\bwe may earn\b",
)
COMPARISON_PATTERNS = (
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\balternative to\b",
    r"\bcompetitor\b",
    r"\bcompare\b",
    r"\bcomparison\b",
)
LOADED_PATTERNS = (
    r"\bterrible\b",
    r"\bamazing\b",
    r"\bdisaster\b",
    r"\bmust-have\b",
    r"\bhate\b",
    r"\blove\b",
)
ONE_SIDED_PATTERNS = (
    r"\bonly\b.*\bsolution\b",
    r"\bwithout any downside\b",
    r"\bperfect for everyone\b",
    r"\bnever fails\b",
)


def analyze_bias_risk(
    *,
    url: str,
    title: str | None,
    text: str,
) -> SourceBiasRisk:
    domain = host_domain(url)
    path = url_path(url)
    joined = "\n".join([domain, path, title or "", text_head(text)]).lower()
    signals: list[str] = []
    mitigations: list[str] = []
    score = 0.18

    promotional = count_pattern(joined, PROMOTIONAL_PATTERNS)
    affiliate = count_pattern(joined, AFFILIATE_PATTERNS)
    comparison = count_pattern(joined, COMPARISON_PATTERNS)
    loaded = count_pattern(joined, LOADED_PATTERNS)
    one_sided = count_pattern(joined, ONE_SIDED_PATTERNS)

    if promotional:
        score += min(0.28, promotional * 0.045)
        signals.append("Promotional or superlative language detected.")
    if affiliate:
        score += min(0.34, affiliate * 0.12)
        signals.append("Affiliate, sponsored, or commission language detected.")
    if comparison:
        score += min(0.22, comparison * 0.045)
        signals.append("Competitor comparison or vendor-comparison framing detected.")
    if loaded:
        score += min(0.16, loaded * 0.04)
        signals.append("Emotionally loaded language detected.")
    if one_sided:
        score += min(0.2, one_sided * 0.08)
        signals.append("One-sided absolute claims detected.")
    if "/pricing" in path or "/compare" in path or "/versus" in path:
        score += 0.08
        signals.append("Commercial pricing or comparison URL path.")
    if "/docs" in path or "/reference" in path:
        score -= 0.08
        mitigations.append("Documentation/reference path lowers bias risk.")
    if any(term in joined for term in ("references", "methodology", "doi:", "data source")):
        score -= 0.07
        mitigations.append("References, methodology, or data-source language mitigates bias risk.")
    if domain.endswith((".gov", ".edu")):
        score -= 0.08
        mitigations.append("Institutional domain lowers commercial conflict risk.")

    risk_score = round(clamp(score), 3)
    if risk_score >= 0.62:
        level = "high"
    elif risk_score >= 0.36:
        level = "medium"
    else:
        level = "low"

    reasons = signals + mitigations
    if not reasons:
        reasons.append("No strong bias or conflict indicators were detected.")

    return SourceBiasRisk(
        score=risk_score,
        risk_level=level,  # type: ignore[arg-type]
        signals=signals,
        mitigating_factors=mitigations,
        reasons=reasons,
    )
