from __future__ import annotations

import re
from collections import Counter

from .contracts import (
    KernelWarning,
    ResearchComplexity,
    ResearchIntent,
    stable_id,
    tokenize,
)

COMPARISON_TERMS = {"compare", " vs ", "versus", "better", "alternative", "tradeoff", "trade-off"}
TECHNICAL_TERMS = {
    "api",
    "backend",
    "architecture",
    "langgraph",
    "langchain",
    "fastapi",
    "database",
    "orchestration",
    "deployment",
    "scaling",
    "source code",
    "sdk",
    "framework",
    "library",
}
LEGAL_TERMS = {"law", "regulation", "compliance", "policy", "terms", "liability", "gdpr", "privacy"}
FINANCIAL_TERMS = {
    "investment",
    "stock",
    "pricing",
    "cost",
    "revenue",
    "valuation",
    "risk",
    "margin",
    "market",
}
MEDICAL_TERMS = {"diagnosis", "treatment", "symptoms", "medication", "doctor", "health", "clinical"}
CURRENT_TERMS = {
    "latest",
    "current",
    "today",
    "now",
    "recent",
    "2026",
    "pricing",
    "release",
    "version",
}
ACADEMIC_TERMS = {"paper", "literature", "study", "citation", "methodology", "benchmark"}
IMPLEMENTATION_TERMS = {"implement", "build", "plan", "roadmap", "migrate", "integrate", "deploy"}
BROAD_TERMS = {"overview", "everything", "landscape", "all", "best", "comprehensive", "deep dive"}
AMBIGUOUS_TERMS = {"thing", "stuff", "it", "good", "bad", "what about", "help me"}


def _matches(question_lc: str, terms: set[str]) -> list[str]:
    found: list[str] = []
    padded = f" {question_lc} "
    for term in terms:
        if " " in term:
            if term in padded:
                found.append(term.strip())
        elif re.search(rf"\b{re.escape(term)}\b", question_lc):
            found.append(term)
    return sorted(found)


def _named_entities(question: str) -> list[str]:
    entities = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:[- ][A-Z][A-Za-z0-9]*)*\b", question)
    return [e for e in entities if e.lower() not in {"I", "The", "A", "An"}]


def analyze_request(
    question: str, urls: list[str] | None = None
) -> tuple[ResearchIntent, ResearchComplexity, list[KernelWarning]]:
    urls = urls or []
    q = " ".join(question.split())
    q_lc = q.lower()
    signals = {
        "comparison": _matches(q_lc, COMPARISON_TERMS),
        "technical": _matches(q_lc, TECHNICAL_TERMS),
        "legal": _matches(q_lc, LEGAL_TERMS),
        "financial": _matches(q_lc, FINANCIAL_TERMS),
        "medical": _matches(q_lc, MEDICAL_TERMS),
        "current": _matches(q_lc, CURRENT_TERMS),
        "academic": _matches(q_lc, ACADEMIC_TERMS),
        "implementation": _matches(q_lc, IMPLEMENTATION_TERMS),
        "broad": _matches(q_lc, BROAD_TERMS),
        "ambiguous": _matches(q_lc, AMBIGUOUS_TERMS),
    }
    reasons: list[str] = []
    label = "general_research"
    confidence = 0.55

    if signals["medical"]:
        label, confidence = "medical_health_review", 0.9
        reasons.append("Detected medical or health-review terms.")
    elif signals["legal"]:
        label, confidence = "legal_policy_review", 0.88
        reasons.append("Detected legal, policy, compliance, or liability terms.")
    elif signals["financial"] and (
        "risk" in signals["financial"]
        or "investment" in signals["financial"]
        or "stock" in signals["financial"]
    ):
        label, confidence = "financial_risk_review", 0.84
        reasons.append("Detected financial risk or investment terms.")
    elif (
        re.match(r"^\s*(what|who|when|where)\s+(is|are|was|were)\b", q_lc)
        and not signals["current"]
    ):
        label, confidence = "factual_answer", 0.72
        reasons.append("Short direct question resembles a factual answer request.")
    elif signals["comparison"] and signals["technical"]:
        label, confidence = "technical_due_diligence", 0.86
        reasons.append("Detected both comparative and technical architecture terms.")
    elif signals["comparison"]:
        label, confidence = "comparative_analysis", 0.78
        reasons.append("Detected comparison or tradeoff language.")
    elif signals["implementation"] and signals["technical"]:
        label, confidence = "implementation_planning", 0.8
        reasons.append("Detected implementation planning for technical work.")
    elif signals["technical"]:
        label, confidence = "library_or_framework_review", 0.74
        reasons.append("Detected technical library, API, backend, or framework terms.")
    elif signals["academic"]:
        label, confidence = "academic_literature_review", 0.75
        reasons.append("Detected academic literature or benchmark language.")
    elif signals["current"]:
        label, confidence = "news_or_current_review", 0.72
        reasons.append("Detected currentness-sensitive terms.")
    elif "?" in q and len(tokenize(q)) <= 8:
        label, confidence = "factual_answer", 0.68
        reasons.append("Short direct question resembles a factual answer request.")
    if "source code" in q_lc:
        label, confidence = "source_code_research", max(confidence, 0.82)
        reasons.append("Detected source-code research terms.")
    if "vendor" in q_lc:
        label, confidence = "vendor_evaluation", max(confidence, 0.78)
        reasons.append("Detected vendor-evaluation language.")
    if "risk" in q_lc and label not in {
        "financial_risk_review",
        "medical_health_review",
        "legal_policy_review",
    }:
        label, confidence = "risk_assessment", max(confidence, 0.72)
        reasons.append("Detected general risk-assessment language.")
    if not q.strip():
        label, confidence = "unknown", 0.0
        reasons.append("Question was empty.")

    intent = ResearchIntent(
        intent_id=stable_id("intent", label, q_lc),
        label=label,  # type: ignore[arg-type]
        confidence_score=confidence,
        reasons=reasons or ["No specialized intent signals dominated the request."],
        signals={k: v for k, v in signals.items() if v},
    )

    entities = _named_entities(q)
    numbers = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", q)
    dimensions: list[str] = []
    score = 0.12
    if len(q) > 80:
        score += 0.12
        dimensions.append("long_question")
    if len(q) > 180:
        score += 0.12
        dimensions.append("very_long_question")
    if len(urls) == 0:
        score += 0.08
        dimensions.append("missing_urls")
    elif len(urls) > 3:
        score += 0.12
        dimensions.append("many_urls")
    if len(entities) >= 2:
        score += min(0.16, 0.04 * len(entities))
        dimensions.append("named_entities")
    for name in ("comparison", "current", "academic", "implementation", "broad", "ambiguous"):
        if signals[name]:
            score += 0.07
            dimensions.append(name)
    if signals["technical"]:
        score += 0.09
        dimensions.append("technical_vocabulary")
    sensitive = bool(signals["legal"] or signals["medical"] or signals["financial"])
    if sensitive:
        score += 0.18
        dimensions.append("sensitive_domain")
    if numbers:
        score += 0.06
        dimensions.append("numeric_terms")
    domain_counter = Counter(
        name
        for name in ("technical", "legal", "financial", "medical", "academic", "current")
        if signals[name]
    )
    if len(domain_counter) >= 2:
        score += 0.12
        dimensions.append("multi_domain_vocabulary")
    score = min(score, 1.0)
    if sensitive:
        level = "sensitive"
    elif signals["ambiguous"] and signals["broad"]:
        level = "adversarial"
    elif score >= 0.74 or len(domain_counter) >= 3:
        level = "multi_domain"
    elif score >= 0.52:
        level = "deep"
    elif score >= 0.28:
        level = "moderate"
    else:
        level = "simple"
    complexity = ResearchComplexity(
        level=level,  # type: ignore[arg-type]
        score=round(score, 4),
        reasons=[f"Detected {dimension.replace('_', ' ')}." for dimension in dimensions]
        or ["Low breadth and low risk signal."],
        detected_dimensions=dimensions,
    )
    warnings: list[KernelWarning] = []
    if not urls:
        warnings.append(
            KernelWarning(
                warning_id=stable_id("warn", "request_analyzer", "missing_urls", q_lc),
                subsystem="request_analyzer",
                code="missing_urls",
                severity="medium",
                message="No URLs were supplied; deterministic rebuild quality depends on existing artifacts.",
                recommended_action="Provide primary sources or enable a source-discovery pass before relying on conclusions.",
            )
        )
    if sensitive:
        warnings.append(
            KernelWarning(
                warning_id=stable_id("warn", "request_analyzer", "sensitive", q_lc),
                subsystem="request_analyzer",
                code="sensitive_domain",
                severity="high",
                message="The request appears to involve legal, medical, or financial risk.",
                recommended_action="Use conservative wording, strict citations, and human expert review.",
            )
        )
    return intent, complexity, warnings
