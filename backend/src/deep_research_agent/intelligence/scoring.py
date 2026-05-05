from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .contracts import ComplexityLevel

COMPARISON_TERMS = {
    "compare",
    "versus",
    "vs",
    "alternative",
    "alternatives",
    "tradeoff",
    "tradeoffs",
    "pros",
    "cons",
    "better",
}

HIGH_STAKES_TERMS = {
    "legal",
    "law",
    "lawsuit",
    "regulation",
    "regulatory",
    "compliance",
    "policy",
    "medical",
    "clinical",
    "health",
    "financial",
    "investment",
    "security",
    "privacy",
    "risk",
    "liability",
    "safety",
}

TIME_SENSITIVE_TERMS = {
    "latest",
    "recent",
    "current",
    "today",
    "yesterday",
    "this year",
    "2024",
    "2025",
    "2026",
    "now",
}

ADVERSARIAL_TERMS = {
    "opposing",
    "counterargument",
    "counterarguments",
    "challenge",
    "dispute",
    "debunk",
    "audit",
    "red team",
    "failure mode",
    "failure modes",
    "risk assessment",
}

DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "technical": {
        "api",
        "backend",
        "deployment",
        "architecture",
        "implementation",
        "database",
        "streaming",
        "checkpoint",
        "tool calling",
        "framework",
        "langgraph",
        "crewai",
    },
    "market": {"market", "pricing", "competitor", "segment", "customer", "adoption", "revenue"},
    "legal_policy": {"legal", "policy", "regulation", "compliance", "contract", "terms"},
    "academic": {"paper", "study", "literature", "citation", "methodology", "peer reviewed"},
    "financial": {"financial", "investment", "budget", "roi", "cost", "funding"},
    "security": {"security", "privacy", "threat", "vulnerability", "incident", "abuse"},
}


class ComplexityAssessment(BaseModel):
    level: ComplexityLevel
    numeric_score: int = Field(..., ge=0, le=100)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    freshness_required: bool = False
    entity_count: int = 0
    domain_count: int = 0


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9.+#-]*", text.lower()))


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE))


def _count_entities(question: str) -> int:
    acronyms = re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", question)
    titled = re.findall(r"\b[A-Z][a-zA-Z0-9.+#-]*(?:\s+[A-Z][a-zA-Z0-9.+#-]*)*", question)
    entities = {e.strip() for e in acronyms + titled if len(e.strip()) > 1}
    return len(entities)


def detect_domains(question: str) -> list[str]:
    lowered = question.lower()
    tokens = _tokens(question)
    found: list[str] = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in tokens or _contains_phrase(lowered, keyword) for keyword in keywords):
            found.append(domain)
    return found


def assess_complexity(question: str, urls: list[str] | None = None) -> ComplexityAssessment:
    urls = urls or []
    tokens = _tokens(question)
    lowered = question.lower()
    word_count = len(re.findall(r"\b\w+\b", question))
    entity_count = _count_entities(question)
    domains = detect_domains(question)

    score = 10
    signals: list[str] = []
    risk_flags: list[str] = []

    if word_count > 22:
        score += 12
        signals.append("long_question")
    if word_count > 45:
        score += 10
        signals.append("very_long_question")
    if entity_count >= 2:
        score += min(18, entity_count * 4)
        signals.append("multiple_entities")
    if any(term in tokens or _contains_phrase(lowered, term) for term in COMPARISON_TERMS):
        score += 15
        signals.append("comparison_requested")
    if any(term in tokens or _contains_phrase(lowered, term) for term in HIGH_STAKES_TERMS):
        score += 14
        signals.append("high_stakes_language")
        risk_flags.append(
            "High-stakes claims require explicit uncertainty and source quality checks."
        )
    freshness_required = any(_contains_phrase(lowered, term) for term in TIME_SENSITIVE_TERMS)
    if freshness_required:
        score += 9
        signals.append("freshness_sensitive")
        risk_flags.append("Freshness/date sensitivity should be highlighted in the report.")
    if urls:
        score += min(9, len(urls) * 3)
        signals.append("provided_sources")
    if len(domains) >= 2:
        score += 18
        signals.append("multi_domain")
        risk_flags.append(
            "The question spans multiple domains; avoid collapsing domain-specific evidence."
        )
    if any(term in tokens or _contains_phrase(lowered, term) for term in ADVERSARIAL_TERMS):
        score += 16
        signals.append("adversarial_or_failure_analysis")
        risk_flags.append("Opposing views or failure modes are required.")
    if any(term in tokens for term in {"best", "should", "recommend", "recommendation"}):
        score += 8
        signals.append("recommendation_requested")
    if not signals:
        signals.append("plain_question")

    score = max(0, min(100, score))
    if "adversarial_or_failure_analysis" in signals and score >= 45:
        level = ComplexityLevel.ADVERSARIAL
    elif len(domains) >= 2 and score >= 45:
        level = ComplexityLevel.MULTI_DOMAIN
    elif score >= 62:
        level = ComplexityLevel.DEEP
    elif score >= 32:
        level = ComplexityLevel.MODERATE
    else:
        level = ComplexityLevel.SIMPLE

    confidence = 0.64 + min(0.26, len(signals) * 0.035)
    if question.strip().endswith("?"):
        confidence += 0.03
    if word_count < 5:
        confidence -= 0.15

    return ComplexityAssessment(
        level=level,
        numeric_score=score,
        confidence_score=max(0.1, min(0.95, confidence)),
        signals=signals,
        risk_flags=risk_flags,
        freshness_required=freshness_required,
        entity_count=entity_count,
        domain_count=len(domains),
    )
