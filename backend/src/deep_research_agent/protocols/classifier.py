from __future__ import annotations

import re
from collections import defaultdict
from urllib.parse import urlparse

from .contracts import (
    CitationPolicy,
    EvaluationPolicy,
    FreshnessPolicy,
    IntelligenceProfile,
    PolicyPack,
    ProtocolSelection,
    ProtocolWarning,
    ResearchProtocol,
    SourceRequirement,
    SynthesisPolicy,
    VerificationRequirement,
)
from .instruction_builder import build_protocol_instruction_block
from .policy_packs import policy_packs_for_protocol, warnings_for_policy_packs
from .profiles import get_profile
from .registry import ProtocolRegistry

COMPARISON_RE = re.compile(r"\b(vs\.?|versus|compare|comparison|which is better|alternative)\b", re.I)
DECISION_RE = re.compile(r"\b(choose|adopt|buy|select|migrate|should we|recommend|decision)\b", re.I)
CURRENT_RE = re.compile(r"\b(latest|current|today|recent|breaking|this week|this month|202[5-9]|pricing)\b", re.I)

SIGNAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "medical_or_health_information_review": (
        "medical", "health", "disease", "symptom", "treatment", "drug", "clinical",
        "diagnosis", "therapy", "patient", "vaccine", "dosage", "side effect",
    ),
    "legal_policy_review": (
        "legal", "law", "regulation", "policy", "compliance", "contract", "privacy",
        "gdpr", "hipaa", "terms of service", "liability", "jurisdiction", "statute",
    ),
    "financial_or_investment_risk_review": (
        "investment", "invest", "stock", "revenue", "earnings", "valuation", "financial",
        "portfolio", "sec", "10-k", "10-q", "risk factor", "cash flow", "market cap",
    ),
    "academic_literature_review": (
        "literature review", "paper", "study", "academic", "peer reviewed", "methodology",
        "doi", "meta-analysis", "systematic review", "research gap", "journal",
    ),
    "market_research": (
        "market", "tam", "sam", "som", "competitor", "segment", "trend", "growth",
        "forecast", "customer", "demand", "industry", "market size",
    ),
    "vendor_evaluation": (
        "vendor", "product", "pricing", "sla", "soc 2", "soc2", "procurement", "rfp",
        "enterprise", "security page", "status page", "alternatives",
    ),
    "technical_due_diligence": (
        "due diligence", "technical risk", "architecture", "scalability", "reliability",
        "security", "production", "maintainability", "operational",
    ),
    "software_framework_comparison": (
        "framework", "library", "sdk", "api", "langchain", "langgraph", "django",
        "fastapi", "react", "next.js", "postgres", "redis", "package", "compare",
    ),
    "implementation_planning": (
        "implement", "implementation", "roadmap", "migration", "integrate", "build",
        "deploy", "architecture plan", "steps", "rollout",
    ),
    "source_code_or_library_review": (
        "github", "repository", "repo", "source code", "dependency", "package", "license",
        "changelog", "release notes", "maintainer",
    ),
    "news_or_current_events_review": (
        "latest", "current", "today", "breaking", "news", "recent", "announced",
        "this week", "this month", "developing",
    ),
}

DOMAIN_SIGNALS: dict[str, tuple[str, ...]] = {
    "source_code_or_library_review": ("github.com", "gitlab.com", "bitbucket.org", "npmjs.com", "pypi.org"),
    "academic_literature_review": ("arxiv.org", "pubmed.ncbi.nlm.nih.gov", "scholar.google", "doi.org", "acm.org", "ieee.org", "springer.com"),
    "legal_policy_review": ("law.cornell.edu", "govinfo.gov", "ecfr.gov", "justice.gov", "ftc.gov", "sec.gov"),
    "financial_or_investment_risk_review": ("sec.gov", "investor.", "finance.yahoo.com", "nasdaq.com"),
    "medical_or_health_information_review": ("nih.gov", "cdc.gov", "who.int", "mayoclinic.org", "nejm.org"),
    "vendor_evaluation": ("status.", "trust.", "security.", "pricing."),
    "news_or_current_events_review": ("reuters.com", "apnews.com", "bbc.com", "nytimes.com"),
}


def _domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _contains(text: str, needle: str) -> bool:
    return needle in text if " " in needle or "." in needle or "-" in needle else bool(
        re.search(rf"\b{re.escape(needle)}\b", text)
    )


def _score_protocols(question: str, urls: list[str]) -> tuple[dict[str, float], dict[str, list[str]]]:
    text = " ".join(question.lower().split())
    domains = [_domain(url) for url in urls if url]
    scores: dict[str, float] = defaultdict(float)
    reasons: dict[str, list[str]] = defaultdict(list)

    for protocol_id, keywords in SIGNAL_KEYWORDS.items():
        for keyword in keywords:
            if _contains(text, keyword):
                scores[protocol_id] += 2.0 if " " in keyword else 1.0
                reasons[protocol_id].append(f"Matched keyword `{keyword}`.")

    for protocol_id, fragments in DOMAIN_SIGNALS.items():
        for domain in domains:
            if any(fragment in domain for fragment in fragments):
                scores[protocol_id] += 1.5
                reasons[protocol_id].append(f"Matched URL domain `{domain}`.")

    if COMPARISON_RE.search(question):
        scores["software_framework_comparison"] += 1.2
        scores["vendor_evaluation"] += 0.8
        scores["market_research"] += 0.4
        reasons["software_framework_comparison"].append("Question uses comparison language.")
    if DECISION_RE.search(question):
        scores["implementation_planning"] += 0.9
        scores["vendor_evaluation"] += 0.9
        scores["technical_due_diligence"] += 0.6
        reasons["implementation_planning"].append("Question asks for an adoption or implementation decision.")
    if CURRENT_RE.search(question):
        scores["news_or_current_events_review"] += 1.4
        scores["market_research"] += 0.5
        reasons["news_or_current_events_review"].append("Question uses current/latest language.")

    if "how to" in text or text.startswith(("build ", "implement ", "integrate ")):
        scores["implementation_planning"] += 1.0
        reasons["implementation_planning"].append("Question shape is implementation-oriented.")
    if "vs" in text and any(word in text for word in ("framework", "library", "api", "sdk")):
        scores["software_framework_comparison"] += 1.3
        reasons["software_framework_comparison"].append("Software comparison shape detected.")

    if not scores:
        scores["general_research"] = 1.0
        reasons["general_research"].append("No specialized protocol signal dominated.")
    else:
        scores["general_research"] += 0.2
        reasons["general_research"].append("General research remains a fallback alternative.")
    return dict(scores), dict(reasons)


def _confidence(best: float, second: float) -> float:
    if best <= 0:
        return 0.35
    margin = max(0.0, best - second)
    return round(min(0.95, 0.45 + (best / 10.0) + min(0.25, margin / 6.0)), 3)


def choose_profile_for_protocol(
    protocol_id: str,
    *,
    requested_profile_id: str | None = None,
    mock_mode: bool = False,
) -> IntelligenceProfile:
    if requested_profile_id:
        return get_profile(requested_profile_id)
    if mock_mode:
        return get_profile("offline_mock")
    if protocol_id in {
        "legal_policy_review",
        "financial_or_investment_risk_review",
        "medical_or_health_information_review",
    }:
        return get_profile("conservative_verification")
    if protocol_id in {
        "technical_due_diligence",
        "software_framework_comparison",
        "implementation_planning",
        "source_code_or_library_review",
    }:
        return get_profile("technical_architect")
    if protocol_id in {"academic_literature_review", "news_or_current_events_review"}:
        return get_profile("citation_strict")
    return get_profile("balanced_research")


def _merge_source_requirements(
    protocol: ResearchProtocol, policy_packs: list[PolicyPack]
) -> list[SourceRequirement]:
    merged = list(protocol.required_source_types)
    seen = {(item.source_type, item.required) for item in merged}
    for pack in policy_packs:
        for requirement in pack.source_requirements:
            key = (requirement.source_type, requirement.required)
            if key not in seen:
                merged.append(requirement)
                seen.add(key)
    return merged


def _effective_verification(
    protocol: ResearchProtocol, profile: IntelligenceProfile
) -> VerificationRequirement:
    order = {"low": 0, "standard": 1, "high": 2, "very_high": 3}
    strictness = max(
        protocol.verification_strictness.strictness,
        profile.verification_strictness,
        key=lambda item: order[item],
    )
    return protocol.verification_strictness.copy(update={"strictness": strictness})


def _effective_citation(protocol: ResearchProtocol, profile: IntelligenceProfile) -> CitationPolicy:
    order = {"standard": 0, "strict": 1, "primary_source_required": 2}
    strictness = max(
        protocol.citation_requirements.strictness,
        profile.citation_strictness,
        key=lambda item: order[item],
    )
    return protocol.citation_requirements.copy(update={"strictness": strictness})


def select_protocol(
    *,
    question: str,
    urls: list[str] | None = None,
    requested_protocol_id: str | None = None,
    requested_profile_id: str | None = None,
    mock_mode: bool = False,
    registry: ProtocolRegistry | None = None,
) -> ProtocolSelection:
    registry = registry or ProtocolRegistry()
    clean_urls = [url.strip() for url in urls or [] if url and url.strip()]
    scores, reasons_by_protocol = _score_protocols(question, clean_urls)
    if requested_protocol_id:
        selected_id = requested_protocol_id
        scores[selected_id] = max(scores.get(selected_id, 0.0), 10.0)
        reasons_by_protocol.setdefault(selected_id, []).append("Protocol was explicitly requested.")
    else:
        selected_id = max(scores.items(), key=lambda item: (item[1], item[0]))[0]

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_score = sorted_scores[0][1] if sorted_scores else 1.0
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

    protocol = registry.get(selected_id)
    profile = choose_profile_for_protocol(
        selected_id,
        requested_profile_id=requested_profile_id,
        mock_mode=mock_mode,
    )
    packs = policy_packs_for_protocol(selected_id)
    warnings = list(warnings_for_policy_packs(packs))
    if protocol.safety_warnings.safety_warnings:
        warnings.extend(
            ProtocolWarning(
                code=f"{protocol.protocol_id}_safety",
                message=message,
                severity="high",
                review_recommended=protocol.safety_warnings.require_human_review,
            )
            for message in protocol.safety_warnings.safety_warnings
        )
    if protocol.freshness_requirements.strictness == "current_required":
        warnings.append(
            ProtocolWarning(
                code="current_sources_required",
                message="Use recent, dated sources and state freshness limitations.",
                severity="medium",
                review_recommended=False,
            )
        )

    alternatives = [
        protocol_id for protocol_id, _score in sorted_scores if protocol_id != selected_id
    ][:3]
    if "general_research" not in alternatives and selected_id != "general_research":
        alternatives.append("general_research")

    confidence = 0.98 if requested_protocol_id else _confidence(best_score, second_score)
    review_recommended = (
        profile.review_gate_recommended
        or protocol.safety_warnings.require_human_review
        or any(warning.review_recommended for warning in warnings)
    )
    instruction_block = build_protocol_instruction_block(
        protocol=protocol,
        profile=profile,
        policy_packs=packs,
    )
    return ProtocolSelection(
        selected_protocol=protocol,
        intelligence_profile=profile,
        confidence_score=confidence,
        alternative_protocols=alternatives,
        reasons=reasons_by_protocol.get(selected_id) or ["Selected by deterministic fallback."],
        warnings=warnings,
        policy_packs=packs,
        effective_source_requirements=_merge_source_requirements(protocol, packs),
        effective_verification=_effective_verification(protocol, profile),
        effective_citation_policy=_effective_citation(protocol, profile),
        effective_freshness_policy=FreshnessPolicy(**protocol.freshness_requirements.dict()),
        effective_synthesis_policy=SynthesisPolicy(**protocol.synthesis_profile.dict()),
        effective_evaluation_policy=EvaluationPolicy(**protocol.evaluation_weights.dict()),
        review_recommended=review_recommended,
        instruction_block=instruction_block,
    )

