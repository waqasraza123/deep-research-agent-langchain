from __future__ import annotations

import hashlib
import re

from .contracts import ResearchIntent, ResearchStrategy
from .decomposition import decompose_question, normalize_question
from .scoring import assess_complexity
from .strategy import (
    ARTIFACTS_BY_INTENT,
    agent_instructions,
    build_evidence_requirements,
    build_source_priorities,
    build_verification_plan,
    citation_requirements,
)

INTENT_KEYWORDS: dict[ResearchIntent, set[str]] = {
    ResearchIntent.COMPARATIVE_ANALYSIS: {
        "compare",
        "comparison",
        "versus",
        "vs",
        "tradeoff",
        "tradeoffs",
        "pros",
        "cons",
        "alternative",
        "alternatives",
        "which is better",
    },
    ResearchIntent.TECHNICAL_DUE_DILIGENCE: {
        "due diligence",
        "production",
        "architecture",
        "scalability",
        "observability",
        "security",
        "failure modes",
        "maturity",
        "reliability",
    },
    ResearchIntent.MARKET_RESEARCH: {
        "market",
        "competitors",
        "pricing",
        "customers",
        "segment",
        "adoption",
        "revenue",
        "tam",
        "sam",
        "som",
    },
    ResearchIntent.LEGAL_OR_POLICY_REVIEW: {
        "legal",
        "law",
        "regulation",
        "policy",
        "compliance",
        "jurisdiction",
        "statute",
        "contract",
        "terms",
    },
    ResearchIntent.ACADEMIC_LITERATURE_REVIEW: {
        "literature review",
        "paper",
        "papers",
        "study",
        "studies",
        "peer reviewed",
        "methodology",
        "citation",
        "academic",
    },
    ResearchIntent.IMPLEMENTATION_RESEARCH: {
        "implement",
        "implementation",
        "build",
        "integrate",
        "api",
        "sdk",
        "backend",
        "deploy",
        "tool calling",
    },
    ResearchIntent.RISK_ASSESSMENT: {
        "risk",
        "risks",
        "threat",
        "threats",
        "failure",
        "liability",
        "mitigation",
        "audit",
        "safety",
    },
    ResearchIntent.FACTUAL_SUMMARY: {
        "summarize",
        "summary",
        "what is",
        "explain",
        "overview",
        "brief",
    },
}


def _phrase_present(text: str, phrase: str) -> bool:
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE))


def classify_research_intent(question: str) -> tuple[ResearchIntent, float, list[str]]:
    lowered = question.lower()
    scores: dict[ResearchIntent, int] = {}
    matches: list[str] = []

    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if _phrase_present(lowered, keyword):
                score += 3 if " " in keyword else 2
                matches.append(f"{intent.value}:{keyword}")
        if score:
            scores[intent] = score

    if not scores:
        return ResearchIntent.UNKNOWN, 0.35, []

    # Comparison is intentionally strong because it determines output shape.
    if scores.get(ResearchIntent.COMPARATIVE_ANALYSIS, 0) >= 2:
        intent = ResearchIntent.COMPARATIVE_ANALYSIS
    else:
        intent = max(scores.items(), key=lambda item: item[1])[0]

    total = sum(scores.values())
    confidence = 0.5 + min(0.4, scores[intent] / max(total, 1) * 0.35 + len(scores) * 0.025)
    if intent == ResearchIntent.UNKNOWN:
        confidence = 0.35
    return intent, min(0.95, confidence), sorted(set(matches))


class ResearchPlanner:
    def create_strategy(self, question: str, urls: list[str] | None = None) -> ResearchStrategy:
        urls = [u.strip() for u in (urls or []) if u and u.strip()]
        normalized = normalize_question(question)
        intent, intent_confidence, intent_matches = classify_research_intent(normalized)
        assessment = assess_complexity(normalized, urls)
        decomposition = decompose_question(normalized, intent)

        evidence_requirements = build_evidence_requirements(
            decomposition.required_evidence_types,
            assessment,
        )
        source_priorities = build_source_priorities(decomposition.likely_source_categories)
        verification_plan = build_verification_plan(
            decomposition.subquestions,
            decomposition.likely_source_categories,
            assessment,
        )
        suggested_artifacts = ARTIFACTS_BY_INTENT[intent]
        risk_flags = list(dict.fromkeys([*assessment.risk_flags, *intent_matches]))
        instructions = agent_instructions(
            intent=intent,
            assessment=assessment,
            source_priorities=source_priorities,
            suggested_artifacts=suggested_artifacts,
            assumptions_to_verify=decomposition.assumptions_to_verify,
            possible_opposing_views=decomposition.possible_opposing_views,
        )

        confidence = round(
            min(0.95, (intent_confidence * 0.55) + (assessment.confidence_score * 0.45)),
            3,
        )
        rid = hashlib.sha1((normalized + "\n" + "\n".join(urls)).encode("utf-8")).hexdigest()[:12]

        return ResearchStrategy(
            research_id=rid,
            normalized_question=decomposition.primary_question,
            intent=intent,
            complexity_score=assessment.level,
            confidence_score=confidence,
            subquestions=decomposition.subquestions,
            evidence_requirements=evidence_requirements,
            source_priorities=source_priorities,
            risk_flags=risk_flags,
            suggested_artifacts=suggested_artifacts,
            agent_instructions=instructions,
            citation_requirements=citation_requirements(assessment),
            verification_plan=verification_plan,
            required_definitions=decomposition.required_definitions,
            assumptions_to_verify=decomposition.assumptions_to_verify,
            possible_opposing_views=decomposition.possible_opposing_views,
            missing_information_warnings=decomposition.missing_information_warnings,
            source_urls=urls,
        )


def create_research_strategy(question: str, urls: list[str] | None = None) -> ResearchStrategy:
    return ResearchPlanner().create_strategy(question, urls)
