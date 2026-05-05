from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import EvidenceType, ResearchIntent, SourceCategory, SubQuestion
from .scoring import detect_domains

DEFINITION_TERMS = {
    "production": "Define what production readiness means for this question.",
    "research agent": "Define the expected agent workflow and output contract.",
    "backend": (
        "Define backend constraints such as persistence, APIs, deployment, "
        "and observability."
    ),
    "compliance": "Define the applicable jurisdiction, policy scope, and compliance standard.",
    "market": "Define market geography, segment, timeframe, and buyer persona.",
    "risk": "Define risk categories and acceptable severity thresholds.",
}


@dataclass(frozen=True)
class DecompositionResult:
    primary_question: str
    subquestions: list[SubQuestion]
    required_evidence_types: list[EvidenceType]
    likely_source_categories: list[SourceCategory]
    required_definitions: list[str]
    assumptions_to_verify: list[str]
    possible_opposing_views: list[str]
    missing_information_warnings: list[str]


def normalize_question(question: str) -> str:
    q = re.sub(r"\s+", " ", question).strip()
    if q and q[-1] not in ".?!":
        q += "?"
    return q


def extract_entities(question: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9.+#-]*(?:\s+[A-Z][A-Za-z0-9.+#-]*)*", question)
    acronyms = re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", question)
    seen: set[str] = set()
    out: list[str] = []
    for item in candidates + acronyms:
        clean = item.strip()
        if len(clean) < 2 or clean.lower() in {"compare", "summarize"}:
            continue
        if clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out[:8]


def _base_evidence(intent: ResearchIntent) -> list[EvidenceType]:
    if intent == ResearchIntent.LEGAL_OR_POLICY_REVIEW:
        return [
            EvidenceType.LEGAL_OR_POLICY_TEXT,
            EvidenceType.PRIMARY_SOURCE,
            EvidenceType.EXPERT_ANALYSIS,
        ]
    if intent == ResearchIntent.ACADEMIC_LITERATURE_REVIEW:
        return [
            EvidenceType.ACADEMIC_PAPER,
            EvidenceType.PRIMARY_SOURCE,
            EvidenceType.EXPERT_ANALYSIS,
        ]
    if intent == ResearchIntent.MARKET_RESEARCH:
        return [EvidenceType.MARKET_DATA, EvidenceType.PRIMARY_SOURCE, EvidenceType.CURRENT_SOURCE]
    if intent == ResearchIntent.IMPLEMENTATION_RESEARCH:
        return [
            EvidenceType.OFFICIAL_DOCUMENTATION,
            EvidenceType.IMPLEMENTATION_EXAMPLE,
            EvidenceType.SOURCE_CODE,
        ]
    if intent == ResearchIntent.TECHNICAL_DUE_DILIGENCE:
        return [
            EvidenceType.OFFICIAL_DOCUMENTATION,
            EvidenceType.IMPLEMENTATION_EXAMPLE,
            EvidenceType.BENCHMARK_OR_METRIC,
            EvidenceType.RISK_OR_INCIDENT_RECORD,
        ]
    if intent == ResearchIntent.RISK_ASSESSMENT:
        return [
            EvidenceType.RISK_OR_INCIDENT_RECORD,
            EvidenceType.PRIMARY_SOURCE,
            EvidenceType.EXPERT_ANALYSIS,
        ]
    return [
        EvidenceType.PRIMARY_SOURCE,
        EvidenceType.OFFICIAL_DOCUMENTATION,
        EvidenceType.EXPERT_ANALYSIS,
    ]


def _source_categories(intent: ResearchIntent, domains: list[str]) -> list[SourceCategory]:
    categories: list[SourceCategory] = []
    if "technical" in domains or intent in {
        ResearchIntent.IMPLEMENTATION_RESEARCH,
        ResearchIntent.TECHNICAL_DUE_DILIGENCE,
        ResearchIntent.COMPARATIVE_ANALYSIS,
    }:
        categories.extend(
            [
                SourceCategory.OFFICIAL_DOCS,
                SourceCategory.SOURCE_CODE,
                SourceCategory.RELEASE_NOTES,
                SourceCategory.BENCHMARKS,
            ]
        )
    if intent == ResearchIntent.LEGAL_OR_POLICY_REVIEW or "legal_policy" in domains:
        categories.extend(
            [SourceCategory.LEGAL_DATABASES, SourceCategory.REGULATOR_OR_POLICY_SITES]
        )
    if intent == ResearchIntent.MARKET_RESEARCH or "market" in domains:
        categories.extend(
            [
                SourceCategory.COMPANY_OR_PRODUCT_PAGES,
                SourceCategory.MARKET_REPORTS,
                SourceCategory.NEWS_OR_DISCLOSURES,
            ]
        )
    if intent == ResearchIntent.ACADEMIC_LITERATURE_REVIEW or "academic" in domains:
        categories.append(SourceCategory.ACADEMIC_DATABASES)
    if SourceCategory.COMMUNITY_DISCUSSION not in categories:
        categories.append(SourceCategory.COMMUNITY_DISCUSSION)

    seen: set[SourceCategory] = set()
    out: list[SourceCategory] = []
    for item in categories:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _subquestion(
    idx: int,
    question: str,
    rationale: str,
    evidence_types: list[EvidenceType],
    source_categories: list[SourceCategory],
    *,
    priority: int = 3,
    opposing_view_required: bool = False,
) -> SubQuestion:
    return SubQuestion(
        id=f"SQ{idx}",
        question=question,
        rationale=rationale,
        priority=priority,
        evidence_types=evidence_types,
        source_categories=source_categories,
        opposing_view_required=opposing_view_required,
    )


def _technical_comparison_subquestions(
    subject: str,
    sources: list[SourceCategory],
) -> list[SubQuestion]:
    evidence = [EvidenceType.OFFICIAL_DOCUMENTATION, EvidenceType.IMPLEMENTATION_EXAMPLE]
    questions = [
        (
            "How do the options differ in orchestration model, state management, and control flow?",
            "Architecture determines how reliably the backend can run long research tasks.",
        ),
        (
            "What persistence, checkpointing, resume, and failure-recovery "
            "capabilities are documented?",
            "Research agents need traceable recovery when tools, models, or network calls fail.",
        ),
        (
            "How mature are streaming, tool calling, human review, and observability integrations?",
            "Operational interfaces determine whether the backend can be debugged and trusted.",
        ),
        (
            "What deployment, scaling, and ecosystem support exists for production use?",
            "Production readiness depends on maintenance, hosting patterns, "
            "and integration surface.",
        ),
        (
            "What known failure modes, lock-in risks, and long-term maintenance "
            "costs should be expected?",
            "A useful recommendation must expose downside scenarios, not just feature coverage.",
        ),
    ]
    return [
        _subquestion(
            i,
            f"{q} Scope: {subject}",
            rationale,
            evidence,
            sources,
            priority=5 if i < 3 else 4,
        )
        for i, (q, rationale) in enumerate(questions, start=1)
    ]


def decompose_question(question: str, intent: ResearchIntent) -> DecompositionResult:
    normalized = normalize_question(question)
    entities = extract_entities(question)
    domains = detect_domains(question)
    evidence = _base_evidence(intent)
    sources = _source_categories(intent, domains)
    subject = ", ".join(entities) if entities else "the requested topic"

    subquestions: list[SubQuestion] = []
    if intent == ResearchIntent.COMPARATIVE_ANALYSIS and ("technical" in domains or entities):
        subquestions.extend(_technical_comparison_subquestions(subject, sources))
    else:
        subquestions.extend(
            [
                _subquestion(
                    1,
                    f"What is the precise scope and factual background for {subject}?",
                    "The report needs a grounded baseline before analysis.",
                    evidence[:2],
                    sources[:3],
                    priority=5,
                ),
                _subquestion(
                    2,
                    "Which claims are best supported by primary or official sources?",
                    "The agent should separate high-confidence evidence from "
                    "secondary interpretation.",
                    [EvidenceType.PRIMARY_SOURCE],
                    sources[:3],
                    priority=5,
                ),
                _subquestion(
                    3,
                    "What constraints, risks, or exceptions could change the conclusion?",
                    "Edge cases and uncertainty prevent overconfident recommendations.",
                    [EvidenceType.RISK_OR_INCIDENT_RECORD, EvidenceType.EXPERT_ANALYSIS],
                    sources,
                    priority=4,
                    opposing_view_required=True,
                ),
            ]
        )

    if intent == ResearchIntent.MARKET_RESEARCH:
        subquestions.append(
            _subquestion(
                len(subquestions) + 1,
                "What market size, buyer segment, pricing, and adoption signals are available?",
                "Market claims require current quantitative and qualitative evidence.",
                [EvidenceType.MARKET_DATA, EvidenceType.CURRENT_SOURCE],
                sources,
                priority=5,
            )
        )
    elif intent == ResearchIntent.LEGAL_OR_POLICY_REVIEW:
        subquestions.append(
            _subquestion(
                len(subquestions) + 1,
                "Which jurisdictions, statutes, policies, or enforcement guidance "
                "control the answer?",
                "Legal and policy analysis is invalid without scope and authority checks.",
                [EvidenceType.LEGAL_OR_POLICY_TEXT, EvidenceType.PRIMARY_SOURCE],
                sources,
                priority=5,
            )
        )
    elif intent == ResearchIntent.ACADEMIC_LITERATURE_REVIEW:
        subquestions.append(
            _subquestion(
                len(subquestions) + 1,
                "What are the strongest studies, methods, limitations, and consensus gaps?",
                "Literature reviews must distinguish evidence quality from publication volume.",
                [EvidenceType.ACADEMIC_PAPER],
                sources,
                priority=5,
            )
        )

    required_definitions = [
        explanation
        for term, explanation in DEFINITION_TERMS.items()
        if re.search(rf"\b{re.escape(term)}\b", normalized, flags=re.IGNORECASE)
    ]
    if entities:
        required_definitions.extend(
            f"Define the role and version/scope of {entity}." for entity in entities[:4]
        )

    assumptions = [
        "Verify that provided URLs are authoritative enough for the requested conclusion.",
        "Verify whether source publication or update dates are relevant to the question.",
    ]
    if intent == ResearchIntent.COMPARATIVE_ANALYSIS:
        assumptions.append(
            "Verify that comparison criteria are weighted consistently across options."
        )
    if "technical" in domains:
        assumptions.append(
            "Verify compatibility with production backend constraints, not just demo examples."
        )

    opposing = [
        "A source may favor one option because it is vendor-authored or community-promoted.",
        (
            "A technically weaker option may be operationally simpler or cheaper "
            "in the user's context."
        ),
    ]
    if intent in {ResearchIntent.LEGAL_OR_POLICY_REVIEW, ResearchIntent.RISK_ASSESSMENT}:
        opposing.append(
            "Different jurisdictions, standards, or risk tolerances may support "
            "different conclusions."
        )

    warnings: list[str] = []
    if not entities and intent != ResearchIntent.FACTUAL_SUMMARY:
        warnings.append(
            "The question does not name clear entities; the agent may need to infer scope."
        )
    if not re.search(r"\b(202[0-9]|latest|current|recent|today|now)\b", normalized, re.IGNORECASE):
        warnings.append(
            "No timeframe was provided; note source dates and avoid implying current status."
        )
    if "best" in normalized.lower() or "should" in normalized.lower():
        warnings.append(
            "Recommendation language requires explicit criteria before ranking options."
        )

    return DecompositionResult(
        primary_question=normalized,
        subquestions=subquestions,
        required_evidence_types=evidence,
        likely_source_categories=sources,
        required_definitions=required_definitions,
        assumptions_to_verify=assumptions,
        possible_opposing_views=opposing,
        missing_information_warnings=warnings,
    )
