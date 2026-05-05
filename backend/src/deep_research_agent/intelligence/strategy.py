from __future__ import annotations

from .contracts import (
    EvidenceRequirement,
    EvidenceType,
    ResearchIntent,
    SourceCategory,
    SourcePriority,
    SubQuestion,
    VerificationStep,
)
from .scoring import ComplexityAssessment

ARTIFACTS_BY_INTENT: dict[ResearchIntent, list[str]] = {
    ResearchIntent.FACTUAL_SUMMARY: ["plan.md", "notes.md", "sources.json", "report.md"],
    ResearchIntent.COMPARATIVE_ANALYSIS: [
        "plan.md",
        "notes.md",
        "sources.json",
        "comparison_matrix.md",
        "report.md",
    ],
    ResearchIntent.TECHNICAL_DUE_DILIGENCE: [
        "plan.md",
        "notes.md",
        "sources.json",
        "risk_register.md",
        "report.md",
    ],
    ResearchIntent.MARKET_RESEARCH: [
        "plan.md",
        "notes.md",
        "sources.json",
        "market_snapshot.md",
        "report.md",
    ],
    ResearchIntent.LEGAL_OR_POLICY_REVIEW: [
        "plan.md",
        "notes.md",
        "sources.json",
        "authority_table.md",
        "report.md",
    ],
    ResearchIntent.ACADEMIC_LITERATURE_REVIEW: [
        "plan.md",
        "notes.md",
        "sources.json",
        "literature_matrix.md",
        "report.md",
    ],
    ResearchIntent.IMPLEMENTATION_RESEARCH: [
        "plan.md",
        "notes.md",
        "sources.json",
        "implementation_notes.md",
        "report.md",
    ],
    ResearchIntent.RISK_ASSESSMENT: [
        "plan.md",
        "notes.md",
        "sources.json",
        "risk_register.md",
        "report.md",
    ],
    ResearchIntent.UNKNOWN: ["plan.md", "notes.md", "sources.json", "report.md"],
}


SOURCE_RATIONALES: dict[SourceCategory, str] = {
    SourceCategory.OFFICIAL_DOCS: "Official docs define supported behavior and public commitments.",
    SourceCategory.SOURCE_CODE: (
        "Source code confirms implementation details and maintenance signals."
    ),
    SourceCategory.RELEASE_NOTES: (
        "Release notes expose maturity, breaking changes, and recent direction."
    ),
    SourceCategory.BENCHMARKS: "Benchmarks and metrics support performance or reliability claims.",
    SourceCategory.ACADEMIC_DATABASES: (
        "Academic indexes help identify peer-reviewed evidence and limitations."
    ),
    SourceCategory.LEGAL_DATABASES: (
        "Legal databases provide controlling authority and case or statute text."
    ),
    SourceCategory.REGULATOR_OR_POLICY_SITES: (
        "Regulators and policy owners provide authoritative guidance."
    ),
    SourceCategory.COMPANY_OR_PRODUCT_PAGES: (
        "Company pages clarify positioning, pricing, and roadmap claims."
    ),
    SourceCategory.MARKET_REPORTS: (
        "Market reports provide adoption, sizing, and buyer-context evidence."
    ),
    SourceCategory.NEWS_OR_DISCLOSURES: (
        "News and disclosures capture current events and public incidents."
    ),
    SourceCategory.COMMUNITY_DISCUSSION: (
        "Community discussion can reveal rough edges but should not dominate."
    ),
}


EVIDENCE_DESCRIPTIONS: dict[EvidenceType, str] = {
    EvidenceType.PRIMARY_SOURCE: "Use original sources for core factual claims.",
    EvidenceType.OFFICIAL_DOCUMENTATION: (
        "Use official documentation for stated capabilities and limits."
    ),
    EvidenceType.SOURCE_CODE: (
        "Use source code or repository metadata for implementation and maintenance claims."
    ),
    EvidenceType.IMPLEMENTATION_EXAMPLE: (
        "Use concrete examples for feasibility and integration claims."
    ),
    EvidenceType.BENCHMARK_OR_METRIC: "Use metrics for performance, maturity, or adoption claims.",
    EvidenceType.ACADEMIC_PAPER: (
        "Use papers for methods, evidence quality, and consensus boundaries."
    ),
    EvidenceType.LEGAL_OR_POLICY_TEXT: (
        "Use controlling text for legal, regulatory, or policy conclusions."
    ),
    EvidenceType.MARKET_DATA: "Use market data for adoption, pricing, and demand claims.",
    EvidenceType.EXPERT_ANALYSIS: (
        "Use expert analysis to interpret tradeoffs after primary evidence."
    ),
    EvidenceType.RISK_OR_INCIDENT_RECORD: (
        "Use incident records or documented failures for risk claims."
    ),
    EvidenceType.CURRENT_SOURCE: (
        "Use date-stamped current sources when freshness affects the answer."
    ),
}


def build_evidence_requirements(
    evidence_types: list[EvidenceType],
    assessment: ComplexityAssessment,
) -> list[EvidenceRequirement]:
    seen: set[EvidenceType] = set()
    out: list[EvidenceRequirement] = []
    for item in evidence_types:
        if item in seen:
            continue
        seen.add(item)
        min_sources = 2 if assessment.level.value in {"deep", "adversarial", "multi_domain"} else 1
        out.append(
            EvidenceRequirement(
                evidence_type=item,
                description=EVIDENCE_DESCRIPTIONS[item],
                priority=(
                    5
                    if item
                    in {EvidenceType.PRIMARY_SOURCE, EvidenceType.OFFICIAL_DOCUMENTATION}
                    else 4
                ),
                minimum_sources=min_sources,
            )
        )
    if assessment.freshness_required and EvidenceType.CURRENT_SOURCE not in seen:
        out.append(
            EvidenceRequirement(
                evidence_type=EvidenceType.CURRENT_SOURCE,
                description=EVIDENCE_DESCRIPTIONS[EvidenceType.CURRENT_SOURCE],
                priority=5,
                minimum_sources=2,
            )
        )
    return out


def build_source_priorities(categories: list[SourceCategory]) -> list[SourcePriority]:
    out: list[SourcePriority] = []
    for idx, category in enumerate(categories):
        out.append(
            SourcePriority(
                category=category,
                rationale=SOURCE_RATIONALES[category],
                priority=max(1, 5 - min(idx, 3)),
            )
        )
    return out


def build_verification_plan(
    subquestions: list[SubQuestion],
    source_categories: list[SourceCategory],
    assessment: ComplexityAssessment,
) -> list[VerificationStep]:
    steps = [
        VerificationStep(
            claim_area="Scope and definitions",
            method=(
                "Confirm that key terms, entities, versions, jurisdictions, "
                "and timeframe are explicit."
            ),
            priority=5,
            required_sources=source_categories[:3],
        ),
        VerificationStep(
            claim_area="Core factual claims",
            method=(
                "Cross-check important claims against primary or official evidence "
                "before using them."
            ),
            priority=5,
            required_sources=source_categories[:3],
        ),
    ]
    if any(sq.opposing_view_required for sq in subquestions) or assessment.level.value in {
        "adversarial",
        "multi_domain",
    }:
        steps.append(
            VerificationStep(
                claim_area="Opposing views and failure modes",
                method=(
                    "Identify credible contrary evidence and explain when it changes "
                    "the conclusion."
                ),
                priority=5,
                required_sources=source_categories,
            )
        )
    if assessment.freshness_required:
        steps.append(
            VerificationStep(
                claim_area="Freshness",
                method=(
                    "Check source publication dates and avoid stale claims when status "
                    "may have changed."
                ),
                priority=5,
                required_sources=source_categories,
            )
        )
    steps.append(
        VerificationStep(
            claim_area="Citation integrity",
            method=(
                "Ensure every non-obvious claim in the final report maps to a captured "
                "source id."
            ),
            priority=4,
            required_sources=source_categories[:4],
        )
    )
    return steps


def citation_requirements(assessment: ComplexityAssessment) -> list[str]:
    requirements = [
        "Cite every factual claim that affects the conclusion using source ids such as [S1].",
        "Prefer primary and official sources over summaries when they disagree.",
        "Separate source facts from interpretation in notes.md and report.md.",
    ]
    if assessment.freshness_required:
        requirements.append("Include publication or update dates for freshness-sensitive claims.")
    if assessment.risk_flags:
        requirements.append("Mark uncertain, high-stakes, or weakly sourced claims explicitly.")
    return requirements


def agent_instructions(
    *,
    intent: ResearchIntent,
    assessment: ComplexityAssessment,
    source_priorities: list[SourcePriority],
    suggested_artifacts: list[str],
    assumptions_to_verify: list[str],
    possible_opposing_views: list[str],
) -> str:
    source_text = ", ".join(item.category.value for item in source_priorities[:5])
    artifacts_text = ", ".join(suggested_artifacts)
    lines = [
        f"Research intent: {intent.value}. Complexity: {assessment.level.value}.",
        "Structure the work around the strategy subquestions before drafting conclusions.",
        f"Prioritize evidence from: {source_text or 'the most authoritative available sources'}.",
        (
            "Do not assume missing definitions, current status, jurisdiction, "
            "version, or comparison criteria."
        ),
        "When evidence conflicts, explain the conflict, source quality, and resulting uncertainty.",
        f"Produce or preserve these artifacts when relevant: {artifacts_text}.",
    ]
    if assumptions_to_verify:
        lines.append("Verify these assumptions: " + " ".join(assumptions_to_verify[:4]))
    if possible_opposing_views or assessment.level.value == "adversarial":
        lines.append(
            "Include credible opposing views and failure modes before making recommendations."
        )
    if assessment.freshness_required:
        lines.append("Highlight source dates and whether freshness affects confidence.")
    return "\n".join(lines)
