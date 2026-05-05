from __future__ import annotations

from copy import deepcopy

from .contracts import (
    CitationPolicy,
    EvaluationPolicy,
    FreshnessPolicy,
    ProtocolRule,
    ResearchProtocol,
    SafetyPolicy,
    SourceRequirement,
    SourceType,
    SynthesisPolicy,
    VerificationRequirement,
)
from .errors import UnknownProtocolError
from .validators import validate_protocol

BASE_ARTIFACTS = [
    "protocol_selection.json",
    "protocol_selection.md",
    "intelligence_profile.json",
    "protocol_instructions.md",
    "policy_requirements.json",
    "policy_warnings.md",
    "plan.md",
    "notes.md",
    "sources.json",
    "report.md",
    "source_audit.json",
    "evidence_ledger.json",
    "synthesis_output.json",
    "evaluation.json",
]


def _source(
    source_type: SourceType,
    rationale: str,
    *,
    minimum_count: int = 1,
    required: bool = True,
    freshness_days: int | None = None,
    examples: list[str] | None = None,
) -> SourceRequirement:
    return SourceRequirement(
        source_type=source_type,
        minimum_count=minimum_count,
        required=required,
        rationale=rationale,
        freshness_days=freshness_days,
        examples=examples or [],
    )


def _weights(**weights: float) -> EvaluationPolicy:
    return EvaluationPolicy(
        required=True,
        weights=weights,
        minimum_overall_score=0.65,
        fail_on_unsupported_decisive_claims=weights.get("citation_quality", 0.0) >= 0.25,
    )


def _protocol(
    *,
    protocol_id: str,
    name: str,
    description: str,
    matching_signals: list[str],
    required_source_types: list[SourceRequirement],
    preferred_source_types: list[SourceType],
    disallowed_or_low_value_source_types: list[SourceType] | None = None,
    strictness: str = "standard",
    citation: CitationPolicy | None = None,
    freshness: FreshnessPolicy | None = None,
    synthesis: SynthesisPolicy | None = None,
    evaluation: EvaluationPolicy | None = None,
    safety: SafetyPolicy | None = None,
    review_when: list[str] | None = None,
    rules: list[ProtocolRule] | None = None,
) -> ResearchProtocol:
    return validate_protocol(
        ResearchProtocol(
            protocol_id=protocol_id,
            name=name,
            description=description,
            matching_signals=matching_signals,
            required_source_types=required_source_types,
            preferred_source_types=preferred_source_types,
            disallowed_or_low_value_source_types=disallowed_or_low_value_source_types or [],
            verification_strictness=VerificationRequirement(
                strictness=strictness,  # type: ignore[arg-type]
                minimum_independent_sources=2 if strictness in {"high", "very_high"} else 1,
                require_primary_source_for_decisive_claims=strictness in {"high", "very_high"},
                require_contradiction_scan=True,
                require_uncertainty_boundaries=True,
            ),
            citation_requirements=citation or CitationPolicy(),
            freshness_requirements=freshness or FreshnessPolicy(),
            required_artifacts=BASE_ARTIFACTS,
            synthesis_profile=synthesis or SynthesisPolicy(),
            evaluation_weights=evaluation or _weights(
                coverage=0.22,
                citation_quality=0.24,
                source_quality=0.20,
                balance=0.14,
                freshness=0.10,
                hallucination_risk=0.10,
            ),
            safety_warnings=safety or SafetyPolicy(),
            operator_review_required_when=review_when or [],
            rules=rules or [],
        )
    )


def built_in_protocols() -> dict[str, ResearchProtocol]:
    common_low_value: list[SourceType] = ["community_discussion", "expert_analysis"]
    protocols = [
        _protocol(
            protocol_id="general_research",
            name="General Research",
            description="Default source-grounded research protocol for non-specialized questions.",
            matching_signals=["what is", "summarize", "overview", "explain", "research"],
            required_source_types=[
                _source("primary_source", "At least one primary or authoritative source should anchor the answer.", required=False),
                _source("expert_analysis", "Expert analysis can add context when clearly cited.", required=False),
            ],
            preferred_source_types=["primary_source", "official_docs", "expert_analysis", "news"],
            disallowed_or_low_value_source_types=["community_discussion"],
            synthesis=SynthesisPolicy(
                profile="deep_research_report",
                required_sections=["Answer", "Evidence", "Limitations", "Sources"],
                forbidden_overclaims=["Unsupported certainty", "Claims beyond captured sources"],
                required_uncertainty_language=["State when sources are incomplete."],
            ),
        ),
        _protocol(
            protocol_id="technical_due_diligence",
            name="Technical Due Diligence",
            description="Evaluates architecture, operational risk, implementation maturity, security, and maintainability.",
            matching_signals=["due diligence", "architecture", "scalability", "security", "production", "reliability", "technical risk"],
            required_source_types=[
                _source("official_docs", "Official documentation is needed for capabilities and constraints."),
                _source("source_code", "Repository or source artifacts support maturity and implementation claims.", required=False),
                _source("release_notes", "Release notes help verify maintenance and version stability.", required=False, freshness_days=540),
            ],
            preferred_source_types=["official_docs", "source_code", "release_notes", "benchmark", "standards_or_specification"],
            disallowed_or_low_value_source_types=common_low_value,
            strictness="high",
            citation=CitationPolicy(
                strictness="strict",
                require_claim_level_citations=True,
                primary_source_required_for=["architecture claims", "security claims", "operational constraints"],
                disallowed_citation_sources=["community_discussion"],
            ),
            freshness=FreshnessPolicy(strictness="prefer_recent", max_age_days=540, require_publication_dates=True),
            synthesis=SynthesisPolicy(
                profile="technical_due_diligence",
                required_sections=["Executive assessment", "Architecture", "Risks", "Evidence gaps", "Recommendation"],
                forbidden_overclaims=["Do not certify security or production readiness from secondary sources alone."],
                required_uncertainty_language=["Use conditional language for untested operational claims."],
            ),
            evaluation=_weights(coverage=0.18, citation_quality=0.25, source_quality=0.22, freshness=0.12, hallucination_risk=0.13, balance=0.10),
            review_when=["High-impact adoption decision", "Security, compliance, or production readiness claim"],
        ),
        _protocol(
            protocol_id="software_framework_comparison",
            name="Software Framework Comparison",
            description="Compares frameworks, libraries, SDKs, APIs, or platforms for adoption decisions.",
            matching_signals=["vs", "versus", "compare", "best framework", "choose", "adopt", "migrate", "library"],
            required_source_types=[
                _source("official_docs", "Feature and API claims should come from official docs."),
                _source("source_code", "Repository evidence helps assess activity and implementation surface.", required=False),
                _source("benchmark", "Benchmarks are useful when performance is part of the decision.", required=False, freshness_days=365),
            ],
            preferred_source_types=["official_docs", "source_code", "release_notes", "benchmark", "implementation_example"],
            disallowed_or_low_value_source_types=["community_discussion"],
            strictness="high",
            citation=CitationPolicy(strictness="strict", require_claim_level_citations=True, primary_source_required_for=["feature comparison", "API compatibility", "pricing or licensing"]),
            freshness=FreshnessPolicy(strictness="prefer_recent", max_age_days=365, require_publication_dates=True),
            synthesis=SynthesisPolicy(
                profile="comparative_report",
                required_sections=["Decision context", "Comparison matrix", "Tradeoffs", "Risks", "Recommendation"],
                forbidden_overclaims=["Do not declare a winner without criteria and cited evidence."],
                required_uncertainty_language=["Flag version, benchmark, and ecosystem gaps."],
            ),
            evaluation=_weights(coverage=0.20, citation_quality=0.24, source_quality=0.20, balance=0.18, freshness=0.10, hallucination_risk=0.08),
        ),
        _protocol(
            protocol_id="implementation_planning",
            name="Implementation Planning",
            description="Turns research into an evidence-backed implementation plan, migration plan, or integration roadmap.",
            matching_signals=["implement", "implementation plan", "roadmap", "migrate", "integrate", "build", "architecture plan"],
            required_source_types=[
                _source("official_docs", "Implementation steps should be grounded in official docs."),
                _source("implementation_example", "Examples help verify real integration shape.", required=False),
                _source("source_code", "Source code helps validate APIs and edge cases.", required=False),
            ],
            preferred_source_types=["official_docs", "implementation_example", "source_code", "release_notes"],
            strictness="high",
            citation=CitationPolicy(strictness="strict", require_claim_level_citations=True, primary_source_required_for=["required steps", "API calls", "configuration"]),
            freshness=FreshnessPolicy(strictness="prefer_recent", max_age_days=540, require_publication_dates=True),
            synthesis=SynthesisPolicy(
                profile="implementation_plan",
                required_sections=["Assumptions", "Plan", "Dependencies", "Risks", "Validation checklist"],
                forbidden_overclaims=["Do not imply code has been tested unless it was actually run."],
                required_uncertainty_language=["Mark unverified steps and environment assumptions."],
            ),
            evaluation=_weights(coverage=0.22, citation_quality=0.24, source_quality=0.18, freshness=0.10, hallucination_risk=0.16, balance=0.10),
            review_when=["Production rollout plan", "Security-sensitive integration"],
        ),
        _protocol(
            protocol_id="source_code_or_library_review",
            name="Source Code Or Library Review",
            description="Reviews a repository, package, library, or API for behavior, quality, risk, or fit.",
            matching_signals=["github", "repo", "repository", "package", "library", "api", "sdk", "dependency", "code review"],
            required_source_types=[
                _source("source_code", "Source repository or package metadata is the primary evidence."),
                _source("official_docs", "Official docs explain intended behavior and support policy.", required=False),
                _source("release_notes", "Release and changelog history informs maintenance risk.", required=False, freshness_days=365),
            ],
            preferred_source_types=["source_code", "official_docs", "release_notes", "benchmark"],
            disallowed_or_low_value_source_types=["community_discussion", "expert_analysis"],
            strictness="high",
            citation=CitationPolicy(strictness="strict", require_claim_level_citations=True, primary_source_required_for=["API behavior", "license", "maintenance", "security posture"]),
            freshness=FreshnessPolicy(strictness="current_required", max_age_days=365, require_publication_dates=True),
            synthesis=SynthesisPolicy(
                profile="technical_due_diligence",
                required_sections=["Scope", "Findings", "Risks", "Evidence", "Adoption notes"],
                forbidden_overclaims=["Do not infer runtime behavior without code, docs, or tests."],
                required_uncertainty_language=["Separate observed evidence from inferred risk."],
            ),
            evaluation=_weights(coverage=0.18, citation_quality=0.26, source_quality=0.24, freshness=0.14, hallucination_risk=0.12, balance=0.06),
        ),
        _protocol(
            protocol_id="legal_policy_review",
            name="Legal Or Policy Review",
            description="Reviews statutes, regulations, policies, agency guidance, or compliance requirements.",
            matching_signals=["legal", "law", "regulation", "policy", "compliance", "terms", "privacy", "contract", "gdpr", "hipaa"],
            required_source_types=[
                _source("legal_text", "Primary legal or policy text is required for legal/policy claims."),
                _source("regulator_guidance", "Regulator or official policy guidance should support interpretations.", required=False, freshness_days=365),
                _source("court_or_agency_record", "Court or agency records are required when specific cases or decisions matter.", required=False),
            ],
            preferred_source_types=["legal_text", "regulator_guidance", "court_or_agency_record", "primary_source"],
            disallowed_or_low_value_source_types=["community_discussion", "expert_analysis", "news"],
            strictness="very_high",
            citation=CitationPolicy(strictness="primary_source_required", require_claim_level_citations=True, primary_source_required_for=["legal requirement", "compliance obligation", "policy interpretation"]),
            freshness=FreshnessPolicy(strictness="current_required", max_age_days=365, require_publication_dates=True),
            synthesis=SynthesisPolicy(
                profile="risk_review",
                required_sections=["Scope", "Authorities reviewed", "Requirements", "Uncertainties", "Human review"],
                forbidden_overclaims=["Do not provide legal advice.", "Do not assert final compliance status."],
                required_uncertainty_language=["Use informational, jurisdiction-limited language."],
            ),
            evaluation=_weights(coverage=0.18, citation_quality=0.30, source_quality=0.20, freshness=0.15, hallucination_risk=0.12, balance=0.05),
            safety=SafetyPolicy(
                conservative_language=True,
                professional_advice_disclaimer=True,
                require_human_review=True,
                safety_warnings=["This is legal/policy information, not legal advice."],
                operator_review_required_when=["Any compliance, liability, or legal decision will be made from the report."],
            ),
            review_when=["Any legal or compliance decision", "Jurisdiction-specific interpretation"],
        ),
        _protocol(
            protocol_id="market_research",
            name="Market Research",
            description="Assesses market size, trends, competitors, customer segments, demand, or adoption.",
            matching_signals=["market", "tam", "sam", "som", "trend", "competitor", "customers", "growth", "forecast"],
            required_source_types=[
                _source("market_data", "Market claims require data or disclosed methodology.", required=False, freshness_days=365),
                _source("company_disclosure", "Company disclosures support competitor and product claims.", required=False),
                _source("news", "Current market developments may require recent reporting.", required=False, freshness_days=180),
            ],
            preferred_source_types=["market_data", "company_disclosure", "financial_filing", "news", "expert_analysis"],
            disallowed_or_low_value_source_types=["community_discussion"],
            strictness="high",
            citation=CitationPolicy(strictness="strict", require_claim_level_citations=True, primary_source_required_for=["market size", "financial metric", "competitor claim"]),
            freshness=FreshnessPolicy(strictness="prefer_recent", max_age_days=365, require_publication_dates=True),
            synthesis=SynthesisPolicy(
                profile="deep_research_report",
                required_sections=["Market definition", "Evidence", "Competitors", "Risks", "Unknowns"],
                forbidden_overclaims=["Do not treat estimates as facts without methodology."],
                required_uncertainty_language=["Label estimates, ranges, and inferred trends."],
            ),
            evaluation=_weights(coverage=0.22, citation_quality=0.24, source_quality=0.18, balance=0.14, freshness=0.14, hallucination_risk=0.08),
        ),
        _protocol(
            protocol_id="vendor_evaluation",
            name="Product Or Vendor Evaluation",
            description="Evaluates vendor/product fit, tradeoffs, claims, pricing, risk, and adoption readiness.",
            matching_signals=["vendor", "product", "pricing", "sla", "soc2", "enterprise", "procurement", "rfp", "alternatives"],
            required_source_types=[
                _source("vendor_documentation", "Vendor claims should be cited to vendor docs or disclosures."),
                _source("company_disclosure", "Company pages, filings, or status/security pages support operational claims.", required=False),
                _source("benchmark", "Third-party benchmarks are useful when performance claims matter.", required=False),
            ],
            preferred_source_types=["vendor_documentation", "company_disclosure", "official_docs", "benchmark", "news"],
            disallowed_or_low_value_source_types=["community_discussion"],
            strictness="high",
            citation=CitationPolicy(strictness="strict", require_claim_level_citations=True, primary_source_required_for=["pricing", "SLA", "security posture", "feature support"]),
            freshness=FreshnessPolicy(strictness="current_required", max_age_days=180, require_publication_dates=True),
            synthesis=SynthesisPolicy(
                profile="decision_memo",
                required_sections=["Decision criteria", "Vendor evidence", "Risks", "Alternatives", "Recommendation"],
                forbidden_overclaims=["Do not validate vendor marketing claims without independent or primary evidence."],
                required_uncertainty_language=["Flag vendor-provided evidence and missing independent verification."],
            ),
            evaluation=_weights(coverage=0.20, citation_quality=0.25, source_quality=0.18, balance=0.18, freshness=0.12, hallucination_risk=0.07),
            review_when=["Procurement or contract decision", "Security or compliance dependency"],
        ),
        _protocol(
            protocol_id="academic_literature_review",
            name="Academic Literature Review",
            description="Reviews academic literature, papers, methods, evidence quality, and research gaps.",
            matching_signals=["literature review", "paper", "study", "academic", "peer reviewed", "methodology", "meta-analysis"],
            required_source_types=[
                _source("academic_paper", "Academic claims should be grounded in papers or proceedings."),
                _source("systematic_review", "Systematic reviews or meta-analyses are preferred for evidence summaries.", required=False),
            ],
            preferred_source_types=["academic_paper", "systematic_review", "primary_source"],
            disallowed_or_low_value_source_types=["community_discussion", "news", "expert_analysis"],
            strictness="high",
            citation=CitationPolicy(strictness="strict", require_claim_level_citations=True, primary_source_required_for=["study finding", "methodology", "effect claim"]),
            freshness=FreshnessPolicy(strictness="prefer_recent", max_age_days=1460, require_publication_dates=True),
            synthesis=SynthesisPolicy(
                profile="literature_style_review",
                required_sections=["Research question", "Evidence base", "Methods quality", "Findings", "Gaps"],
                forbidden_overclaims=["Do not generalize beyond study design or population."],
                required_uncertainty_language=["State study limitations and evidence quality."],
            ),
            evaluation=_weights(coverage=0.24, citation_quality=0.26, source_quality=0.22, balance=0.12, freshness=0.08, hallucination_risk=0.08),
            review_when=["Clinical, legal, financial, or policy decision based on academic findings"],
        ),
        _protocol(
            protocol_id="financial_or_investment_risk_review",
            name="Financial Or Investment Risk Review",
            description="Reviews financial, investment, accounting, pricing, market, or counterparty risk.",
            matching_signals=["investment", "stock", "revenue", "risk", "valuation", "financial", "earnings", "sec", "10-k", "portfolio"],
            required_source_types=[
                _source("financial_filing", "Financial filings or official disclosures are required for financial claims."),
                _source("market_data", "Market and pricing claims require dated market data.", required=False, freshness_days=30),
                _source("company_disclosure", "Company disclosures support strategy and operating claims.", required=False, freshness_days=180),
            ],
            preferred_source_types=["financial_filing", "company_disclosure", "market_data", "regulator_guidance", "news"],
            disallowed_or_low_value_source_types=["community_discussion", "expert_analysis"],
            strictness="very_high",
            citation=CitationPolicy(strictness="primary_source_required", require_claim_level_citations=True, primary_source_required_for=["financial metric", "risk factor", "investment implication"]),
            freshness=FreshnessPolicy(strictness="current_required", max_age_days=90, require_publication_dates=True),
            synthesis=SynthesisPolicy(
                profile="risk_review",
                required_sections=["Scope", "Financial evidence", "Risk factors", "Uncertainties", "Human review"],
                forbidden_overclaims=["Do not provide investment advice.", "Do not predict returns as facts."],
                required_uncertainty_language=["Use risk-focused, informational language."],
            ),
            evaluation=_weights(coverage=0.18, citation_quality=0.30, source_quality=0.20, freshness=0.16, hallucination_risk=0.12, balance=0.04),
            safety=SafetyPolicy(
                conservative_language=True,
                professional_advice_disclaimer=True,
                require_human_review=True,
                safety_warnings=["This is financial risk information, not investment advice."],
                operator_review_required_when=["Any investment, accounting, tax, or trading decision will be made from the report."],
            ),
            review_when=["Investment or trading decision", "Material financial risk claim"],
        ),
        _protocol(
            protocol_id="medical_or_health_information_review",
            name="Medical Or Health Information Review",
            description="Reviews medical, health, clinical, drug, treatment, or public-health information.",
            matching_signals=["medical", "health", "disease", "symptom", "treatment", "drug", "clinical", "diagnosis", "therapy"],
            required_source_types=[
                _source("medical_guideline", "Medical claims should be anchored in guidelines or public-health authorities."),
                _source("clinical_source", "Clinical studies or authoritative clinical sources support treatment claims.", required=False),
                _source("systematic_review", "Systematic reviews are preferred for treatment efficacy claims.", required=False),
            ],
            preferred_source_types=["medical_guideline", "clinical_source", "systematic_review", "academic_paper"],
            disallowed_or_low_value_source_types=["community_discussion", "news", "expert_analysis"],
            strictness="very_high",
            citation=CitationPolicy(strictness="primary_source_required", require_claim_level_citations=True, primary_source_required_for=["diagnosis", "treatment", "safety", "dosage"]),
            freshness=FreshnessPolicy(strictness="current_required", max_age_days=730, require_publication_dates=True),
            synthesis=SynthesisPolicy(
                profile="risk_review",
                required_sections=["Scope", "Evidence", "Safety considerations", "Uncertainties", "Clinician review"],
                forbidden_overclaims=["Do not provide medical advice.", "Do not diagnose or recommend treatment decisions."],
                required_uncertainty_language=["Use informational language and recommend qualified clinical review."],
            ),
            evaluation=_weights(coverage=0.18, citation_quality=0.30, source_quality=0.22, freshness=0.12, hallucination_risk=0.14, balance=0.04),
            safety=SafetyPolicy(
                conservative_language=True,
                professional_advice_disclaimer=True,
                require_human_review=True,
                safety_warnings=["This is health information, not medical advice."],
                operator_review_required_when=["Any diagnosis, treatment, safety, medication, or clinical decision will be made from the report."],
            ),
            review_when=["Diagnosis, treatment, medication, safety, or clinical decision"],
        ),
        _protocol(
            protocol_id="news_or_current_events_review",
            name="News Or Current Events Review",
            description="Reviews current events, recent changes, developing stories, and time-sensitive public claims.",
            matching_signals=["latest", "current", "today", "recent", "news", "breaking", "this week", "2026"],
            required_source_types=[
                _source("news", "Current-events claims require dated reporting or official updates.", freshness_days=14),
                _source("primary_source", "Official statements or primary records should anchor decisive claims.", required=False, freshness_days=14),
            ],
            preferred_source_types=["primary_source", "news", "company_disclosure", "regulator_guidance"],
            disallowed_or_low_value_source_types=["community_discussion", "expert_analysis"],
            strictness="high",
            citation=CitationPolicy(strictness="strict", require_claim_level_citations=True, primary_source_required_for=["breaking development", "official action", "quote or numeric claim"]),
            freshness=FreshnessPolicy(strictness="current_required", max_age_days=14, require_publication_dates=True),
            synthesis=SynthesisPolicy(
                profile="deep_research_report",
                required_sections=["As of date", "What is known", "What is uncertain", "Source timeline", "Caveats"],
                forbidden_overclaims=["Do not treat developing reports as settled facts."],
                required_uncertainty_language=["State the research date and distinguish confirmed from reported claims."],
            ),
            evaluation=_weights(coverage=0.20, citation_quality=0.25, source_quality=0.16, freshness=0.22, balance=0.10, hallucination_risk=0.07),
            review_when=["Operational, legal, financial, health, or safety action based on current event"],
        ),
    ]
    return {protocol.protocol_id: protocol for protocol in protocols}


class ProtocolRegistry:
    def __init__(self, protocols: dict[str, ResearchProtocol] | None = None):
        self._protocols = protocols or built_in_protocols()

    def list_protocols(self) -> list[ResearchProtocol]:
        return [deepcopy(protocol) for protocol in sorted(self._protocols.values(), key=lambda item: item.protocol_id)]

    def get(self, protocol_id: str) -> ResearchProtocol:
        protocol = self._protocols.get(protocol_id)
        if protocol is None:
            raise UnknownProtocolError(protocol_id)
        return deepcopy(protocol)

    def ids(self) -> list[str]:
        return sorted(self._protocols)


DEFAULT_REGISTRY = ProtocolRegistry()


def get_protocol(protocol_id: str) -> ResearchProtocol:
    return DEFAULT_REGISTRY.get(protocol_id)

