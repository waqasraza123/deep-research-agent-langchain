from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResearchIntent(str, Enum):
    FACTUAL_SUMMARY = "factual_summary"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TECHNICAL_DUE_DILIGENCE = "technical_due_diligence"
    MARKET_RESEARCH = "market_research"
    LEGAL_OR_POLICY_REVIEW = "legal_or_policy_review"
    ACADEMIC_LITERATURE_REVIEW = "academic_literature_review"
    IMPLEMENTATION_RESEARCH = "implementation_research"
    RISK_ASSESSMENT = "risk_assessment"
    UNKNOWN = "unknown"


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    DEEP = "deep"
    ADVERSARIAL = "adversarial"
    MULTI_DOMAIN = "multi_domain"


class EvidenceType(str, Enum):
    PRIMARY_SOURCE = "primary_source"
    OFFICIAL_DOCUMENTATION = "official_documentation"
    SOURCE_CODE = "source_code"
    IMPLEMENTATION_EXAMPLE = "implementation_example"
    BENCHMARK_OR_METRIC = "benchmark_or_metric"
    ACADEMIC_PAPER = "academic_paper"
    LEGAL_OR_POLICY_TEXT = "legal_or_policy_text"
    MARKET_DATA = "market_data"
    EXPERT_ANALYSIS = "expert_analysis"
    RISK_OR_INCIDENT_RECORD = "risk_or_incident_record"
    CURRENT_SOURCE = "current_source"


class SourceCategory(str, Enum):
    OFFICIAL_DOCS = "official_docs"
    SOURCE_CODE = "source_code"
    RELEASE_NOTES = "release_notes"
    BENCHMARKS = "benchmarks"
    ACADEMIC_DATABASES = "academic_databases"
    LEGAL_DATABASES = "legal_databases"
    REGULATOR_OR_POLICY_SITES = "regulator_or_policy_sites"
    COMPANY_OR_PRODUCT_PAGES = "company_or_product_pages"
    MARKET_REPORTS = "market_reports"
    NEWS_OR_DISCLOSURES = "news_or_disclosures"
    COMMUNITY_DISCUSSION = "community_discussion"


class SubQuestion(BaseModel):
    id: str = Field(..., min_length=2)
    question: str = Field(..., min_length=8)
    rationale: str = Field(..., min_length=8)
    priority: int = Field(default=3, ge=1, le=5)
    evidence_types: list[EvidenceType] = Field(default_factory=list)
    source_categories: list[SourceCategory] = Field(default_factory=list)
    opposing_view_required: bool = False


class EvidenceRequirement(BaseModel):
    evidence_type: EvidenceType
    description: str = Field(..., min_length=8)
    priority: int = Field(default=3, ge=1, le=5)
    minimum_sources: int = Field(default=1, ge=1, le=5)


class SourcePriority(BaseModel):
    category: SourceCategory
    rationale: str = Field(..., min_length=8)
    priority: int = Field(default=3, ge=1, le=5)


class VerificationStep(BaseModel):
    claim_area: str = Field(..., min_length=3)
    method: str = Field(..., min_length=8)
    priority: int = Field(default=3, ge=1, le=5)
    required_sources: list[SourceCategory] = Field(default_factory=list)


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


class ResearchStrategy(BaseModel):
    research_id: str = Field(..., min_length=8)
    normalized_question: str = Field(..., min_length=5)
    intent: ResearchIntent
    complexity_score: ComplexityLevel
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    subquestions: list[SubQuestion] = Field(default_factory=list)
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list)
    source_priorities: list[SourcePriority] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    suggested_artifacts: list[str] = Field(default_factory=list)
    agent_instructions: str = Field(..., min_length=20)
    citation_requirements: list[str] = Field(default_factory=list)
    verification_plan: list[VerificationStep] = Field(default_factory=list)
    required_definitions: list[str] = Field(default_factory=list)
    assumptions_to_verify: list[str] = Field(default_factory=list)
    possible_opposing_views: list[str] = Field(default_factory=list)
    missing_information_warnings: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return _model_to_dict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_json_dict(), ensure_ascii=False, indent=indent) + "\n"

    def to_markdown(self) -> str:
        lines = [
            "# Research Strategy",
            "",
            f"- Research ID: `{self.research_id}`",
            f"- Intent: `{self.intent.value}`",
            f"- Complexity: `{self.complexity_score.value}`",
            f"- Confidence: `{self.confidence_score:.2f}`",
            "",
            "## Question",
            "",
            self.normalized_question,
            "",
            "## Subquestions",
            "",
        ]
        for sq in self.subquestions:
            lines.append(f"{sq.id}. {sq.question}")
            lines.append(f"   - Rationale: {sq.rationale}")
            evidence = ", ".join(e.value for e in sq.evidence_types) or "not specified"
            lines.append(f"   - Evidence: {evidence}")
            lines.append("")

        lines.extend(["## Evidence Requirements", ""])
        for requirement in self.evidence_requirements:
            lines.append(
                f"- P{requirement.priority} `{requirement.evidence_type.value}`: "
                f"{requirement.description} (minimum sources: {requirement.minimum_sources})"
            )

        lines.extend(["", "## Source Priorities", ""])
        for priority in self.source_priorities:
            lines.append(
                f"- P{priority.priority} `{priority.category.value}`: {priority.rationale}"
            )

        lines.extend(["", "## Verification Plan", ""])
        for step in self.verification_plan:
            cats = ", ".join(c.value for c in step.required_sources) or "source-appropriate"
            lines.append(f"- P{step.priority} {step.claim_area}: {step.method} Sources: {cats}.")

        if self.risk_flags:
            lines.extend(["", "## Risk Flags", ""])
            lines.extend(f"- {flag}" for flag in self.risk_flags)

        if self.missing_information_warnings:
            lines.extend(["", "## Missing Information Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.missing_information_warnings)

        lines.extend(["", "## Agent Instructions", "", self.agent_instructions.strip(), ""])
        return "\n".join(lines)
