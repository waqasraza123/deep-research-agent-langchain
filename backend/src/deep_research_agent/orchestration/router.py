from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any

from deep_research_agent.intelligence.scoring import assess_complexity, detect_domains

from .contracts import (
    ExecutionDecision,
    ResearchTaskEdge,
    ResearchTaskGraph,
    ResearchTaskNode,
    ResearchTaskType,
    SpecialistRole,
)

COMPARISON_TERMS = {
    "compare",
    "comparison",
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

LEGAL_MEDICAL_FINANCIAL_POLICY_TERMS = {
    "legal",
    "law",
    "lawsuit",
    "regulation",
    "regulatory",
    "compliance",
    "policy",
    "jurisdiction",
    "contract",
    "terms",
    "medical",
    "clinical",
    "health",
    "diagnosis",
    "treatment",
    "financial",
    "finance",
    "investment",
    "invest",
    "roi",
    "tax",
}

TECHNICAL_TERMS = {
    "api",
    "sdk",
    "backend",
    "frontend",
    "database",
    "architecture",
    "implementation",
    "implement",
    "deploy",
    "deployment",
    "production",
    "scalability",
    "observability",
    "security",
    "failure mode",
    "failure modes",
    "code",
    "library",
    "framework",
}

DATE_SENSITIVE_TERMS = {
    "latest",
    "recent",
    "current",
    "today",
    "yesterday",
    "this week",
    "this month",
    "this year",
    "now",
    "2024",
    "2025",
    "2026",
}

BROAD_OPEN_TERMS = {
    "overview",
    "landscape",
    "deep dive",
    "research",
    "analyze",
    "analysis",
    "recommend",
    "recommendation",
    "should",
    "best",
    "evaluate",
}


@dataclass(frozen=True)
class RoutingContext:
    thread_id: str
    question: str
    urls: list[str]
    follow_links: bool = False
    max_links_per_source: int = 0
    available_source_count: int = 0
    strategy: dict[str, Any] | None = None


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9.+#-]*", text.lower()))


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text, flags=re.IGNORECASE))


def _has_any(question: str, terms: set[str]) -> bool:
    lowered = question.lower()
    tokens = _tokens(question)
    return any(term in tokens or _contains(lowered, term) for term in terms)


TASK_ROLES: dict[ResearchTaskType, SpecialistRole] = {
    ResearchTaskType.QUESTION_NORMALIZATION: SpecialistRole.PLANNER,
    ResearchTaskType.SOURCE_TRIAGE: SpecialistRole.SOURCE_TRIAGER,
    ResearchTaskType.SOURCE_EXTRACTION: SpecialistRole.SOURCE_TRIAGER,
    ResearchTaskType.EVIDENCE_COLLECTION: SpecialistRole.EVIDENCE_COLLECTOR,
    ResearchTaskType.SUBQUESTION_ANSWERING: SpecialistRole.DOMAIN_ANALYST,
    ResearchTaskType.CONTRADICTION_SCAN: SpecialistRole.SKEPTICAL_REVIEWER,
    ResearchTaskType.RISK_SCAN: SpecialistRole.RISK_REVIEWER,
    ResearchTaskType.SYNTHESIS: SpecialistRole.SYNTHESIS_WRITER,
    ResearchTaskType.CITATION_REVIEW: SpecialistRole.CITATION_AUDITOR,
    ResearchTaskType.FINAL_REPORT_REVIEW: SpecialistRole.SKEPTICAL_REVIEWER,
}


TASK_TITLES: dict[ResearchTaskType, str] = {
    ResearchTaskType.QUESTION_NORMALIZATION: "Question normalization",
    ResearchTaskType.SOURCE_TRIAGE: "Source triage",
    ResearchTaskType.SOURCE_EXTRACTION: "Source extraction",
    ResearchTaskType.EVIDENCE_COLLECTION: "Evidence collection",
    ResearchTaskType.SUBQUESTION_ANSWERING: "Subquestion answering",
    ResearchTaskType.CONTRADICTION_SCAN: "Contradiction scan",
    ResearchTaskType.RISK_SCAN: "Risk scan",
    ResearchTaskType.SYNTHESIS: "Synthesis",
    ResearchTaskType.CITATION_REVIEW: "Citation review",
    ResearchTaskType.FINAL_REPORT_REVIEW: "Final report review",
}


class AdaptiveRouter:
    def route(self, context: RoutingContext) -> ResearchTaskGraph:
        question = context.question.strip()
        urls = [u.strip() for u in context.urls if u and u.strip()]
        assessment = assess_complexity(question, urls)
        domains = detect_domains(question)

        has_comparison = _has_any(question, COMPARISON_TERMS)
        high_stakes = _has_any(question, LEGAL_MEDICAL_FINANCIAL_POLICY_TERMS)
        technical = _has_any(question, TECHNICAL_TERMS)
        date_sensitive = _has_any(question, DATE_SENSITIVE_TERMS) or assessment.freshness_required
        broad_open = _has_any(question, BROAD_OPEN_TERMS) or len(question.split()) > 22
        missing_urls = not urls
        has_sources = context.available_source_count > 0 or bool(urls)
        follow_links = bool(context.follow_links and context.max_links_per_source > 0)

        signals = list(assessment.signals)
        if has_comparison:
            signals.append("comparison_terms")
        if high_stakes:
            signals.append("legal_medical_financial_or_policy_terms")
        if technical:
            signals.append("technical_implementation_language")
        if date_sensitive:
            signals.append("date_sensitive_language")
        if broad_open:
            signals.append("broad_or_open_ended_phrasing")
        if missing_urls:
            signals.append("missing_urls")
        if follow_links:
            signals.append("follow_links_enabled")
        if context.available_source_count:
            signals.append(f"available_source_count:{context.available_source_count}")

        confidence_policy = "conservative" if high_stakes or date_sensitive else "standard"
        if technical and (has_comparison or broad_open):
            confidence_policy = "technical_due_diligence"

        decisions = self._decisions(
            has_comparison=has_comparison,
            high_stakes=high_stakes,
            technical=technical,
            date_sensitive=date_sensitive,
            broad_open=broad_open,
            missing_urls=missing_urls,
            has_sources=has_sources,
            follow_links=follow_links,
            available_source_count=context.available_source_count,
            complexity_level=assessment.level.value,
            confidence_policy=confidence_policy,
            signals=signals,
        )

        nodes = [
            ResearchTaskNode(
                id=self.node_id(task_type),
                task_type=task_type,
                specialist_role=TASK_ROLES[task_type],
                title=TASK_TITLES[task_type],
                description=self._description(task_type),
                depends_on=self._dependencies(task_type),
                decision=decisions[task_type],
                metadata={
                    "complexity_level": assessment.level.value,
                    "complexity_score": assessment.numeric_score,
                    "domains": domains,
                },
            )
            for task_type in ResearchTaskType
        ]
        edges = self._edges(nodes)
        graph_id = hashlib.sha1(
            (question + "\n" + "\n".join(urls) + "\n" + ",".join(signals)).encode("utf-8")
        ).hexdigest()[:12]
        return ResearchTaskGraph(
            graph_id=graph_id,
            thread_id=context.thread_id,
            question=question,
            urls=urls,
            nodes=nodes,
            edges=edges,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            routing_signals=sorted(set(signals)),
            confidence_policy=confidence_policy,
            metadata={
                "available_source_count": context.available_source_count,
                "follow_links": follow_links,
                "max_links_per_source": context.max_links_per_source,
                "complexity": assessment.dict()
                if not hasattr(assessment, "model_dump")
                else assessment.model_dump(mode="json"),
            },
        )

    @staticmethod
    def node_id(task_type: ResearchTaskType) -> str:
        return f"task_{task_type.value}"

    def _decisions(
        self,
        *,
        has_comparison: bool,
        high_stakes: bool,
        technical: bool,
        date_sensitive: bool,
        broad_open: bool,
        missing_urls: bool,
        has_sources: bool,
        follow_links: bool,
        available_source_count: int,
        complexity_level: str,
        confidence_policy: str,
        signals: list[str],
    ) -> dict[ResearchTaskType, ExecutionDecision]:
        run: dict[ResearchTaskType, tuple[bool, str]] = {
            ResearchTaskType.QUESTION_NORMALIZATION: (
                True,
                "Every run needs a normalized question and assumptions ledger.",
            ),
            ResearchTaskType.SOURCE_TRIAGE: (
                has_sources or missing_urls or has_comparison or high_stakes or date_sensitive,
                "Source quality or source gaps affect this run."
                if has_sources or missing_urls
                else "Comparison or high-stakes work requires source planning.",
            ),
            ResearchTaskType.SOURCE_EXTRACTION: (
                has_sources,
                "Provided or discovered sources need extraction guidance."
                if has_sources
                else "No URLs or available sources were supplied.",
            ),
            ResearchTaskType.EVIDENCE_COLLECTION: (
                has_sources or has_comparison or high_stakes or technical,
                "Claims need evidence mapping from available or expected sources."
                if has_sources
                else "No source-backed evidence can be collected yet.",
            ),
            ResearchTaskType.SUBQUESTION_ANSWERING: (
                has_comparison
                or broad_open
                or complexity_level in {"deep", "adversarial", "multi_domain"},
                "Complex, comparative, or broad questions need subquestion handling."
                if has_comparison or broad_open
                else "Simple focused question does not need decomposition.",
            ),
            ResearchTaskType.CONTRADICTION_SCAN: (
                has_comparison or high_stakes or complexity_level == "adversarial",
                "Comparative or high-stakes work needs a contrary-evidence pass."
                if has_comparison or high_stakes
                else "No contradiction-oriented signals were detected.",
            ),
            ResearchTaskType.RISK_SCAN: (
                high_stakes
                or (technical and (broad_open or has_comparison))
                or complexity_level == "adversarial",
                "High-stakes or technical due diligence requires risk and failure-mode review."
                if high_stakes or technical
                else "Risk scan not required for this scope.",
            ),
            ResearchTaskType.SYNTHESIS: (True, "Every run needs a synthesis contract."),
            ResearchTaskType.CITATION_REVIEW: (
                has_sources or has_comparison or high_stakes or date_sensitive,
                "Citations need review when sources, comparison, or freshness affect claims."
                if has_sources
                else "No citations can be audited before sources exist.",
            ),
            ResearchTaskType.FINAL_REPORT_REVIEW: (
                high_stakes
                or date_sensitive
                or (technical and (broad_open or has_comparison))
                or complexity_level in {"deep", "adversarial", "multi_domain"},
                "Conservative final review is required for high-risk, fresh, or deep research."
                if high_stakes or date_sensitive or technical
                else "Simple or moderate run can use standard synthesis review.",
            ),
        }
        if follow_links:
            should_run, reason = run[ResearchTaskType.SOURCE_TRIAGE]
            run[ResearchTaskType.SOURCE_TRIAGE] = (
                True,
                reason + " Link expansion is enabled, so triage must bound source priority.",
            )
            run[ResearchTaskType.SOURCE_EXTRACTION] = (
                True,
                "Link expansion is enabled; extraction must preserve root/discovered provenance.",
            )
        if available_source_count <= 0 and missing_urls:
            run[ResearchTaskType.CITATION_REVIEW] = (
                False,
                "Citation review skipped because no URLs or available sources exist.",
            )

        return {
            task_type: ExecutionDecision(
                node_id=self.node_id(task_type),
                task_type=task_type,
                specialist_role=TASK_ROLES[task_type],
                should_run=should_run,
                reason=reason,
                confidence_policy=confidence_policy,
                signals=sorted(set(signals)),
            )
            for task_type, (should_run, reason) in run.items()
        }

    @staticmethod
    def _description(task_type: ResearchTaskType) -> str:
        descriptions = {
            ResearchTaskType.QUESTION_NORMALIZATION: (
                "Normalize the question, identify implicit assumptions, and define scope."
            ),
            ResearchTaskType.SOURCE_TRIAGE: (
                "Rank source needs, source gaps, and source-quality risks before extraction."
            ),
            ResearchTaskType.SOURCE_EXTRACTION: "Plan source extraction with provenance, "
            "root/discovered source handling, and limits.",
            ResearchTaskType.EVIDENCE_COLLECTION: (
                "Convert source material into auditable evidence requirements and claim buckets."
            ),
            ResearchTaskType.SUBQUESTION_ANSWERING: (
                "Map the question into answerable subquestions and required comparisons."
            ),
            ResearchTaskType.CONTRADICTION_SCAN: (
                "Look for disagreement, missing counterevidence, and claims likely to conflict."
            ),
            ResearchTaskType.RISK_SCAN: "Identify high-stakes uncertainty, failure modes, "
            "and conservative confidence rules.",
            ResearchTaskType.SYNTHESIS: (
                "Assemble findings into a report contract with clear source boundaries."
            ),
            ResearchTaskType.CITATION_REVIEW: (
                "Audit citation expectations and evidence coverage requirements."
            ),
            ResearchTaskType.FINAL_REPORT_REVIEW: (
                "Review the final report contract for unresolved risk and unsupported conclusions."
            ),
        }
        return descriptions[task_type]

    @staticmethod
    def _dependencies(task_type: ResearchTaskType) -> list[str]:
        node = AdaptiveRouter.node_id
        deps = {
            ResearchTaskType.QUESTION_NORMALIZATION: [],
            ResearchTaskType.SOURCE_TRIAGE: [node(ResearchTaskType.QUESTION_NORMALIZATION)],
            ResearchTaskType.SOURCE_EXTRACTION: [node(ResearchTaskType.SOURCE_TRIAGE)],
            ResearchTaskType.EVIDENCE_COLLECTION: [node(ResearchTaskType.SOURCE_EXTRACTION)],
            ResearchTaskType.SUBQUESTION_ANSWERING: [node(ResearchTaskType.QUESTION_NORMALIZATION)],
            ResearchTaskType.CONTRADICTION_SCAN: [
                node(ResearchTaskType.EVIDENCE_COLLECTION),
                node(ResearchTaskType.SUBQUESTION_ANSWERING),
            ],
            ResearchTaskType.RISK_SCAN: [
                node(ResearchTaskType.EVIDENCE_COLLECTION),
                node(ResearchTaskType.SUBQUESTION_ANSWERING),
            ],
            ResearchTaskType.SYNTHESIS: [
                node(ResearchTaskType.EVIDENCE_COLLECTION),
                node(ResearchTaskType.SUBQUESTION_ANSWERING),
                node(ResearchTaskType.CONTRADICTION_SCAN),
                node(ResearchTaskType.RISK_SCAN),
            ],
            ResearchTaskType.CITATION_REVIEW: [node(ResearchTaskType.SYNTHESIS)],
            ResearchTaskType.FINAL_REPORT_REVIEW: [
                node(ResearchTaskType.SYNTHESIS),
                node(ResearchTaskType.CITATION_REVIEW),
            ],
        }
        return deps[task_type]

    @staticmethod
    def _edges(nodes: list[ResearchTaskNode]) -> list[ResearchTaskEdge]:
        by_id = {node.id: node for node in nodes}
        out: list[ResearchTaskEdge] = []
        for node in nodes:
            for dep in node.depends_on:
                out.append(
                    ResearchTaskEdge(
                        from_node=dep,
                        to_node=node.id,
                        reason=f"{by_id[dep].title} informs {node.title}.",
                        required=True,
                    )
                )
        return out
