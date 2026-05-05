from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from .contracts import (
    ResearchTaskStatus,
    ResearchTaskType,
    SpecialistRole,
    StageInput,
    StageOutput,
)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9.+#-]*", text)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _strategy_list(stage_input: StageInput, key: str) -> list[Any]:
    if not stage_input.strategy:
        return []
    value = stage_input.strategy.get(key)
    return value if isinstance(value, list) else []


def _instruction(title: str, bullets: list[str]) -> str:
    lines = [f"### {title}"]
    lines.extend(f"- {item}" for item in bullets if item)
    return "\n".join(lines).strip()


class SpecialistStage(ABC):
    task_type: ResearchTaskType
    role: SpecialistRole

    def run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        result = self._run(node_id, stage_input)
        result.confidence_score = max(0.0, min(1.0, result.confidence_score))
        return result

    @abstractmethod
    def _run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        raise NotImplementedError


class QuestionNormalizationSpecialist(SpecialistStage):
    task_type = ResearchTaskType.QUESTION_NORMALIZATION
    role = SpecialistRole.PLANNER

    def _run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        question = " ".join(stage_input.question.strip().split())
        findings = [f"Normalized question: {question}"]
        warnings: list[str] = []
        next_steps = ["Keep scope tied to the supplied question and captured sources."]
        word_count = len(_words(question))
        if not question.endswith("?"):
            warnings.append("Question is phrased as a task or statement; preserve that intent.")
        if word_count < 6:
            warnings.append("Question is short; assumptions should be explicit in the report.")
        if not stage_input.urls:
            warnings.append("No URLs were supplied; source limitations must be visible.")
            next_steps.append(
                "Ask the agent to state when claims are unsupported by fetched sources."
            )
        return StageOutput(
            node_id=node_id,
            task_type=self.task_type,
            specialist_role=self.role,
            stage_summary="Question normalized and initial scope risks recorded.",
            findings=findings,
            warnings=warnings,
            required_next_steps=next_steps,
            confidence_score=0.82 if word_count >= 6 else 0.65,
            artifact_updates={"normalized_question": question, "word_count": word_count},
            model_instruction_block=_instruction(
                "Planner scope",
                [
                    f"Answer this normalized research question: {question}",
                    "State assumptions and scope boundaries before making conclusions.",
                    "Do not introduce facts that are not grounded in fetched sources.",
                ],
            ),
        )


class SourceTriageSpecialist(SpecialistStage):
    task_type = ResearchTaskType.SOURCE_TRIAGE
    role = SpecialistRole.SOURCE_TRIAGER

    def _run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        source_count = len(stage_input.urls)
        findings = [
            f"Input URL count: {source_count}.",
            f"Available source count before agent work: {stage_input.available_source_count}.",
        ]
        warnings: list[str] = []
        if source_count == 0:
            warnings.append("No user-provided URLs; research must be treated as source-limited.")
        if stage_input.follow_links:
            findings.append(
                "One-hop link following enabled with max "
                f"{stage_input.max_links_per_source} links per source."
            )
        strategy_priorities = _strategy_list(stage_input, "source_priorities")
        if strategy_priorities:
            labels = [
                str(item.get("category") if isinstance(item, dict) else item)
                for item in strategy_priorities[:4]
            ]
            findings.append("Strategy source priorities: " + ", ".join(labels) + ".")
        next_steps = [
            "Prefer primary, official, or source-owning documents for conclusion-bearing claims.",
            "Preserve source IDs and root/discovered provenance in notes and sources.json.",
        ]
        return StageOutput(
            node_id=node_id,
            task_type=self.task_type,
            specialist_role=self.role,
            stage_summary="Source needs and source limitations triaged.",
            findings=findings,
            warnings=warnings,
            required_next_steps=next_steps,
            confidence_score=0.75 if source_count else 0.45,
            artifact_updates={
                "input_url_count": source_count,
                "source_limited": source_count == 0,
                "follow_links": stage_input.follow_links,
            },
            model_instruction_block=_instruction("Source triage", [*findings, *next_steps]),
        )


class SourceExtractionSpecialist(SpecialistStage):
    task_type = ResearchTaskType.SOURCE_EXTRACTION
    role = SpecialistRole.SOURCE_TRIAGER

    def _run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        findings = [
            "Extraction must keep fetched text separate from interpretation.",
            "Each extracted source should map to a stable source_id such as S1.",
        ]
        warnings: list[str] = []
        if stage_input.available_source_count < len(stage_input.urls):
            warnings.append(
                "Some supplied URLs may not be usable; report source fetch failures explicitly."
            )
        next_steps = [
            "Capture title, final URL, document kind, and local_path for every usable source.",
            "Do not cite pages that failed extraction as factual support.",
        ]
        return StageOutput(
            node_id=node_id,
            task_type=self.task_type,
            specialist_role=self.role,
            stage_summary="Source extraction contract prepared.",
            findings=findings,
            warnings=warnings,
            required_next_steps=next_steps,
            confidence_score=0.7 if stage_input.available_source_count else 0.55,
            artifact_updates={
                "expected_sources": len(stage_input.urls),
                "available_sources": stage_input.available_source_count,
            },
            model_instruction_block=_instruction("Source extraction", [*findings, *next_steps]),
        )


class EvidenceCollectionSpecialist(SpecialistStage):
    task_type = ResearchTaskType.EVIDENCE_COLLECTION
    role = SpecialistRole.EVIDENCE_COLLECTOR

    def _run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        requirements = _strategy_list(stage_input, "evidence_requirements")
        findings = [
            "Evidence should be grouped by claim area before synthesis.",
            f"Detected {len(requirements)} strategy evidence requirement(s).",
        ]
        for item in requirements[:5]:
            if isinstance(item, dict):
                findings.append(
                    f"Evidence requirement: {item.get('evidence_type')} - {item.get('description')}"
                )
        warnings: list[str] = []
        if stage_input.available_source_count == 1:
            warnings.append("Single-source evidence should not be presented as corroborated.")
        if stage_input.available_source_count == 0:
            warnings.append("No usable source count is available for evidence collection.")
        next_steps = [
            "Mark unsupported claims in notes rather than smoothing over missing evidence.",
            "Separate source facts, interpretation, and uncertainty.",
        ]
        return StageOutput(
            node_id=node_id,
            task_type=self.task_type,
            specialist_role=self.role,
            stage_summary="Evidence collection rules generated.",
            findings=findings,
            warnings=warnings,
            required_next_steps=next_steps,
            confidence_score=0.78 if stage_input.available_source_count >= 2 else 0.58,
            artifact_updates={"evidence_requirement_count": len(requirements)},
            model_instruction_block=_instruction("Evidence collection", [*findings, *next_steps]),
        )


class SubquestionAnsweringSpecialist(SpecialistStage):
    task_type = ResearchTaskType.SUBQUESTION_ANSWERING
    role = SpecialistRole.DOMAIN_ANALYST

    def _run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        subquestions = _strategy_list(stage_input, "subquestions")
        findings: list[str] = []
        if subquestions:
            for item in subquestions[:6]:
                if isinstance(item, dict):
                    findings.append(f"Subquestion: {item.get('question')}")
        else:
            sentences = _sentences(stage_input.question)
            findings = [f"Primary question segment: {s}" for s in sentences[:3]]
        next_steps = [
            "Answer each subquestion with its own evidence notes before writing the report.",
            "For comparisons, use the same criteria across each option.",
        ]
        return StageOutput(
            node_id=node_id,
            task_type=self.task_type,
            specialist_role=self.role,
            stage_summary="Subquestion work plan prepared.",
            findings=findings or ["Question can be handled as a single analytical unit."],
            warnings=[],
            required_next_steps=next_steps,
            confidence_score=0.76 if subquestions else 0.62,
            artifact_updates={"subquestion_count": len(subquestions)},
            model_instruction_block=_instruction("Domain analysis", [*(findings[:6]), *next_steps]),
        )


class ContradictionScanSpecialist(SpecialistStage):
    task_type = ResearchTaskType.CONTRADICTION_SCAN
    role = SpecialistRole.SKEPTICAL_REVIEWER

    def _run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        findings = [
            "Check whether sources disagree on definitions, timeframes, or evaluation criteria.",
            "Look for missing counterexamples before stating a preference.",
        ]
        warnings = []
        if stage_input.available_source_count < 2:
            warnings.append("Contradiction scan is weak with fewer than two usable sources.")
        next_steps = [
            "Include credible contrary evidence when it changes the answer.",
            "Avoid collapsing disagreement into a single confident recommendation.",
        ]
        return StageOutput(
            node_id=node_id,
            task_type=self.task_type,
            specialist_role=self.role,
            stage_summary="Contradiction and opposing-view checks prepared.",
            findings=findings,
            warnings=warnings,
            required_next_steps=next_steps,
            confidence_score=0.66 if stage_input.available_source_count >= 2 else 0.42,
            artifact_updates={"minimum_sources_for_strong_scan": 2},
            model_instruction_block=_instruction("Skeptical review", [*findings, *next_steps]),
        )


class RiskScanSpecialist(SpecialistStage):
    task_type = ResearchTaskType.RISK_SCAN
    role = SpecialistRole.RISK_REVIEWER

    def _run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        strategy_risks = _strategy_list(stage_input, "risk_flags")
        findings = [
            "High-stakes or technical claims require conservative wording.",
            "Identify operational, legal, financial, safety, or implementation failure "
            "modes when relevant.",
        ]
        findings.extend(str(flag) for flag in strategy_risks[:5])
        next_steps = [
            "Lower confidence when evidence is stale, indirect, single-source, or "
            "jurisdiction-specific.",
            "Do not provide legal, medical, or financial advice beyond source-grounded "
            "research context.",
        ]
        return StageOutput(
            node_id=node_id,
            task_type=self.task_type,
            specialist_role=self.role,
            stage_summary="Risk and conservative-confidence rules generated.",
            findings=findings,
            warnings=["Heuristic risk flags are not expert review."],
            required_next_steps=next_steps,
            confidence_score=0.64,
            artifact_updates={"risk_flag_count": len(strategy_risks)},
            model_instruction_block=_instruction("Risk review", [*findings, *next_steps]),
        )


class SynthesisSpecialist(SpecialistStage):
    task_type = ResearchTaskType.SYNTHESIS
    role = SpecialistRole.SYNTHESIS_WRITER

    def _run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        executed = [
            out.get("task_type")
            for out in stage_input.previous_outputs.values()
            if out.get("status") == ResearchTaskStatus.SUCCEEDED.value
        ]
        findings = [
            "Final report should lead with the answer, then evidence, caveats, and source limits.",
            f"Synthesis has {len(executed)} completed prior stage(s) available.",
        ]
        next_steps = [
            "Use only fetched sources and orchestration context.",
            "Keep conclusions proportional to evidence quality and source count.",
            "Cite conclusion-bearing claims with source IDs.",
        ]
        return StageOutput(
            node_id=node_id,
            task_type=self.task_type,
            specialist_role=self.role,
            stage_summary="Synthesis contract generated for the research agent.",
            findings=findings,
            warnings=[],
            required_next_steps=next_steps,
            confidence_score=0.8,
            artifact_updates={"prior_completed_stage_count": len(executed)},
            model_instruction_block=_instruction("Synthesis", [*findings, *next_steps]),
        )


class CitationReviewSpecialist(SpecialistStage):
    task_type = ResearchTaskType.CITATION_REVIEW
    role = SpecialistRole.CITATION_AUDITOR

    def _run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        citation_requirements = _strategy_list(stage_input, "citation_requirements")
        findings = citation_requirements[:5] if citation_requirements else [
            "Every factual claim that affects the answer needs a source citation.",
            "Citation IDs in report.md must correspond to sources.json entries.",
        ]
        warnings: list[str] = []
        if stage_input.available_source_count == 0:
            warnings.append(
                "No available sources; citation review can only enforce limitation wording."
            )
        next_steps = [
            "Review report.md for uncited factual claims after agent execution.",
            "Prefer explicit uncertainty over unsupported citation decoration.",
        ]
        return StageOutput(
            node_id=node_id,
            task_type=self.task_type,
            specialist_role=self.role,
            stage_summary="Citation audit criteria prepared.",
            findings=[str(item) for item in findings],
            warnings=warnings,
            required_next_steps=next_steps,
            confidence_score=0.72 if stage_input.available_source_count else 0.5,
            artifact_updates={"citation_requirement_count": len(citation_requirements)},
            model_instruction_block=_instruction(
                "Citation audit", [*(str(item) for item in findings), *next_steps]
            ),
        )


class FinalReportReviewSpecialist(SpecialistStage):
    task_type = ResearchTaskType.FINAL_REPORT_REVIEW
    role = SpecialistRole.SKEPTICAL_REVIEWER

    def _run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        findings = [
            "Final report must distinguish source facts from interpretation.",
            "High-impact recommendations need caveats and source-quality notes.",
        ]
        next_steps = [
            "Before completion, check report.md for overconfident claims.",
            "State source limitations in the conclusion when source coverage is thin.",
        ]
        return StageOutput(
            node_id=node_id,
            task_type=self.task_type,
            specialist_role=self.role,
            stage_summary="Final report review checklist prepared.",
            findings=findings,
            warnings=[],
            required_next_steps=next_steps,
            confidence_score=0.7,
            artifact_updates={"review_focus": "overconfidence_and_source_limits"},
            model_instruction_block=_instruction("Final review", [*findings, *next_steps]),
        )


def default_specialists() -> dict[ResearchTaskType, SpecialistStage]:
    stages: list[SpecialistStage] = [
        QuestionNormalizationSpecialist(),
        SourceTriageSpecialist(),
        SourceExtractionSpecialist(),
        EvidenceCollectionSpecialist(),
        SubquestionAnsweringSpecialist(),
        ContradictionScanSpecialist(),
        RiskScanSpecialist(),
        SynthesisSpecialist(),
        CitationReviewSpecialist(),
        FinalReportReviewSpecialist(),
    ]
    return {stage.task_type: stage for stage in stages}
