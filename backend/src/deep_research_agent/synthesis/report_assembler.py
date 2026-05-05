from __future__ import annotations

from .contracts import (
    ArgumentMap,
    ComparisonMatrix,
    DecisionMemo,
    ReportAssemblyPlan,
    ReportProfile,
    ResearchFinding,
    SynthesisInput,
    UncertaintyBoundary,
)


def choose_report_profile(
    synthesis_input: SynthesisInput,
    comparison_matrix: ComparisonMatrix,
    decision_memo: DecisionMemo,
) -> ReportProfile:
    strategy = synthesis_input.strategy or {}
    intent = str(strategy.get("intent") or "")
    if decision_memo.detected:
        return "decision_memo"
    if comparison_matrix.detected:
        return "comparative_report"
    if intent == "technical_due_diligence":
        return "technical_due_diligence"
    if intent == "risk_assessment":
        return "risk_review"
    if intent == "academic_literature_review":
        return "literature_style_review"
    return "deep_research_report"


def build_report_assembly_plan(
    synthesis_input: SynthesisInput,
    *,
    profile: ReportProfile,
    findings: list[ResearchFinding],
    comparison_matrix: ComparisonMatrix,
    decision_memo: DecisionMemo,
    uncertainty: UncertaintyBoundary,
) -> ReportAssemblyPlan:
    sections = _sections_for(profile, comparison_matrix, decision_memo)
    gaps: list[str] = []
    if not findings:
        gaps.append("No findings were extracted from available artifacts.")
    if not synthesis_input.evidence_ledger:
        gaps.append("Evidence ledger was unavailable; confidence is weaker.")
    if comparison_matrix.detected and len(comparison_matrix.options) < 2:
        gaps.append("Comparative profile lacks two clear options.")
    if decision_memo.detected and (decision_memo.recommendation is None):
        gaps.append("Decision profile lacks a recommendation.")
    if uncertainty.not_verified:
        gaps.append("Some findings are not verified.")

    substantive_findings = [
        finding
        for finding in findings
        if finding.origin in {"evidence_ledger", "notes", "report", "source_summary"}
        and finding.claim_type != "question"
    ]
    safe_to_replace = bool(substantive_findings) and not (
        len(substantive_findings) == len(uncertainty.open_questions) and not uncertainty.known
    )
    warnings = []
    if not safe_to_replace:
        warnings.append("Report replacement skipped because synthesis lacks enough findings.")

    return ReportAssemblyPlan(
        thread_id=synthesis_input.thread_id,
        generated_at=synthesis_input.generated_at,
        profile=profile,
        sections=sections,
        source_artifacts=[
            artifact
            for artifact in (
                "notes.md",
                "report.md",
                "sources.json",
                "evidence_ledger.json",
                "strategy.json",
                "subquestions.json",
                "source_audit.json",
                "source_audit.md",
            )
            if artifact in synthesis_input.available_artifacts
        ],
        safe_to_replace_report=safe_to_replace,
        gaps=gaps,
        warnings=warnings,
    )


def assemble_report(
    synthesis_input: SynthesisInput,
    *,
    profile: ReportProfile,
    findings: list[ResearchFinding],
    argument_map: ArgumentMap,
    comparison_matrix: ComparisonMatrix,
    decision_memo: DecisionMemo,
    uncertainty: UncertaintyBoundary,
    plan: ReportAssemblyPlan,
) -> str:
    lines = [
        "# Synthesized Research Report",
        "",
        "> GENERATED SYNTHESIS: Deterministic backend assembly from run artifacts only. "
        "No model was called for this synthesis report.",
        "",
        f"- Thread: `{synthesis_input.thread_id}`",
        f"- Profile: `{profile}`",
        f"- Generated: `{synthesis_input.generated_at}`",
        f"- Source artifacts: {', '.join(plan.source_artifacts) or 'none detected'}",
        "",
        "## Question",
        "",
        synthesis_input.question or "Question was not available in run metadata.",
        "",
    ]

    if profile == "concise_answer":
        _append_concise(lines, argument_map, findings, uncertainty)
    elif profile == "comparative_report":
        _append_deep_answer(lines, argument_map, findings)
        _append_comparison(lines, comparison_matrix)
        _append_uncertainty(lines, uncertainty)
    elif profile == "decision_memo":
        _append_decision(lines, decision_memo)
        _append_comparison(lines, comparison_matrix)
        _append_uncertainty(lines, uncertainty)
    elif profile == "technical_due_diligence":
        _append_deep_answer(lines, argument_map, findings)
        _append_risks(lines, decision_memo, uncertainty)
        _append_traceability(lines, findings)
        _append_uncertainty(lines, uncertainty)
    elif profile == "risk_review":
        _append_risks(lines, decision_memo, uncertainty)
        _append_deep_answer(lines, argument_map, findings)
        _append_uncertainty(lines, uncertainty)
    elif profile == "literature_style_review":
        _append_deep_answer(lines, argument_map, findings)
        _append_traceability(lines, findings)
        _append_uncertainty(lines, uncertainty)
    else:
        _append_deep_answer(lines, argument_map, findings)
        _append_traceability(lines, findings)
        _append_uncertainty(lines, uncertainty)

    if plan.gaps:
        lines.extend(["", "## Gaps", ""])
        lines.extend(f"- {gap}" for gap in plan.gaps)

    lines.extend(
        [
            "",
            "## Synthesis Artifacts",
            "",
            "- `argument_map.json` / `argument_map.md`",
            "- `comparison_matrix.json` / `comparison_matrix.md`",
            "- `decision_memo.json` / `decision_memo.md`",
            "- `uncertainty_boundaries.json` / `uncertainty_boundaries.md`",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _append_concise(
    lines: list[str],
    argument_map: ArgumentMap,
    findings: list[ResearchFinding],
    uncertainty: UncertaintyBoundary,
) -> None:
    lines.extend(["## Answer", "", argument_map.main_answer, "", "## Key Findings", ""])
    for finding in findings[:6]:
        lines.append(_finding_bullet(finding))
    if uncertainty.uncertain or uncertainty.not_verified:
        lines.extend(["", "## Caveats", ""])
        for item in (uncertainty.uncertain + uncertainty.not_verified)[:5]:
            lines.append(f"- {item}")


def _append_deep_answer(
    lines: list[str],
    argument_map: ArgumentMap,
    findings: list[ResearchFinding],
) -> None:
    lines.extend(["## Answer", "", argument_map.main_answer, "", "## Findings", ""])
    if not findings:
        lines.append("- No findings were extracted from the available artifacts.")
        return
    for finding in findings[:14]:
        lines.append(_finding_bullet(finding))

    supporting = [node for node in argument_map.nodes if node.kind == "supporting_claim"]
    weak = [node for node in argument_map.nodes if node.kind == "weak_evidence"]
    counter = [node for node in argument_map.nodes if node.kind == "counterclaim"]
    if supporting:
        lines.extend(["", "## Supporting Claims", ""])
        lines.extend(f"- {node.text}" for node in supporting[:8])
    if counter:
        lines.extend(["", "## Counterclaims", ""])
        lines.extend(f"- {node.text}" for node in counter[:6])
    if weak:
        lines.extend(["", "## Weak Evidence", ""])
        lines.extend(f"- {node.text}" for node in weak[:8])


def _append_comparison(lines: list[str], matrix: ComparisonMatrix) -> None:
    if not matrix.detected:
        return
    lines.extend(["", "## Comparison Matrix", ""])
    if len(matrix.options) < 2:
        lines.append("- Comparative intent was detected, but options were incomplete.")
        return
    header = "| Dimension | " + " | ".join(matrix.options) + " |"
    divider = "|---|" + "|".join("---" for _ in matrix.options) + "|"
    lines.extend([header, divider])
    cells = {(cell.dimension_id, cell.option): cell for cell in matrix.cells}
    for dimension in matrix.dimensions:
        row = [dimension.label]
        for option in matrix.options:
            cell = cells.get((dimension.dimension_id, option))
            row.append(_escape_table(cell.summary if cell else "No finding."))
        lines.append("| " + " | ".join(row) + " |")


def _append_decision(lines: list[str], memo: DecisionMemo) -> None:
    if not memo.detected:
        return
    lines.extend(["## Decision Memo", "", f"**Decision:** {memo.decision_to_make}", ""])
    if memo.options:
        lines.extend(["**Options:**", ""])
        lines.extend(f"- {option}" for option in memo.options)
        lines.append("")
    if memo.recommendation:
        lines.extend(
            [
                "**Recommendation:**",
                "",
                f"- Stance: `{memo.recommendation.stance}`",
                f"- Option: {memo.recommendation.option or 'not selected'}",
                f"- Summary: {memo.recommendation.summary}",
                f"- Confidence: `{memo.recommendation.confidence_label}`",
                "",
            ]
        )
    if memo.rationale:
        lines.extend(["**Rationale:**", ""])
        lines.extend(f"- {item}" for item in memo.rationale[:8])
        lines.append("")
    if memo.risks:
        lines.extend(["**Risks:**", ""])
        lines.extend(f"- {risk}" for risk in memo.risks[:8])
        lines.append("")
    lines.extend(
        [
            f"**Reversibility:** {memo.reversibility}",
            "",
            f"**Cost / Complexity:** {memo.cost_complexity}",
            "",
            "**Next Validation Steps:**",
            "",
        ]
    )
    lines.extend(f"- {step}" for step in memo.next_validation_steps)


def _append_risks(
    lines: list[str],
    memo: DecisionMemo,
    uncertainty: UncertaintyBoundary,
) -> None:
    lines.extend(["## Risk Review", ""])
    risks = memo.risks + uncertainty.uncertain + uncertainty.not_verified
    if not risks:
        lines.append("- No explicit risks were extracted from available artifacts.")
        return
    for risk in risks[:12]:
        lines.append(f"- {risk}")


def _append_traceability(lines: list[str], findings: list[ResearchFinding]) -> None:
    lines.extend(["", "## Traceability", ""])
    if not findings:
        lines.append("- No traceable findings available.")
        return
    for finding in findings[:20]:
        refs = ", ".join(f"`{ref}`" for ref in finding.artifact_refs) or "`unknown`"
        sources = ", ".join(f"`{sid}`" for sid in finding.source_ids) or "`none`"
        lines.append(
            f"- `{finding.finding_id}` from {refs}; sources: {sources}; "
            f"confidence: `{finding.confidence_label}`."
        )


def _append_uncertainty(lines: list[str], uncertainty: UncertaintyBoundary) -> None:
    lines.extend(["", "## Uncertainty Boundaries", ""])
    groups = [
        ("Known", uncertainty.known),
        ("Likely", uncertainty.likely),
        ("Uncertain", uncertainty.uncertain),
        ("Not Verified", uncertainty.not_verified),
        ("Freshness Dependent", uncertainty.freshness_dependent),
        ("Requires Human Review", uncertainty.requires_human_review),
        ("Requires Primary Sources", uncertainty.requires_primary_sources),
    ]
    for label, items in groups:
        lines.extend([f"### {label}", ""])
        if items:
            lines.extend(f"- {item}" for item in items[:8])
        else:
            lines.append("- None detected from available artifacts.")
        lines.append("")
    if uncertainty.open_questions:
        lines.extend(["### Open Questions", ""])
        lines.extend(f"- {item.text}" for item in uncertainty.open_questions[:8])


def _sections_for(
    profile: ReportProfile,
    comparison_matrix: ComparisonMatrix,
    decision_memo: DecisionMemo,
) -> list[str]:
    if profile == "concise_answer":
        return ["Question", "Answer", "Key Findings", "Caveats"]
    if profile == "decision_memo" or decision_memo.detected:
        return ["Question", "Decision Memo", "Comparison Matrix", "Uncertainty Boundaries", "Gaps"]
    if profile == "comparative_report" or comparison_matrix.detected:
        return ["Question", "Answer", "Findings", "Comparison Matrix", "Uncertainty Boundaries"]
    if profile == "risk_review":
        return ["Question", "Risk Review", "Answer", "Uncertainty Boundaries"]
    return ["Question", "Answer", "Findings", "Traceability", "Uncertainty Boundaries", "Gaps"]


def _finding_bullet(finding: ResearchFinding) -> str:
    sources = ", ".join(f"[{sid}]" for sid in finding.source_ids)
    suffix = f" {sources}" if sources else ""
    return f"- {finding.text}{suffix} (`{finding.confidence_label}`)"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")[:300]
