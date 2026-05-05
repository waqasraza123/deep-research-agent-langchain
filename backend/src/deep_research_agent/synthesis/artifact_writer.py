from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import now_iso_utc

from .argument_map import build_argument_map
from .comparison_matrix import build_comparison_matrix
from .contracts import (
    ArgumentMap,
    ComparisonMatrix,
    DecisionMemo,
    ReportAssemblyPlan,
    SynthesisInput,
    SynthesisOutput,
    UncertaintyBoundary,
)
from .decision_memo import build_decision_memo
from .outline_builder import cluster_findings, extract_findings
from .report_assembler import assemble_report, build_report_assembly_plan, choose_report_profile
from .uncertainty import build_uncertainty_boundaries

SYNTHESIS_ARTIFACTS = (
    "synthesis_input.json",
    "synthesis_output.json",
    "findings.json",
    "finding_clusters.json",
    "argument_map.json",
    "argument_map.md",
    "comparison_matrix.json",
    "comparison_matrix.md",
    "decision_memo.json",
    "decision_memo.md",
    "uncertainty_boundaries.json",
    "uncertainty_boundaries.md",
    "report_assembly_plan.json",
    "report_assembly_plan.md",
)


def rebuild_synthesis_artifacts(
    run_dir: Path,
    *,
    thread_id: str,
    replace_report: bool = True,
) -> SynthesisOutput:
    synthesis_input = load_synthesis_input(run_dir, thread_id=thread_id)
    output = build_synthesis_output(synthesis_input)
    write_synthesis_artifacts(run_dir, output, synthesis_input, replace_report=replace_report)
    return output


def load_synthesis_input(run_dir: Path, *, thread_id: str) -> SynthesisInput:
    available = sorted(
        str(path.relative_to(run_dir)).replace("\\", "/")
        for path in run_dir.rglob("*")
        if path.is_file()
    )
    run_meta = _load_json(run_dir / "run.json") or _load_json(run_dir / ".run.json") or {}
    question = str(
        run_meta.get("question")
        or (run_meta.get("input_snapshot") or {}).get("question")
        or _question_from_metadata(run_dir)
        or ""
    )
    evidence_ledger = _load_json(run_dir / "evidence_ledger.json")
    source_audit = _load_json(run_dir / "source_audit.json")
    strategy = _load_json(run_dir / "strategy.json")
    subquestions = _load_json(run_dir / "subquestions.json")
    if not isinstance(subquestions, list):
        subquestions = (strategy or {}).get("subquestions") if isinstance(strategy, dict) else []
    if not isinstance(subquestions, list):
        subquestions = []

    return SynthesisInput(
        thread_id=thread_id,
        question=question,
        generated_at=now_iso_utc(),
        notes_text=_read_text(run_dir / "notes.md"),
        report_text=_read_text(run_dir / "report.md"),
        sources=_load_sources(run_dir / "sources.json"),
        evidence_ledger=evidence_ledger if isinstance(evidence_ledger, dict) else None,
        source_audit=source_audit if isinstance(source_audit, dict) else None,
        source_audit_text=_read_text(run_dir / "source_audit.md"),
        strategy=strategy if isinstance(strategy, dict) else None,
        subquestions=[item for item in subquestions if isinstance(item, dict)],
        available_artifacts=available,
    )


def build_synthesis_output(synthesis_input: SynthesisInput) -> SynthesisOutput:
    findings = extract_findings(synthesis_input)
    clusters = cluster_findings(findings)
    argument_map = build_argument_map(
        thread_id=synthesis_input.thread_id,
        question=synthesis_input.question,
        generated_at=synthesis_input.generated_at,
        findings=findings,
    )
    comparison_matrix = build_comparison_matrix(synthesis_input, findings)
    decision_memo = build_decision_memo(synthesis_input, findings)
    uncertainty = build_uncertainty_boundaries(synthesis_input, findings)
    profile = choose_report_profile(synthesis_input, comparison_matrix, decision_memo)
    assembly_plan = build_report_assembly_plan(
        synthesis_input,
        profile=profile,
        findings=findings,
        comparison_matrix=comparison_matrix,
        decision_memo=decision_memo,
        uncertainty=uncertainty,
    )
    report_markdown = assemble_report(
        synthesis_input,
        profile=profile,
        findings=findings,
        argument_map=argument_map,
        comparison_matrix=comparison_matrix,
        decision_memo=decision_memo,
        uncertainty=uncertainty,
        plan=assembly_plan,
    )
    warnings = (
        argument_map.warnings
        + comparison_matrix.warnings
        + decision_memo.warnings
        + uncertainty.warnings
        + assembly_plan.warnings
    )
    return SynthesisOutput(
        thread_id=synthesis_input.thread_id,
        generated_at=synthesis_input.generated_at,
        findings=findings,
        clusters=clusters,
        argument_map=argument_map,
        comparison_matrix=comparison_matrix,
        decision_memo=decision_memo,
        uncertainty_boundaries=uncertainty,
        report_assembly_plan=assembly_plan,
        assembled_report_markdown=report_markdown,
        warnings=warnings,
    )


def write_synthesis_artifacts(
    run_dir: Path,
    output: SynthesisOutput,
    synthesis_input: SynthesisInput,
    *,
    replace_report: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "synthesis_input.json", synthesis_input)
    _write_json(run_dir / "synthesis_output.json", output)
    _write_json(run_dir / "findings.json", output.findings)
    _write_json(run_dir / "finding_clusters.json", output.clusters)
    _write_json(run_dir / "argument_map.json", output.argument_map)
    _write_json(run_dir / "comparison_matrix.json", output.comparison_matrix)
    _write_json(run_dir / "decision_memo.json", output.decision_memo)
    _write_json(run_dir / "uncertainty_boundaries.json", output.uncertainty_boundaries)
    _write_json(run_dir / "report_assembly_plan.json", output.report_assembly_plan)

    (run_dir / "argument_map.md").write_text(
        render_argument_map_md(output.argument_map), encoding="utf-8"
    )
    (run_dir / "comparison_matrix.md").write_text(
        render_comparison_matrix_md(output.comparison_matrix), encoding="utf-8"
    )
    (run_dir / "decision_memo.md").write_text(
        render_decision_memo_md(output.decision_memo), encoding="utf-8"
    )
    (run_dir / "uncertainty_boundaries.md").write_text(
        render_uncertainty_boundaries_md(output.uncertainty_boundaries), encoding="utf-8"
    )
    (run_dir / "report_assembly_plan.md").write_text(
        render_report_assembly_plan_md(output.report_assembly_plan), encoding="utf-8"
    )

    if replace_report and output.report_assembly_plan.safe_to_replace_report:
        report_path = run_dir / "report.md"
        raw_path = run_dir / "report.raw.md"
        current = _read_text(report_path)
        if current and "GENERATED SYNTHESIS:" not in current and not raw_path.exists():
            raw_content = current if current.endswith("\n") else current + "\n"
            raw_path.write_text(raw_content, encoding="utf-8")
        report_path.write_text(output.assembled_report_markdown, encoding="utf-8")


def render_argument_map_md(argument_map: ArgumentMap) -> str:
    lines = _artifact_header("Argument Map", argument_map.generated_at)
    lines.extend(["## Main Answer", "", argument_map.main_answer, "", "## Nodes", ""])
    for node in argument_map.nodes:
        lines.extend(
            [
                f"### {node.node_id}",
                "",
                f"- Kind: `{node.kind}`",
                f"- Confidence: `{node.confidence_label}`",
                f"- Review required: {'yes' if node.requires_human_review else 'no'}",
                f"- Findings: {', '.join(f'`{fid}`' for fid in node.finding_ids) or 'none'}",
                f"- Sources: {', '.join(f'`{sid}`' for sid in node.source_ids) or 'none'}",
                "",
                node.text,
                "",
            ]
        )
    lines.extend(["## Relations", ""])
    if argument_map.relations:
        for relation in argument_map.relations:
            lines.append(
                f"- `{relation.source_node_id}` {relation.relation_type} "
                f"`{relation.target_node_id}`: {relation.rationale}"
            )
    else:
        lines.append("- No relations detected.")
    return "\n".join(lines).rstrip() + "\n"


def render_comparison_matrix_md(matrix: ComparisonMatrix) -> str:
    lines = _artifact_header("Comparison Matrix", matrix.generated_at)
    lines.extend(
        [
            f"- Detected: {'yes' if matrix.detected else 'no'}",
            f"- Options: {', '.join(matrix.options) or 'none'}",
            "",
        ]
    )
    if not matrix.detected:
        lines.append("Comparative intent was not detected.")
        return "\n".join(lines).rstrip() + "\n"
    if len(matrix.options) < 2:
        lines.append("Comparative intent was detected, but options were incomplete.")
        return "\n".join(lines).rstrip() + "\n"
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
    return "\n".join(lines).rstrip() + "\n"


def render_decision_memo_md(memo: DecisionMemo) -> str:
    lines = _artifact_header("Decision Memo", memo.generated_at)
    if not memo.detected:
        lines.append("Decision intent was not detected.")
        return "\n".join(lines).rstrip() + "\n"
    lines.extend(
        [
            "## Context",
            "",
            memo.context,
            "",
            "## Decision",
            "",
            memo.decision_to_make,
            "",
            "## Options",
            "",
        ]
    )
    if memo.options:
        lines.extend(f"- {option}" for option in memo.options)
    else:
        lines.append("- None inferred.")
    lines.extend(["", "## Recommendation", ""])
    if memo.recommendation:
        lines.extend(
            [
                f"- Stance: `{memo.recommendation.stance}`",
                f"- Option: {memo.recommendation.option or 'not selected'}",
                f"- Confidence: `{memo.recommendation.confidence_label}`",
                f"- Summary: {memo.recommendation.summary}",
            ]
        )
    else:
        lines.append("- No recommendation generated.")
    lines.extend(["", "## Rationale", ""])
    if memo.rationale:
        lines.extend(f"- {item}" for item in memo.rationale)
    else:
        lines.append("- No rationale extracted.")
    lines.extend(["", "## Risks", ""])
    if memo.risks:
        lines.extend(f"- {item}" for item in memo.risks)
    else:
        lines.append("- No explicit risks extracted.")
    lines.extend(
        [
            "",
            "## Reversibility",
            "",
            memo.reversibility,
            "",
            "## Cost / Complexity",
            "",
            memo.cost_complexity,
            "",
            "## Next Validation Steps",
            "",
        ]
    )
    lines.extend(f"- {step}" for step in memo.next_validation_steps)
    return "\n".join(lines).rstrip() + "\n"


def render_uncertainty_boundaries_md(boundary: UncertaintyBoundary) -> str:
    lines = _artifact_header("Uncertainty Boundaries", boundary.generated_at)
    sections = [
        ("Known", boundary.known),
        ("Likely", boundary.likely),
        ("Uncertain", boundary.uncertain),
        ("Not Verified", boundary.not_verified),
        ("Freshness Dependent", boundary.freshness_dependent),
        ("Requires Human Review", boundary.requires_human_review),
        ("Requires Primary Sources", boundary.requires_primary_sources),
    ]
    for title, items in sections:
        lines.extend([f"## {title}", ""])
        lines.extend(f"- {item}" for item in items) if items else lines.append("- None detected.")
        lines.append("")
    lines.extend(["## Open Questions", ""])
    if boundary.open_questions:
        for item in boundary.open_questions:
            lines.append(f"- `{item.question_id}` {item.text} ({item.reason})")
    else:
        lines.append("- None detected.")
    return "\n".join(lines).rstrip() + "\n"


def render_report_assembly_plan_md(plan: ReportAssemblyPlan) -> str:
    lines = _artifact_header("Report Assembly Plan", plan.generated_at)
    lines.extend(
        [
            f"- Profile: `{plan.profile}`",
            f"- Safe to replace report: {'yes' if plan.safe_to_replace_report else 'no'}",
            f"- Source artifacts: {', '.join(f'`{a}`' for a in plan.source_artifacts) or 'none'}",
            "",
            "## Sections",
            "",
        ]
    )
    lines.extend(f"- {section}" for section in plan.sections)
    lines.extend(["", "## Gaps", ""])
    lines.extend(f"- {gap}" for gap in plan.gaps) if plan.gaps else lines.append("- None detected.")
    if plan.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in plan.warnings)
    return "\n".join(lines).rstrip() + "\n"


def _artifact_header(title: str, generated_at: str) -> list[str]:
    return [
        f"# {title}",
        "",
        "> GENERATED SYNTHESIS: Deterministic backend artifact assembled from existing run files.",
        "",
        f"- Generated: `{generated_at}`",
        "",
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_dump(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dump(value: Any) -> Any:
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _load_json(path: Path) -> Any:
    if not path.exists() or path.is_dir():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_sources(path: Path) -> list[dict[str, Any]]:
    raw = _load_json(path)
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("sources"), list):
        return [item for item in raw["sources"] if isinstance(item, dict)]
    return []


def _question_from_metadata(run_dir: Path) -> str:
    data = _load_json(run_dir / "metadata.json")
    return str(data.get("question") or "") if isinstance(data, dict) else ""


def _read_text(path: Path, *, max_chars: int = 200_000) -> str:
    if not path.exists() or path.is_dir():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")[:300]
