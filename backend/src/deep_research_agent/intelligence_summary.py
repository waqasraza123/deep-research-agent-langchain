from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .artifacts import list_artifacts, now_iso_utc


class IntelligenceSummary(BaseModel):
    thread_id: str
    question: str = ""
    run_status: str = "unknown"
    generated_at: str
    memory_reused_or_suggested: list[dict[str, Any]] = Field(default_factory=list)
    task_graph_stages_executed: list[str] = Field(default_factory=list)
    source_quality_summary: dict[str, Any] = Field(default_factory=dict)
    top_sources: list[dict[str, Any]] = Field(default_factory=list)
    source_warnings: list[dict[str, Any]] = Field(default_factory=list)
    synthesis_outputs_generated: list[str] = Field(default_factory=list)
    evaluation_score: float | None = None
    hallucination_risk: dict[str, Any] = Field(default_factory=dict)
    coverage_gaps: list[dict[str, Any]] = Field(default_factory=list)
    recommended_operator_actions: list[str] = Field(default_factory=list)
    artifacts_generated: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def rebuild_intelligence_summary_artifacts(
    run_dir: Path,
    *,
    runs_dir: Path,
    thread_id: str,
) -> IntelligenceSummary:
    summary = build_intelligence_summary(run_dir, runs_dir=runs_dir, thread_id=thread_id)
    write_intelligence_summary_artifacts(run_dir, summary)
    return summary


def build_intelligence_summary(
    run_dir: Path,
    *,
    runs_dir: Path,
    thread_id: str,
) -> IntelligenceSummary:
    run_meta = _load_json(run_dir / "run.json") or _load_json(run_dir / ".run.json") or {}
    memory_context = _load_json(run_dir / "memory_context.json") or {}
    task_graph = _load_json(run_dir / "task_graph.json") or {}
    stage_outputs = _load_json(run_dir / "stage_outputs.json") or {}
    source_audit = _load_json(run_dir / "source_audit.json") or {}
    synthesis = _load_json(run_dir / "synthesis_output.json") or {}
    evaluation = _load_json(run_dir / "evaluation.json") or {}
    hallucination = _load_json(run_dir / "hallucination_risk.json") or {}
    coverage = _load_json(run_dir / "coverage_gaps.json")

    artifact_paths = [artifact.path for artifact in list_artifacts(runs_dir, thread_id)]
    source_summary = source_audit.get("summary") if isinstance(source_audit, dict) else {}
    source_summary = source_summary if isinstance(source_summary, dict) else {}
    audits = source_audit.get("audits") if isinstance(source_audit, dict) else []
    audits = [item for item in audits if isinstance(item, dict)]
    top_sources = sorted(
        audits,
        key=lambda item: float(item.get("final_source_score") or 0.0),
        reverse=True,
    )[:5]
    source_warnings = []
    for audit in audits:
        for warning in audit.get("warnings") or []:
            if isinstance(warning, dict):
                source_warnings.append({"source_id": audit.get("source_id"), **warning})

    memory_decisions = memory_context.get("suggested_source_reuse_candidates") or []
    if not isinstance(memory_decisions, list):
        memory_decisions = []

    stage_names = _stage_names(task_graph, stage_outputs)
    synthesis_outputs = [
        path
        for path in artifact_paths
        if path
        in {
            "argument_map.json",
            "comparison_matrix.json",
            "decision_memo.json",
            "uncertainty_boundaries.json",
            "synthesis_output.json",
            "report_assembly_plan.json",
        }
    ]
    coverage_gaps = coverage if isinstance(coverage, list) else []
    recommendations = _operator_actions(
        source_summary=source_summary,
        evaluation=evaluation,
        hallucination=hallucination,
        coverage_gaps=coverage_gaps,
        memory_decisions=memory_decisions,
        synthesis=synthesis,
    )

    question = str(
        run_meta.get("question") or (run_meta.get("input_snapshot") or {}).get("question") or ""
    )
    return IntelligenceSummary(
        thread_id=thread_id,
        question=question,
        run_status=str(run_meta.get("status") or "unknown"),
        generated_at=now_iso_utc(),
        memory_reused_or_suggested=memory_decisions,
        task_graph_stages_executed=stage_names,
        source_quality_summary=source_summary,
        top_sources=[
            {
                "source_id": item.get("source_id"),
                "url": item.get("url"),
                "title": item.get("title"),
                "score": item.get("final_source_score"),
                "recommended_usage": item.get("recommended_usage"),
            }
            for item in top_sources
        ],
        source_warnings=source_warnings[:20],
        synthesis_outputs_generated=synthesis_outputs,
        evaluation_score=evaluation.get("overall_score") if isinstance(evaluation, dict) else None,
        hallucination_risk=hallucination if isinstance(hallucination, dict) else {},
        coverage_gaps=[gap for gap in coverage_gaps if isinstance(gap, dict)][:20],
        recommended_operator_actions=recommendations,
        artifacts_generated=artifact_paths,
        warnings=_summary_warnings(memory_context, source_summary, evaluation, hallucination),
    )


def write_intelligence_summary_artifacts(run_dir: Path, summary: IntelligenceSummary) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = summary.model_dump(mode="json") if hasattr(summary, "model_dump") else summary.dict()
    (run_dir / "intelligence_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "intelligence_summary.md").write_text(
        render_intelligence_summary_md(summary),
        encoding="utf-8",
    )


def render_intelligence_summary_md(summary: IntelligenceSummary) -> str:
    lines = [
        "# Intelligence Summary",
        "",
        f"- Thread: `{summary.thread_id}`",
        f"- Status: `{summary.run_status}`",
        f"- Generated: `{summary.generated_at}`",
        f"- Evaluation score: {_fmt_score(summary.evaluation_score)}",
        f"- Hallucination risk: {_fmt_score(summary.hallucination_risk.get('risk_score'))}",
        "",
        "## Memory",
        "",
    ]
    if summary.memory_reused_or_suggested:
        for item in summary.memory_reused_or_suggested[:8]:
            lines.append(
                f"- {item.get('reuse_reason', 'memory_match')} "
                f"(confidence {_fmt_score(item.get('confidence_score'))})"
            )
    else:
        lines.append("- No prior memory reuse candidates.")

    lines.extend(["", "## Source Quality", ""])
    lines.append(
        f"- Sources: {summary.source_quality_summary.get('source_count', 0)}; "
        f"usable: {summary.source_quality_summary.get('usable_source_count', 0)}; "
        f"average score: {_fmt_score(summary.source_quality_summary.get('average_final_score'))}"
    )
    for source in summary.top_sources:
        lines.append(
            f"- `{source.get('source_id')}` {_fmt_score(source.get('score'))}: "
            f"{source.get('title') or source.get('url')} ({source.get('recommended_usage')})"
        )

    lines.extend(["", "## Orchestration", ""])
    lines.append(", ".join(f"`{stage}`" for stage in summary.task_graph_stages_executed) or "None")

    lines.extend(["", "## Synthesis And Evaluation", ""])
    lines.append(
        "- Synthesis outputs: "
        + (", ".join(f"`{path}`" for path in summary.synthesis_outputs_generated) or "none")
    )
    lines.append(f"- Coverage gaps: {len(summary.coverage_gaps)}")

    lines.extend(["", "## Operator Actions", ""])
    if summary.recommended_operator_actions:
        lines.extend(f"- {action}" for action in summary.recommended_operator_actions)
    else:
        lines.append("- No immediate deterministic action required.")

    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary.warnings)
    return "\n".join(lines).rstrip() + "\n"


def _load_json(path: Path) -> Any:
    try:
        if not path.exists() or path.is_dir():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stage_names(task_graph: Any, stage_outputs: Any) -> list[str]:
    names: list[str] = []
    outputs = (
        stage_outputs.get("stage_outputs") if isinstance(stage_outputs, dict) else stage_outputs
    )
    if isinstance(outputs, list):
        for item in outputs:
            if isinstance(item, dict):
                names.append(str(item.get("task_type") or item.get("stage") or item.get("task_id")))
    if names:
        return [name for name in names if name and name != "None"]
    tasks = task_graph.get("tasks") if isinstance(task_graph, dict) else []
    if isinstance(tasks, list):
        for item in tasks:
            if isinstance(item, dict):
                names.append(str(item.get("task_type") or item.get("task_id")))
    return [name for name in names if name and name != "None"]


def _operator_actions(
    *,
    source_summary: dict[str, Any],
    evaluation: Any,
    hallucination: Any,
    coverage_gaps: list[Any],
    memory_decisions: list[Any],
    synthesis: Any,
) -> list[str]:
    actions: list[str] = []
    if source_summary.get("sources_needing_verification"):
        actions.append("Verify cautionary sources with primary or official sources before citing.")
    if source_summary.get("sources_to_avoid"):
        actions.append("Do not cite sources marked for exclusion.")
    if isinstance(evaluation, dict) and float(evaluation.get("overall_score") or 0.0) < 0.65:
        actions.append("Review low-scoring research quality criteria before publishing.")
    if isinstance(hallucination, dict) and float(hallucination.get("risk_score") or 0.0) >= 0.45:
        actions.append("Resolve hallucination-risk findings against captured source text.")
    if coverage_gaps:
        actions.append("Close coverage gaps or explicitly document limitations.")
    if any(isinstance(item, dict) and item.get("freshness_warning") for item in memory_decisions):
        actions.append("Refresh stale memory suggestions before treating them as current context.")
    if isinstance(synthesis, dict) and synthesis.get("warnings"):
        actions.append("Review synthesis warnings and report assembly plan.")
    return list(dict.fromkeys(actions))


def _summary_warnings(
    memory_context: Any,
    source_summary: dict[str, Any],
    evaluation: Any,
    hallucination: Any,
) -> list[str]:
    warnings: list[str] = []
    if isinstance(memory_context, dict):
        warnings.extend(str(item) for item in memory_context.get("stale_warnings") or [])
    warnings.extend(str(item) for item in source_summary.get("citation_risks") or [])
    if isinstance(evaluation, dict):
        warnings.extend(str(item) for item in evaluation.get("warnings") or [])
    if isinstance(hallucination, dict) and hallucination.get("severity") in {"high", "critical"}:
        warnings.append("High hallucination risk detected.")
    return list(dict.fromkeys(warnings))[:30]


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"
