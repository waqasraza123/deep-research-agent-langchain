from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.evidence.contracts import EvidenceLedger
from deep_research_agent.evidence.ledger import load_source_documents

from .balance import assess_balance
from .contracts import (
    EvaluationRecommendation,
    ResearchEvaluation,
    model_to_plain,
)
from .coverage import detect_coverage_gaps
from .freshness_review import assess_freshness
from .hallucination_risk import assess_hallucination_risk
from .rubric import score_rubric

EVALUATION_ARTIFACTS = (
    "evaluation.json",
    "evaluation.md",
    "coverage_gaps.json",
    "coverage_gaps.md",
    "hallucination_risk.json",
    "hallucination_risk.md",
    "quality_score.json",
    "quality_score.md",
)


def rebuild_evaluation_artifacts(run_dir: Path, *, thread_id: str) -> ResearchEvaluation:
    evaluation = build_research_evaluation(run_dir, thread_id=thread_id)
    write_evaluation_artifacts(run_dir, evaluation)
    return evaluation


def build_research_evaluation(run_dir: Path, *, thread_id: str) -> ResearchEvaluation:
    generated_at = now_iso_utc()
    report_text = _read_report_text(run_dir)
    notes_text = _read_text(run_dir / "notes.md")
    sources = _load_sources(run_dir)
    evidence_ledger = _load_evidence_ledger(run_dir / "evidence_ledger.json")
    source_documents = load_source_documents(run_dir)
    source_texts = [doc.text for doc in source_documents if doc.text]
    artifacts_present = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    question = _load_question(run_dir)
    subquestions = _load_subquestions(run_dir)
    warnings: list[str] = []

    if not evidence_ledger:
        warnings.append("No evidence ledger was available; evaluation confidence is limited.")
    if not source_texts:
        warnings.append("No source text was available; source support could not be fully checked.")
    if not report_text.strip():
        warnings.append(
            "No report text was available; quality score is necessarily low-confidence."
        )

    coverage_gaps = detect_coverage_gaps(
        question=question,
        report_text=report_text,
        notes_text=notes_text,
        sources=sources,
        evidence_ledger=evidence_ledger,
        subquestions=subquestions,
    )
    hallucination_risk = assess_hallucination_risk(
        report_text=report_text,
        notes_text=notes_text,
        source_texts=source_texts,
        evidence_ledger=evidence_ledger,
    )
    balance = assess_balance(question=question, report_text=report_text, sources=sources)
    freshness = assess_freshness(question=question, report_text=report_text, sources=sources)
    rubric, criterion_scores, citation_quality, overall_score = score_rubric(
        question=question,
        report_text=report_text,
        notes_text=notes_text,
        sources=sources,
        artifacts_present=artifacts_present,
        coverage_gaps=coverage_gaps,
        hallucination_risk=hallucination_risk,
        balance=balance,
        freshness=freshness,
        evidence_ledger=evidence_ledger,
    )
    confidence = "high" if evidence_ledger and source_texts else "medium" if sources else "low"
    if confidence == "low":
        overall_score = min(overall_score, 0.55)
        warnings.append(
            "Low-confidence evaluations are capped and must not be treated as verified."
        )

    evaluation_id = "EV-" + hashlib.sha1(
        f"{thread_id}:{generated_at}:{overall_score}".encode("utf-8")
    ).hexdigest()[:12]
    recommendations = _recommendations(
        criterion_scores=criterion_scores,
        coverage_gaps=coverage_gaps,
        hallucination_count=len(hallucination_risk.findings),
    )
    return ResearchEvaluation(
        evaluation_id=evaluation_id,
        thread_id=thread_id,
        generated_at=generated_at,
        question=question,
        overall_score=overall_score,
        confidence=confidence,  # type: ignore[arg-type]
        rubric=rubric,
        criterion_scores=criterion_scores,
        coverage_gaps=coverage_gaps,
        hallucination_risk=hallucination_risk,
        balance=balance,
        freshness=freshness,
        citation_quality=citation_quality,
        recommendations=recommendations,
        warnings=warnings,
        artifacts_used=sorted(_evaluation_inputs_present(artifacts_present)),
    )


def write_evaluation_artifacts(run_dir: Path, evaluation: ResearchEvaluation) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "evaluation.json", model_to_plain(evaluation))
    (run_dir / "evaluation.md").write_text(render_evaluation_md(evaluation), encoding="utf-8")
    _write_json(
        run_dir / "coverage_gaps.json",
        [model_to_plain(gap) for gap in evaluation.coverage_gaps],
    )
    (run_dir / "coverage_gaps.md").write_text(render_coverage_gaps_md(evaluation), encoding="utf-8")
    _write_json(run_dir / "hallucination_risk.json", model_to_plain(evaluation.hallucination_risk))
    (run_dir / "hallucination_risk.md").write_text(
        render_hallucination_risk_md(evaluation), encoding="utf-8"
    )
    _write_json(run_dir / "quality_score.json", quality_score_payload(evaluation))
    (run_dir / "quality_score.md").write_text(render_quality_score_md(evaluation), encoding="utf-8")


def quality_score_payload(evaluation: ResearchEvaluation) -> dict[str, Any]:
    return {
        "thread_id": evaluation.thread_id,
        "evaluation_id": evaluation.evaluation_id,
        "generated_at": evaluation.generated_at,
        "overall_score": evaluation.overall_score,
        "confidence": evaluation.confidence,
        "hallucination_risk_score": evaluation.hallucination_risk.risk_score,
        "coverage_gap_count": len(evaluation.coverage_gaps),
        "citation_quality_score": evaluation.citation_quality.score,
        "freshness_score": evaluation.freshness.score,
        "balance_score": evaluation.balance.score,
        "criteria": {
            score.criterion_key: {
                "score": score.score,
                "severity": score.severity,
                "reasons": score.reasons,
            }
            for score in evaluation.criterion_scores
        },
        "warnings": evaluation.warnings,
    }


def render_evaluation_md(evaluation: ResearchEvaluation) -> str:
    lines = [
        "# Research Evaluation",
        "",
        f"- Thread: `{evaluation.thread_id}`",
        f"- Evaluation: `{evaluation.evaluation_id}`",
        f"- Generated: `{evaluation.generated_at}`",
        f"- Overall score: {evaluation.overall_score:.3f}",
        f"- Confidence: `{evaluation.confidence}`",
        f"- Method: `{evaluation.method}`",
        "",
    ]
    if evaluation.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in evaluation.warnings)
        lines.append("")
    lines.extend(["## Criteria", ""])
    for score in evaluation.criterion_scores:
        lines.extend(
            [
                f"### {score.criterion_key}",
                "",
                f"- Score: {score.score:.3f}",
                f"- Severity: `{score.severity}`",
                "- Reasons:",
            ]
        )
        lines.extend(f"  - {reason}" for reason in score.reasons)
        lines.extend(["- Suggested fix: " + score.suggested_fix, ""])
    if evaluation.recommendations:
        lines.extend(["## Recommendations", ""])
        for rec in evaluation.recommendations:
            lines.extend(
                [
                    f"### {rec.title}",
                    "",
                    f"- Priority: `{rec.priority}`",
                    f"- Rationale: {rec.rationale}",
                    f"- Action: {rec.suggested_action}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def render_coverage_gaps_md(evaluation: ResearchEvaluation) -> str:
    lines = ["# Coverage Gaps", ""]
    if not evaluation.coverage_gaps:
        lines.append("No coverage gaps detected by deterministic checks.")
        return "\n".join(lines) + "\n"
    for gap in evaluation.coverage_gaps:
        lines.extend(
            [
                f"## {gap.gap_id}",
                "",
                f"- Kind: `{gap.kind}`",
                f"- Severity: `{gap.severity}`",
                f"- Description: {gap.description}",
                f"- Suggested fix: {gap.suggested_fix}",
            ]
        )
        if gap.evidence:
            lines.append("- Evidence:")
            lines.extend(f"  - {item}" for item in gap.evidence[:5])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_hallucination_risk_md(evaluation: ResearchEvaluation) -> str:
    risk = evaluation.hallucination_risk
    lines = [
        "# Hallucination Risk",
        "",
        f"- Risk score: {risk.risk_score:.3f}",
        f"- Severity: `{risk.severity}`",
        f"- Confidence: `{risk.confidence}`",
        "",
    ]
    if not risk.findings:
        lines.append("No hallucination risk findings detected by deterministic checks.")
        return "\n".join(lines) + "\n"
    for finding in risk.findings:
        lines.extend(
            [
                f"## {finding.finding_id}",
                "",
                f"- Kind: `{finding.kind}`",
                f"- Severity: `{finding.severity}`",
                f"- Reason: {finding.reason}",
                f"- Text: {finding.text}",
                f"- Suggested fix: {finding.suggested_fix}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_quality_score_md(evaluation: ResearchEvaluation) -> str:
    lines = [
        "# Quality Score",
        "",
        f"- Overall score: {evaluation.overall_score:.3f}",
        f"- Confidence: `{evaluation.confidence}`",
        f"- Coverage gaps: {len(evaluation.coverage_gaps)}",
        f"- Hallucination risk: {evaluation.hallucination_risk.risk_score:.3f}",
        f"- Citation quality: {evaluation.citation_quality.score:.3f}",
        f"- Freshness: {evaluation.freshness.score:.3f}",
        f"- Balance: {evaluation.balance.score:.3f}",
        "",
        "## Lowest Criteria",
        "",
    ]
    for score in sorted(evaluation.criterion_scores, key=lambda item: item.score)[:5]:
        reason = score.reasons[0] if score.reasons else "No reason provided."
        lines.append(f"- `{score.criterion_key}` {score.score:.3f}: {reason}")
    return "\n".join(lines).rstrip() + "\n"


def _recommendations(
    *,
    criterion_scores: list[Any],
    coverage_gaps: list[Any],
    hallucination_count: int,
) -> list[EvaluationRecommendation]:
    recommendations: list[EvaluationRecommendation] = []
    for score in sorted(criterion_scores, key=lambda item: item.score)[:4]:
        if score.score >= 0.68:
            continue
        recommendations.append(
            EvaluationRecommendation(
                priority=score.severity,
                title=f"Improve {score.criterion_key.replace('_', ' ')}",
                rationale="; ".join(score.reasons[:2]),
                suggested_action=score.suggested_fix,
                affected_artifacts=score.affected_artifacts,
            )
        )
    high_gaps = [gap for gap in coverage_gaps if gap.severity in {"high", "critical"}]
    if high_gaps:
        recommendations.append(
            EvaluationRecommendation(
                priority="high",
                title="Close high-severity coverage gaps",
                rationale=f"{len(high_gaps)} high-severity coverage gap(s) detected.",
                suggested_action="Answer missing questions and add required source support.",
                affected_artifacts=["report.md", "sources.json"],
            )
        )
    if hallucination_count:
        recommendations.append(
            EvaluationRecommendation(
                priority="high",
                title="Reduce hallucination risk",
                rationale=f"{hallucination_count} hallucination risk finding(s) detected.",
                suggested_action="Remove unsupported values/entities or add source evidence.",
                affected_artifacts=["report.md", "evidence_ledger.json"],
            )
        )
    return recommendations[:8]


def _load_question(run_dir: Path) -> str:
    run_json = _load_json(run_dir / "run.json")
    if isinstance(run_json, dict):
        question = run_json.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()
        snapshot = run_json.get("input_snapshot")
        if isinstance(snapshot, dict) and isinstance(snapshot.get("question"), str):
            return snapshot["question"].strip()
    strategy = _load_json(run_dir / "strategy.json")
    if isinstance(strategy, dict) and isinstance(strategy.get("question"), str):
        return strategy["question"].strip()
    return ""


def _read_report_text(run_dir: Path) -> str:
    report_text = _read_text(run_dir / "report.md")
    if report_text.strip():
        return report_text
    synthesis = _load_json(run_dir / "synthesis_output.json")
    if isinstance(synthesis, dict):
        assembled = synthesis.get("assembled_report_markdown")
        if isinstance(assembled, str) and assembled.strip():
            return assembled
    return report_text


def _load_subquestions(run_dir: Path) -> list[str]:
    raw = _load_json(run_dir / "subquestions.json")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            for key in ("question", "text", "description"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    out.append(value.strip())
                    break
    return out


def _load_sources(run_dir: Path) -> list[dict[str, Any]]:
    raw = _load_json(run_dir / "sources.json")
    if isinstance(raw, list):
        return _merge_source_audit(run_dir, [item for item in raw if isinstance(item, dict)])
    if isinstance(raw, dict) and isinstance(raw.get("sources"), list):
        return _merge_source_audit(
            run_dir,
            [item for item in raw["sources"] if isinstance(item, dict)],
        )
    return []


def _merge_source_audit(run_dir: Path, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit_raw = _load_json(run_dir / "source_audit.json")
    if not isinstance(audit_raw, dict) or not isinstance(audit_raw.get("audits"), list):
        return sources
    audits = [item for item in audit_raw["audits"] if isinstance(item, dict)]
    by_id = {str(item.get("source_id")): item for item in audits if item.get("source_id")}
    by_url = {str(item.get("url")): item for item in audits if item.get("url")}
    merged: list[dict[str, Any]] = []
    for source in sources:
        copy = dict(source)
        audit = by_id.get(str(copy.get("source_id") or copy.get("id"))) or by_url.get(
            str(copy.get("final_url") or copy.get("url") or "")
        )
        if audit:
            copy["final_quality_score"] = audit.get("final_source_score")
            copy["source_audit"] = {
                "recommended_usage": audit.get("recommended_usage"),
                "authority_score": _nested_score(audit, "authority_score"),
                "credibility_score": _nested_score(audit, "credibility_score"),
                "freshness_score": _nested_score(audit, "freshness_score"),
                "bias_risk": audit.get("bias_risk_score", {}).get("risk_level")
                if isinstance(audit.get("bias_risk_score"), dict)
                else None,
            }
        merged.append(copy)
    return merged


def _nested_score(audit: dict[str, Any], key: str) -> float | None:
    value = audit.get(key)
    if isinstance(value, dict):
        raw = value.get("score")
        try:
            return float(raw)
        except Exception:
            return None
    return None


def _load_evidence_ledger(path: Path) -> EvidenceLedger | None:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        return None
    validate = getattr(EvidenceLedger, "model_validate", None)
    try:
        if callable(validate):
            return validate(raw)
        return EvidenceLedger.parse_obj(raw)
    except Exception:
        return None


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path, *, max_chars: int = 250_000) -> str:
    if not path.exists() or path.is_dir():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _evaluation_inputs_present(artifacts_present: set[str]) -> set[str]:
    inputs = {
        "report.md",
        "report.raw.md",
        "notes.md",
        "sources.json",
        "strategy.json",
        "subquestions.json",
        "evidence_ledger.json",
        "evidence_coverage.json",
        "source_graph.json",
        "source_audit.json",
        "synthesis_input.json",
        "synthesis_output.json",
        "findings.json",
        "argument_map.json",
        "comparison_matrix.json",
        "decision_memo.json",
        "uncertainty_boundaries.json",
    }
    return inputs & artifacts_present
