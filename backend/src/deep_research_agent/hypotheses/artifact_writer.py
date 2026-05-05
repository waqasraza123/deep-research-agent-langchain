from __future__ import annotations

import json
from pathlib import Path

from .confidence import update_confidences
from .contracts import (
    ConfidenceLevel,
    HypothesisSet,
    HypothesisStatus,
    HypothesisSummary,
    model_to_plain,
)
from .contradiction_mapper import map_contradictions
from .evidence_tester import test_hypotheses
from .generator import generate_hypotheses, load_hypothesis_input
from .hypothesis_graph import build_hypothesis_graph

HYPOTHESIS_ARTIFACTS = (
    "hypotheses.json",
    "hypotheses.md",
    "hypothesis_tests.json",
    "hypothesis_tests.md",
    "hypothesis_graph.json",
    "hypothesis_graph.md",
    "confidence_updates.json",
    "confidence_updates.md",
)


def rebuild_hypothesis_artifacts(run_dir: Path, *, thread_id: str) -> HypothesisSet:
    build_input = load_hypothesis_input(run_dir, thread_id=thread_id)
    hypothesis_set = generate_hypotheses(build_input)
    hypothesis_set = test_hypotheses(hypothesis_set, build_input, run_dir)
    hypothesis_set = map_contradictions(hypothesis_set, build_input)
    hypothesis_set = update_confidences(hypothesis_set, build_input)
    hypothesis_set.summary = build_hypothesis_summary(hypothesis_set)
    hypothesis_set.graph = build_hypothesis_graph(hypothesis_set)
    write_hypothesis_artifacts(run_dir, hypothesis_set)
    return hypothesis_set


def write_hypothesis_artifacts(run_dir: Path, hypothesis_set: HypothesisSet) -> list[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    graph = hypothesis_set.graph
    files = {
        "hypotheses.json": _json_text(hypothesis_set),
        "hypotheses.md": render_hypotheses_md(hypothesis_set),
        "hypothesis_tests.json": _json_text(
            {
                "thread_id": hypothesis_set.thread_id,
                "generated_at": hypothesis_set.generated_at,
                "results": model_to_plain(hypothesis_set.test_results),
            }
        ),
        "hypothesis_tests.md": render_hypothesis_tests_md(hypothesis_set),
        "hypothesis_graph.json": _json_text(graph or {}),
        "hypothesis_graph.md": render_hypothesis_graph_md(hypothesis_set),
        "confidence_updates.json": _json_text(
            {
                "thread_id": hypothesis_set.thread_id,
                "generated_at": hypothesis_set.generated_at,
                "updates": model_to_plain(hypothesis_set.confidence_updates),
            }
        ),
        "confidence_updates.md": render_confidence_updates_md(hypothesis_set),
    }
    for rel_path, content in files.items():
        (run_dir / rel_path).write_text(content, encoding="utf-8")
    return sorted(files)


def build_hypothesis_summary(hypothesis_set: HypothesisSet) -> HypothesisSummary:
    counts = {status: 0 for status in HypothesisStatus}
    for hypothesis in hypothesis_set.hypotheses:
        counts[hypothesis.status] += 1
    updates = {item.hypothesis_id: item for item in hypothesis_set.confidence_updates}
    average = (
        sum(item.posterior_score for item in hypothesis_set.confidence_updates)
        / len(hypothesis_set.confidence_updates)
        if hypothesis_set.confidence_updates
        else 0.0
    )
    ranked = sorted(
        hypothesis_set.hypotheses,
        key=lambda hyp: updates.get(hyp.hypothesis_id).posterior_score
        if updates.get(hyp.hypothesis_id)
        else 0.0,
        reverse=True,
    )
    unresolved: list[str] = []
    for result in hypothesis_set.test_results:
        unresolved.extend(result.unresolved)
    warnings = list(hypothesis_set.summary.warnings)
    if average < 0.5:
        warnings.append("Average hypothesis confidence is below medium; conclusions are tentative.")
    if counts[HypothesisStatus.CONTRADICTED]:
        warnings.append("At least one hypothesis is contradicted by available evidence.")
    if counts[HypothesisStatus.NEEDS_MORE_EVIDENCE]:
        warnings.append("At least one hypothesis requires more evidence.")
    return HypothesisSummary(
        thread_id=hypothesis_set.thread_id,
        generated_at=hypothesis_set.generated_at,
        total_hypotheses=len(hypothesis_set.hypotheses),
        supported=counts[HypothesisStatus.SUPPORTED],
        partially_supported=counts[HypothesisStatus.PARTIALLY_SUPPORTED],
        contradicted=counts[HypothesisStatus.CONTRADICTED],
        unsupported=counts[HypothesisStatus.UNSUPPORTED],
        inconclusive=counts[HypothesisStatus.INCONCLUSIVE],
        needs_more_evidence=counts[HypothesisStatus.NEEDS_MORE_EVIDENCE],
        average_confidence=round(average, 3),
        strongest_hypothesis_ids=[hyp.hypothesis_id for hyp in ranked[:3]],
        weakest_hypothesis_ids=[hyp.hypothesis_id for hyp in reversed(ranked[-3:])],
        unresolved_questions=sorted(dict.fromkeys(unresolved))[:12],
        warnings=sorted(dict.fromkeys(warnings)),
    )


def render_hypotheses_md(hypothesis_set: HypothesisSet) -> str:
    updates = {item.hypothesis_id: item for item in hypothesis_set.confidence_updates}
    lines = [
        "# Hypotheses",
        "",
        f"- Thread: `{hypothesis_set.thread_id}`",
        f"- Generated: `{hypothesis_set.generated_at}`",
        f"- Total hypotheses: {hypothesis_set.summary.total_hypotheses}",
        f"- Average confidence: {hypothesis_set.summary.average_confidence:.3f}",
        "",
        "## Summary",
        "",
        f"- Supported: {hypothesis_set.summary.supported}",
        f"- Partially supported: {hypothesis_set.summary.partially_supported}",
        f"- Contradicted: {hypothesis_set.summary.contradicted}",
        f"- Unsupported: {hypothesis_set.summary.unsupported}",
        f"- Inconclusive: {hypothesis_set.summary.inconclusive}",
        f"- Needs more evidence: {hypothesis_set.summary.needs_more_evidence}",
        "",
    ]
    if hypothesis_set.summary.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in hypothesis_set.summary.warnings)
        lines.append("")
    lines.extend(["## Hypotheses", ""])
    for hypothesis in hypothesis_set.hypotheses:
        update = updates.get(hypothesis.hypothesis_id)
        level = update.confidence_level.value if update else ConfidenceLevel.VERY_LOW.value
        posterior = update.posterior_score if update else 0.0
        lines.extend(
            [
                f"### {hypothesis.hypothesis_id}",
                "",
                f"- Type: `{hypothesis.hypothesis_type.value}`",
                f"- Status: `{hypothesis.status.value}`",
                f"- Confidence: `{level}` ({posterior:.3f})",
                f"- Origin: `{hypothesis.origin}`",
                f"- Text: {hypothesis.text}",
            ]
        )
        if hypothesis.competing_hypothesis_ids:
            lines.append(
                "- Competes with: "
                + ", ".join(f"`{hid}`" for hid in hypothesis.competing_hypothesis_ids)
            )
        if hypothesis.unresolved_questions:
            lines.append("- Unresolved:")
            lines.extend(f"  - {item}" for item in hypothesis.unresolved_questions[:4])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_hypothesis_tests_md(hypothesis_set: HypothesisSet) -> str:
    hypotheses = {item.hypothesis_id: item for item in hypothesis_set.hypotheses}
    lines = ["# Hypothesis Tests", ""]
    for result in hypothesis_set.test_results:
        hypothesis = hypotheses.get(result.hypothesis_id)
        lines.extend(
            [
                f"## {result.hypothesis_id}",
                "",
                hypothesis.text if hypothesis else "",
                "",
                f"- Status: `{result.status.value}`",
                f"- Support score: {result.support_score:.3f}",
                f"- Opposition score: {result.opposition_score:.3f}",
                f"- Source diversity: {result.source_diversity}",
                f"- Primary sources: {result.primary_source_count}",
                f"- Citation-ready sources: {result.citation_ready_count}",
                "",
                "### Supporting Evidence",
                "",
            ]
        )
        if result.supporting_evidence:
            for evidence in result.supporting_evidence[:4]:
                lines.append(
                    f"- `{evidence.source_id or evidence.claim_id or evidence.evidence_id}` "
                    f"score {evidence.score:.3f}: {_short(evidence.matched_text)}"
                )
        else:
            lines.append("- None")
        lines.extend(["", "### Opposing Evidence", ""])
        if result.opposing_evidence:
            for evidence in result.opposing_evidence[:4]:
                lines.append(
                    f"- `{evidence.source_id or evidence.claim_id or evidence.evidence_id}` "
                    f"score {evidence.score:.3f}: {_short(evidence.matched_text)}"
                )
        else:
            lines.append("- None")
        if result.unresolved:
            lines.extend(["", "### Unresolved", ""])
            lines.extend(f"- {item}" for item in result.unresolved)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_confidence_updates_md(hypothesis_set: HypothesisSet) -> str:
    hypotheses = {item.hypothesis_id: item for item in hypothesis_set.hypotheses}
    lines = ["# Confidence Updates", ""]
    for update in hypothesis_set.confidence_updates:
        hypothesis = hypotheses.get(update.hypothesis_id)
        lines.extend(
            [
                f"## {update.hypothesis_id}",
                "",
                hypothesis.text if hypothesis else "",
                "",
                f"- Prior: {update.prior_score:.3f}",
                f"- Posterior: {update.posterior_score:.3f}",
                f"- Level: `{update.confidence_level.value}`",
                f"- Domain sensitivity: `{update.domain_sensitivity}`",
                f"- Human review: {'yes' if update.needs_human_review else 'no'}",
                "",
                "### Factors",
                "",
            ]
        )
        if update.factors:
            lines.extend(f"- {item}" for item in update.factors)
        else:
            lines.append("- None")
        lines.extend(["", "### Penalties", ""])
        if update.penalties:
            lines.extend(f"- {item}" for item in update.penalties)
        else:
            lines.append("- None")
        lines.extend(["", "### Missing Evidence", ""])
        if update.missing_evidence:
            lines.extend(f"- {item}" for item in update.missing_evidence)
        else:
            lines.append("- None")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_hypothesis_graph_md(hypothesis_set: HypothesisSet) -> str:
    graph = hypothesis_set.graph
    lines = ["# Hypothesis Graph", ""]
    if graph is None:
        lines.append("Graph was not generated.")
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            f"- Graph: `{graph.graph_id}`",
            f"- Nodes: {len(graph.nodes)}",
            f"- Edges: {len(graph.edges)}",
            "",
            "## Nodes",
            "",
        ]
    )
    for node in graph.nodes[:120]:
        lines.append(f"- `{node.node_id}` `{node.node_type}`: {_short(node.label, max_chars=140)}")
    lines.extend(["", "## Edges", ""])
    for edge in graph.edges[:180]:
        lines.append(
            f"- `{edge.source}` -[{edge.relationship} {edge.weight:.2f}]-> `{edge.target}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def _json_text(value: object) -> str:
    return json.dumps(model_to_plain(value), indent=2, ensure_ascii=False) + "\n"


def _short(text: str, *, max_chars: int = 220) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:") + "..."
