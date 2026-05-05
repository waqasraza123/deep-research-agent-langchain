from __future__ import annotations

import json
from pathlib import Path

from .contracts import SourceAuditBatch, model_to_plain


def _json_text(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def render_source_audit_markdown(batch: SourceAuditBatch) -> str:
    lines = [
        "# Source Audit",
        "",
        f"Question: {batch.question}",
        "",
        "## Summary",
        "",
        f"- Sources audited: {batch.summary.source_count}",
        f"- Usable sources: {batch.summary.usable_source_count}",
        f"- Average final score: {batch.summary.average_final_score:.3f}",
        f"- Ranked source IDs: {', '.join(batch.summary.ranked_source_ids) or 'None'}",
        "",
        "## Operator Guidance",
        "",
        batch.summary.instruction_block or "No source guidance was generated.",
        "",
        "## Ranked Sources",
        "",
    ]
    ranked = sorted(batch.audits, key=lambda item: item.final_source_score, reverse=True)
    if not ranked:
        lines.append("- None")
    for audit in ranked:
        lines.extend(
            [
                f"### {audit.source_id}: {audit.title or audit.domain or audit.url}",
                "",
                f"- URL: {audit.url}",
                f"- Domain: {audit.domain}",
                f"- Type: {audit.source_type}",
                f"- Final score: {audit.final_source_score:.3f}",
                f"- Recommended usage: `{audit.recommended_usage}`",
                f"- Credibility: {audit.credibility_score.score:.3f}",
                f"- Freshness: {audit.freshness_score.status} ({audit.freshness_score.score:.3f})",
                "- Authority: "
                f"{audit.authority_score.score:.3f} ({audit.authority_score.source_role})",
                f"- Primary likelihood: {audit.primary_source_likelihood.likelihood:.3f}",
                "- Bias risk: "
                f"{audit.bias_risk_score.risk_level} ({audit.bias_risk_score.score:.3f})",
                f"- Citation readiness: {audit.citation_readiness_score.score:.3f}",
                "",
            ]
        )
        if audit.reasons:
            lines.append("Reasons:")
            lines.extend(f"- {reason}" for reason in audit.reasons)
            lines.append("")
        if audit.warnings:
            lines.append("Warnings:")
            lines.extend(
                f"- `{warning.severity}` `{warning.code}`: {warning.message}"
                for warning in audit.warnings
            )
            lines.append("")

    lines.extend(["## Gaps", ""])
    for label, values in (
        ("Coverage", batch.summary.coverage_gaps),
        ("Freshness", batch.summary.freshness_gaps),
        ("Authority", batch.summary.authority_gaps),
        ("Citation risks", batch.summary.citation_risks),
    ):
        lines.append(f"### {label}")
        lines.append("")
        if values:
            lines.extend(f"- {value}" for value in values)
        else:
            lines.append("- None")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_source_warnings_markdown(batch: SourceAuditBatch) -> str:
    lines = ["# Source Warnings", ""]
    warnings = [(audit, warning) for audit in batch.audits for warning in audit.warnings]
    if not warnings and not batch.summary.citation_risks:
        lines.append("- None")
        return "\n".join(lines) + "\n"
    for audit, warning in warnings:
        lines.append(
            f"- `{warning.severity}` `{warning.code}` in {audit.source_id} "
            f"({audit.domain}): {warning.message}"
        )
    if batch.summary.citation_risks:
        lines.extend(["", "## Citation Risks", ""])
        lines.extend(f"- {risk}" for risk in batch.summary.citation_risks)
    return "\n".join(lines).rstrip() + "\n"


def source_rankings_payload(batch: SourceAuditBatch) -> dict:
    ranked = sorted(batch.audits, key=lambda item: item.final_source_score, reverse=True)
    return {
        "thread_id": batch.thread_id,
        "question": batch.question,
        "generated_at": batch.generated_at,
        "ranked_sources": [
            {
                "rank": idx,
                "source_id": audit.source_id,
                "url": audit.url,
                "domain": audit.domain,
                "title": audit.title,
                "final_source_score": audit.final_source_score,
                "recommended_usage": audit.recommended_usage,
                "freshness_status": audit.freshness_score.status,
                "authority_score": audit.authority_score.score,
                "bias_risk_level": audit.bias_risk_score.risk_level,
                "citation_ready": audit.citation_readiness_score.citation_ready,
            }
            for idx, audit in enumerate(ranked, start=1)
        ],
        "recommended_primary_sources": batch.summary.recommended_primary_sources,
        "sources_needing_verification": batch.summary.sources_needing_verification,
        "sources_to_avoid": batch.summary.sources_to_avoid,
        "coverage_gaps": batch.summary.coverage_gaps,
        "freshness_gaps": batch.summary.freshness_gaps,
        "authority_gaps": batch.summary.authority_gaps,
    }


def citation_readiness_payload(batch: SourceAuditBatch) -> dict:
    ranked = sorted(batch.audits, key=lambda item: item.final_source_score, reverse=True)
    return {
        "thread_id": batch.thread_id,
        "question": batch.question,
        "generated_at": batch.generated_at,
        "sources": [
            {
                "source_id": audit.source_id,
                "url": audit.url,
                "domain": audit.domain,
                "title": audit.title,
                "citation_readiness": model_to_plain(audit.citation_readiness_score),
                "recommended_usage": audit.recommended_usage,
            }
            for audit in ranked
        ],
        "citation_risks": batch.summary.citation_risks,
    }


def write_source_audit_artifacts(thread_dir: Path, batch: SourceAuditBatch) -> list[str]:
    files = {
        "source_audit.json": _json_text(model_to_plain(batch)),
        "source_audit.md": render_source_audit_markdown(batch),
        "source_rankings.json": _json_text(source_rankings_payload(batch)),
        "source_warnings.md": render_source_warnings_markdown(batch),
        "citation_readiness.json": _json_text(citation_readiness_payload(batch)),
    }
    for rel_path, content in files.items():
        (thread_dir / rel_path).write_text(content, encoding="utf-8")
    return sorted(files)
