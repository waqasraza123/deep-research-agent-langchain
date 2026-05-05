from __future__ import annotations

import json
from pathlib import Path

from .contracts import SourceSafetyBatch, model_to_plain

SOURCE_SAFETY_ARTIFACTS = (
    "source_safety.json",
    "source_safety.md",
    "prompt_injection_findings.json",
    "prompt_injection_findings.md",
    "source_poisoning_findings.json",
    "source_poisoning_findings.md",
    "sanitized_sources.json",
    "trust_boundary_policy.md",
)


def _json_text(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write_source_safety_artifacts(thread_dir: Path, batch: SourceSafetyBatch) -> list[str]:
    files = {
        "source_safety.json": _json_text(model_to_plain(batch)),
        "source_safety.md": render_source_safety_markdown(batch),
        "prompt_injection_findings.json": _json_text(prompt_injection_payload(batch)),
        "prompt_injection_findings.md": render_prompt_injection_markdown(batch),
        "source_poisoning_findings.json": _json_text(source_poisoning_payload(batch)),
        "source_poisoning_findings.md": render_source_poisoning_markdown(batch),
        "sanitized_sources.json": _json_text(sanitized_sources_payload(batch)),
        "trust_boundary_policy.md": render_trust_boundary_policy_markdown(batch),
    }
    for rel_path, content in files.items():
        (thread_dir / rel_path).write_text(content, encoding="utf-8")
    return sorted(files)


def prompt_injection_payload(batch: SourceSafetyBatch) -> dict:
    findings = [
        model_to_plain(finding)
        for assessment in batch.assessments
        for finding in assessment.prompt_injection_findings
    ]
    return {"thread_id": batch.thread_id, "generated_at": batch.generated_at, "findings": findings}


def source_poisoning_payload(batch: SourceSafetyBatch) -> dict:
    findings = [
        model_to_plain(finding)
        for assessment in batch.assessments
        for finding in assessment.source_poisoning_findings
    ]
    return {"thread_id": batch.thread_id, "generated_at": batch.generated_at, "findings": findings}


def sanitized_sources_payload(batch: SourceSafetyBatch) -> dict:
    return {
        "thread_id": batch.thread_id,
        "generated_at": batch.generated_at,
        "sources": [model_to_plain(a.sanitized_content) for a in batch.assessments],
    }


def render_source_safety_markdown(batch: SourceSafetyBatch) -> str:
    lines = [
        "# Source Safety",
        "",
        f"- Thread: `{batch.thread_id or 'ad-hoc'}`",
        f"- Generated: `{batch.generated_at}`",
        f"- Sources assessed: {len(batch.assessments)}",
        f"- Critical: {batch.summary.get('critical', 0)}",
        f"- High: {batch.summary.get('high', 0)}",
        f"- Medium: {batch.summary.get('medium', 0)}",
        "",
        "## Policy",
        "",
        batch.policy.agent_instruction_block,
        "",
        "## Assessments",
        "",
    ]
    if not batch.assessments:
        lines.append("- None")
    for assessment in batch.assessments:
        risk = assessment.risk_score
        lines.extend(
            [
                f"### {assessment.source_id}: {assessment.title or assessment.url or 'Untitled'}",
                "",
                f"- URL: {assessment.url}",
                f"- Risk: `{risk.risk_level}` ({risk.numeric_score:.1f}/100)",
                f"- Recommended action: `{risk.recommended_action}`",
                f"- Agent context allowed: `{assessment.sanitized_content.agent_context_allowed}`",
                "- Sanitized artifact: "
                f"`{assessment.sanitized_content.sanitized_local_path or 'n/a'}`",
                "",
            ]
        )
        lines.append("Reasons:")
        lines.extend(f"- {reason}" for reason in risk.reasons)
        lines.append("")
        if assessment.warnings:
            lines.append("Warnings:")
            lines.extend(f"- `{w.risk_level}` `{w.code}`: {w.message}" for w in assessment.warnings)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_prompt_injection_markdown(batch: SourceSafetyBatch) -> str:
    findings = [
        (assessment, finding)
        for assessment in batch.assessments
        for finding in assessment.prompt_injection_findings
    ]
    lines = ["# Prompt Injection Findings", ""]
    if not findings:
        lines.append("- None")
        return "\n".join(lines) + "\n"
    for assessment, finding in findings:
        lines.extend(
            [
                f"## {finding.finding_id}",
                "",
                f"- Source: {assessment.source_id} ({assessment.url})",
                f"- Risk: `{finding.risk_level}`",
                f"- Category: `{finding.category}`",
                f"- Pattern: `{finding.pattern}`",
                f"- Recommended action: `{finding.recommended_action}`",
                f"- Explanation: {finding.explanation}",
                f"- Evidence: `{finding.matched_text}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_source_poisoning_markdown(batch: SourceSafetyBatch) -> str:
    findings = [
        (assessment, finding)
        for assessment in batch.assessments
        for finding in assessment.source_poisoning_findings
    ]
    lines = ["# Source Poisoning Findings", ""]
    if not findings:
        lines.append("- None")
        return "\n".join(lines) + "\n"
    for assessment, finding in findings:
        lines.extend(
            [
                f"## {finding.finding_id}",
                "",
                f"- Source: {assessment.source_id} ({assessment.url})",
                f"- Risk: `{finding.risk_level}`",
                f"- Category: `{finding.category}`",
                f"- Recommended action: `{finding.recommended_action}`",
                f"- Explanation: {finding.explanation}",
                f"- Evidence: `{finding.evidence}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_trust_boundary_policy_markdown(batch: SourceSafetyBatch) -> str:
    policy = batch.policy
    lines = [
        "# Trust Boundary Policy",
        "",
        f"- Policy ID: `{policy.policy_id}`",
        f"- Source content is untrusted: `{policy.source_content_is_untrusted}`",
        f"- Evidence only: `{policy.evidence_only}`",
        f"- Block instruction override: `{policy.block_instruction_override}`",
        f"- Quote suspicious sections: `{policy.quote_suspicious_sections}`",
        f"- Exclude critical from agent context: `{policy.exclude_critical_from_agent_context}`",
        "",
        "## Agent Instruction Block",
        "",
        policy.agent_instruction_block,
        "",
        "## Wrapper Markers",
        "",
        f"- Begin: `{policy.wrapper_preamble}`",
        f"- End: `{policy.wrapper_postamble}`",
    ]
    return "\n".join(lines).rstrip() + "\n"
