from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import ProtocolSelection, model_to_plain

PROTOCOL_ARTIFACTS = (
    "protocol_selection.json",
    "protocol_selection.md",
    "intelligence_profile.json",
    "protocol_instructions.md",
    "policy_requirements.json",
    "policy_warnings.md",
)


def _json_text(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def render_protocol_selection_markdown(selection: ProtocolSelection) -> str:
    protocol = selection.selected_protocol
    profile = selection.intelligence_profile
    lines = [
        "# Protocol Selection",
        "",
        f"- Protocol: `{protocol.protocol_id}` ({protocol.name})",
        f"- Profile: `{profile.profile_id}` ({profile.name})",
        f"- Confidence: `{selection.confidence_score:.3f}`",
        f"- Review recommended: `{selection.review_recommended}`",
        f"- Alternatives: {', '.join(f'`{item}`' for item in selection.alternative_protocols) or 'none'}",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in selection.reasons)
    lines.extend(["", "## Source Requirements", ""])
    for requirement in selection.effective_source_requirements:
        required = "required" if requirement.required else "preferred"
        lines.append(
            f"- `{requirement.source_type}` ({required}, min {requirement.minimum_count}): "
            f"{requirement.rationale}"
        )
    lines.extend(["", "## Warnings", ""])
    if not selection.warnings:
        lines.append("- None")
    for warning in selection.warnings:
        lines.append(
            f"- `{warning.severity}` `{warning.code}`: {warning.message} "
            f"(review: {warning.review_recommended})"
        )
    return "\n".join(lines).rstrip() + "\n"


def policy_requirements_payload(selection: ProtocolSelection) -> dict[str, Any]:
    return {
        "protocol_id": selection.selected_protocol.protocol_id,
        "profile_id": selection.intelligence_profile.profile_id,
        "source_requirements": [
            model_to_plain(requirement) for requirement in selection.effective_source_requirements
        ],
        "verification": model_to_plain(selection.effective_verification),
        "citation_policy": model_to_plain(selection.effective_citation_policy),
        "freshness_policy": model_to_plain(selection.effective_freshness_policy),
        "synthesis_policy": model_to_plain(selection.effective_synthesis_policy),
        "evaluation_policy": model_to_plain(selection.effective_evaluation_policy),
        "review_recommended": selection.review_recommended,
        "policy_packs": [model_to_plain(pack) for pack in selection.policy_packs],
    }


def render_policy_warnings_markdown(selection: ProtocolSelection) -> str:
    lines = ["# Policy Warnings", ""]
    if not selection.warnings:
        lines.append("- None")
    for warning in selection.warnings:
        lines.append(
            f"- `{warning.severity}` `{warning.code}`: {warning.message} "
            f"(review recommended: `{warning.review_recommended}`)"
        )
    if selection.selected_protocol.safety_warnings.professional_advice_disclaimer:
        lines.extend(
            [
                "",
                "## Professional Advice Boundary",
                "",
                "- This protocol is informational. It must not be presented as professional advice.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_protocol_artifacts(thread_dir: Path, selection: ProtocolSelection) -> list[str]:
    thread_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "protocol_selection.json": _json_text(model_to_plain(selection)),
        "protocol_selection.md": render_protocol_selection_markdown(selection),
        "intelligence_profile.json": _json_text(model_to_plain(selection.intelligence_profile)),
        "protocol_instructions.md": selection.instruction_block,
        "policy_requirements.json": _json_text(policy_requirements_payload(selection)),
        "policy_warnings.md": render_policy_warnings_markdown(selection),
    }
    for name, content in files.items():
        (thread_dir / name).write_text(content, encoding="utf-8")
    return list(files)

