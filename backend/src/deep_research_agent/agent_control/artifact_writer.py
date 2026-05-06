from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import artifact_abs_path, ensure_thread_dir
from .contracts import (
    AgentControlPlan,
    AgentControlSummary,
    AgentHandoff,
    AgentOutputValidation,
    AgentTraceEvent,
    ArtifactContract,
    CompiledInstruction,
    ContextBundle,
    FilesystemPolicy,
    PolicyViolation,
    SkillSelection,
    SubagentSpec,
    ToolPolicy,
    json_text,
    model_to_plain,
)

CONTROL_ARTIFACTS = (
    "agent_control_plan.json",
    "agent_control_plan.md",
    "role_selection.json",
    "role_selection.md",
    "skill_selection.json",
    "skill_selection.md",
    "agent_policies.json",
    "agent_policies.md",
    "tool_policies.json",
    "filesystem_policies.json",
    "context_bundles.json",
    "context_bundles.md",
    "trust_boundary.md",
    "source_context_warnings.json",
    "source_context_warnings.md",
    "compiled_instructions.json",
    "supervisor_instructions.md",
    "subagent_instructions.md",
    "subagent_specs.json",
    "subagent_specs.md",
    "artifact_contracts.json",
    "artifact_contracts.md",
    "agent_handoffs.json",
    "agent_handoffs.md",
    "agent_trace.jsonl",
    "agent_trace.md",
    "trace_analysis.json",
    "trace_analysis.md",
    "policy_violations.json",
    "policy_violations.md",
    "handoff_validation.json",
    "handoff_validation.md",
    "agent_output_validation.json",
    "agent_output_validation.md",
    "agent_control_summary.json",
    "agent_control_summary.md",
    "control_plane_warnings.md",
    "agent_control_error.json",
)


def write_text_artifact(runs_dir: Path, thread_id: str, rel_path: str, content: str) -> str:
    path = artifact_abs_path(runs_dir, thread_id, rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return rel_path


def write_json_artifact(runs_dir: Path, thread_id: str, rel_path: str, payload: Any) -> str:
    return write_text_artifact(runs_dir, thread_id, rel_path, json_text(payload))


def read_json_artifact(runs_dir: Path, thread_id: str, rel_path: str) -> Any:
    import json

    path = artifact_abs_path(runs_dir, thread_id, rel_path)
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(rel_path)
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(rows: list[tuple[str, Any]]) -> str:
    lines = ["| Field | Value |", "| --- | --- |"]
    for key, value in rows:
        rendered = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        lines.append(f"| {key} | {rendered.replace('|', '/') or '-'} |")
    return "\n".join(lines)


def render_skill_selection(selection: SkillSelection) -> str:
    lines = ["# Skill Selection", "", f"Question: {selection.question}", ""]
    lines.append("## Selected")
    for skill in selection.selected_skills:
        reasons = "; ".join(selection.selection_reasons.get(skill.skill_id, []))
        lines.append(f"- `{skill.skill_id}` ({skill.required_role or 'shared'}): {reasons}")
    lines.extend(["", "## Skipped"])
    for skill_id in selection.skipped_skills:
        lines.append(f"- `{skill_id}`")
    if selection.warnings:
        lines.extend(["", "## Warnings", *[f"- {w}" for w in selection.warnings]])
    return "\n".join(lines) + "\n"


def render_policies(tool_policies: list[ToolPolicy], fs_policies: list[FilesystemPolicy]) -> str:
    lines = ["# Agent Policies", "", "## Tool Policies"]
    for policy in tool_policies:
        lines.append(f"### {policy.role.value}")
        lines.append(
            markdown_table(
                [
                    ("allowed_tools", policy.allowed_tools),
                    ("denied_tools", policy.denied_tools),
                    ("allow_network", policy.allow_network),
                    ("allow_source_fetch", policy.allow_source_fetch),
                    ("max_tool_calls", policy.max_tool_calls),
                ]
            )
        )
    lines.extend(["", "## Filesystem Policies"])
    for policy in fs_policies:
        lines.append(f"### {policy.role.value}")
        lines.append(
            markdown_table(
                [
                    ("readable_paths", policy.readable_paths),
                    ("writable_paths", policy.writable_paths),
                    ("write_mode", policy.write_mode.value),
                    ("required_artifacts", policy.required_artifacts),
                ]
            )
        )
    return "\n\n".join(lines) + "\n"


def render_context_bundles(bundles: list[ContextBundle]) -> str:
    lines = ["# Context Bundles", ""]
    for bundle in bundles:
        lines.extend(
            [
                f"## {bundle.role.value}",
                markdown_table(
                    [
                        ("bundle_id", bundle.bundle_id),
                        ("max_chars", bundle.max_chars),
                        ("truncation_applied", bundle.truncation_applied),
                        ("warnings", bundle.warnings),
                    ]
                ),
                "",
            ]
        )
    return "\n".join(lines)


def render_compiled_instructions(instructions: list[CompiledInstruction]) -> str:
    lines = ["# Subagent Instructions", ""]
    for instruction in instructions:
        lines.extend(
            [
                f"## {instruction.role.value}",
                instruction.system_instructions,
                "",
                "### Role",
                instruction.role_instructions,
                "",
                "### Skills",
                instruction.skill_instructions or "- none",
                "",
            ]
        )
    return "\n".join(lines)


def render_subagent_specs(specs: list[SubagentSpec]) -> str:
    lines = ["# Subagent Specs", ""]
    for spec in specs:
        lines.append(f"## {spec.name}")
        lines.append(
            markdown_table(
                [
                    ("role", spec.role.value),
                    ("tools", spec.tools),
                    ("skills", spec.skills),
                    ("writable_artifacts", spec.writable_artifacts),
                    ("warnings", spec.warnings),
                ]
            )
        )
    return "\n\n".join(lines) + "\n"


def render_handoffs(handoffs: list[AgentHandoff]) -> str:
    lines = ["# Agent Handoffs", ""]
    for handoff in handoffs:
        lines.append(
            f"- `{handoff.from_role.value}` -> `{handoff.to_role.value}`: "
            f"{handoff.reason} ({handoff.status.value})"
        )
    return "\n".join(lines) + "\n"


def render_contracts(contracts: list[ArtifactContract]) -> str:
    lines = ["# Artifact Contracts", ""]
    for contract in contracts:
        lines.append(
            f"- `{contract.artifact_name}`: producer `{contract.producer_role.value}`, "
            f"required={contract.required}, format={contract.format}"
        )
    return "\n".join(lines) + "\n"


def render_validations(validations: list[AgentOutputValidation]) -> str:
    lines = ["# Agent Output Validation", ""]
    for item in validations:
        status = "pass" if item.valid else "fail"
        lines.append(
            f"- `{item.artifact_name}`: {status}; exists={item.exists}; "
            f"errors={'; '.join(item.errors) or '-'}; warnings={'; '.join(item.warnings) or '-'}"
        )
    return "\n".join(lines) + "\n"


def render_violations(violations: list[PolicyViolation]) -> str:
    lines = ["# Policy Violations", ""]
    if not violations:
        lines.append("No policy violations recorded.")
    for violation in violations:
        lines.append(
            f"- {violation.severity}: `{violation.role.value}` {violation.message} "
            f"({violation.policy_type})"
        )
    return "\n".join(lines) + "\n"


def render_summary(summary: AgentControlSummary) -> str:
    return (
        "# Agent Control Summary\n\n"
        + markdown_table(
            [
                ("thread_id", summary.thread_id),
                ("roles_selected", summary.roles_selected),
                ("skills_selected", summary.skills_selected),
                ("subagents_created", summary.subagents_created),
                ("handoffs_planned", summary.handoffs_planned),
                ("handoffs_completed", summary.handoffs_completed),
                ("policy_violations", summary.policy_violations),
                ("missing_artifacts", summary.missing_artifacts),
                ("invalid_outputs", summary.invalid_outputs),
                ("trace_event_count", summary.trace_event_count),
                ("warnings", summary.warnings),
                ("recommended_actions", summary.recommended_actions),
            ]
        )
        + "\n"
    )


def write_plan_artifacts(runs_dir: Path, plan: AgentControlPlan) -> list[str]:
    ensure_thread_dir(runs_dir, plan.thread_id)
    paths = [
        write_json_artifact(runs_dir, plan.thread_id, "agent_control_plan.json", plan),
        write_json_artifact(runs_dir, plan.thread_id, "role_selection.json", plan.selected_roles),
        write_json_artifact(runs_dir, plan.thread_id, "skill_selection.json", plan.selected_skills),
        write_json_artifact(
            runs_dir,
            plan.thread_id,
            "agent_policies.json",
            {
                "tool_policies": plan.tool_policies,
                "filesystem_policies": plan.filesystem_policies,
                "warnings": plan.policy_warnings,
            },
        ),
        write_json_artifact(runs_dir, plan.thread_id, "tool_policies.json", plan.tool_policies),
        write_json_artifact(
            runs_dir, plan.thread_id, "filesystem_policies.json", plan.filesystem_policies
        ),
        write_json_artifact(runs_dir, plan.thread_id, "context_bundles.json", plan.context_bundles),
        write_json_artifact(
            runs_dir,
            plan.thread_id,
            "compiled_instructions.json",
            [p for p in [plan.supervisor_instruction] if p is not None],
        ),
        write_json_artifact(runs_dir, plan.thread_id, "subagent_specs.json", plan.subagents),
        write_json_artifact(
            runs_dir, plan.thread_id, "agent_handoffs.json", plan.expected_handoffs
        ),
    ]
    paths.extend(
        [
            write_text_artifact(
                runs_dir, plan.thread_id, "agent_control_plan.md", render_plan(plan)
            ),
            write_text_artifact(
                runs_dir, plan.thread_id, "role_selection.md", render_roles(plan.selected_roles)
            ),
            write_text_artifact(
                runs_dir,
                plan.thread_id,
                "skill_selection.md",
                render_skill_selection(plan.selected_skills),
            ),
            write_text_artifact(
                runs_dir,
                plan.thread_id,
                "agent_policies.md",
                render_policies(plan.tool_policies, plan.filesystem_policies),
            ),
            write_text_artifact(
                runs_dir,
                plan.thread_id,
                "context_bundles.md",
                render_context_bundles(plan.context_bundles),
            ),
            write_text_artifact(
                runs_dir,
                plan.thread_id,
                "supervisor_instructions.md",
                plan.supervisor_instruction.system_instructions
                if plan.supervisor_instruction
                else "",
            ),
            write_text_artifact(
                runs_dir,
                plan.thread_id,
                "subagent_instructions.md",
                render_compiled_instructions(
                    [p for p in [plan.supervisor_instruction] if p is not None]
                ),
            ),
            write_text_artifact(
                runs_dir, plan.thread_id, "subagent_specs.md", render_subagent_specs(plan.subagents)
            ),
            write_text_artifact(
                runs_dir,
                plan.thread_id,
                "agent_handoffs.md",
                render_handoffs(plan.expected_handoffs),
            ),
            write_text_artifact(
                runs_dir,
                plan.thread_id,
                "policy_warnings.md",
                "# Policy Warnings\n\n" + "\n".join(f"- {w}" for w in plan.policy_warnings) + "\n",
            ),
        ]
    )
    return paths


def render_roles(roles: list[Any]) -> str:
    lines = ["# Role Selection", ""]
    for role in roles:
        role_data = model_to_plain(role)
        lines.append(f"- `{role_data['role']}`: {role_data['purpose']}")
    return "\n".join(lines) + "\n"


def render_plan(plan: AgentControlPlan) -> str:
    return (
        "# Agent Control Plan\n\n"
        + markdown_table(
            [
                ("plan_id", plan.plan_id),
                ("thread_id", plan.thread_id),
                ("detected_intent", plan.detected_intent),
                ("roles", [role.role.value for role in plan.selected_roles]),
                ("skills", [skill.skill_id for skill in plan.selected_skills.selected_skills]),
                ("subagents", [agent.name for agent in plan.subagents]),
                ("required_artifacts", plan.required_artifacts),
                ("warnings", plan.policy_warnings),
            ]
        )
        + "\n"
    )


def append_trace_jsonl(runs_dir: Path, thread_id: str, event: AgentTraceEvent) -> str:
    path = artifact_abs_path(runs_dir, thread_id, "agent_trace.jsonl")
    with path.open("a", encoding="utf-8") as f:
        f.write(json_text(event).strip() + "\n")
    return "agent_trace.jsonl"
