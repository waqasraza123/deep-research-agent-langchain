from __future__ import annotations

from typing import Any

from .contracts import (
    AgentControlPlan,
    AgentControlSettings,
    CompiledInstruction,
    ResearchAgentRole,
    SubagentSpec,
    stable_id,
)


class SubagentBuilder:
    def __init__(self, *, deepagents_supports_subagents: bool = False):
        self.deepagents_supports_subagents = deepagents_supports_subagents
        self.warnings: list[str] = []

    def build_subagent_specs(self, control_plan: AgentControlPlan) -> list[SubagentSpec]:
        if not control_plan.selected_roles:
            return []
        instructions = []
        if control_plan.supervisor_instruction:
            instructions.append(control_plan.supervisor_instruction)
        return control_plan.subagents or []

    def build_subagent_spec(
        self,
        role_name: str,
        compiled_instruction: CompiledInstruction,
        *,
        tools: list[str],
        readable_artifacts: list[str],
        writable_artifacts: list[str],
        skills: list[str],
        max_context_chars: int,
    ) -> SubagentSpec:
        return SubagentSpec(
            subagent_id=stable_id("subagent", compiled_instruction.role.value, role_name),
            role=compiled_instruction.role,
            name=role_name,
            description=f"{role_name} specialist for governed research.",
            system_prompt=compiled_instruction.system_instructions,
            tools=tools,
            readable_artifacts=readable_artifacts,
            writable_artifacts=writable_artifacts,
            skills=skills,
            context_limits={"max_context_chars": max_context_chars},
            expected_outputs=compiled_instruction.artifact_requirements,
            warnings=list(compiled_instruction.warnings),
        )

    def validate_subagent_count(
        self, specs: list[SubagentSpec], settings: AgentControlSettings
    ) -> list[SubagentSpec]:
        if len(specs) <= settings.max_subagents:
            return specs
        self.warnings.append(
            f"subagent count reduced from {len(specs)} to {settings.max_subagents}"
        )
        return specs[: settings.max_subagents]

    def convert_to_deepagents_config(self, subagent_specs: list[SubagentSpec]) -> dict[str, Any]:
        if not self.deepagents_supports_subagents:
            return self.fallback_to_instruction_only_mode(
                "Installed Deep Agents interface is not assumed to expose enforceable "
                "subagent hooks."
            )
        return {
            "subagents": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "system_prompt": spec.system_prompt,
                    "tools": spec.tools,
                }
                for spec in subagent_specs
            ],
            "warnings": list(self.warnings),
        }

    def fallback_to_instruction_only_mode(self, reason: str) -> dict[str, Any]:
        warning = f"Subagent enforcement fallback: {reason}"
        if warning not in self.warnings:
            self.warnings.append(warning)
        return {"subagents": [], "mode": "instruction_only", "warnings": list(self.warnings)}


def specs_from_compiled(
    *,
    instructions: list[CompiledInstruction],
    plan_roles: dict[ResearchAgentRole, Any],
    settings: AgentControlSettings,
) -> tuple[list[SubagentSpec], list[str]]:
    builder = SubagentBuilder()
    specs: list[SubagentSpec] = []
    for instruction in instructions:
        if instruction.role == ResearchAgentRole.SUPERVISOR:
            continue
        role_def = plan_roles[instruction.role]
        role_skills = [
            line.split(":", 1)[0].strip("- ")
            for line in instruction.skill_instructions.splitlines()
            if line.strip()
        ]
        specs.append(
            builder.build_subagent_spec(
                role_name=role_def.name,
                compiled_instruction=instruction,
                tools=role_def.allowed_tools,
                readable_artifacts=role_def.readable_artifacts,
                writable_artifacts=role_def.writable_artifacts,
                skills=role_skills,
                max_context_chars=role_def.max_context_chars,
            )
        )
    specs = builder.validate_subagent_count(specs, settings)
    builder.fallback_to_instruction_only_mode(
        "Control plane records subagent specs and injects prompts; direct runtime "
        "enforcement is advisory."
    )
    return specs, builder.warnings
