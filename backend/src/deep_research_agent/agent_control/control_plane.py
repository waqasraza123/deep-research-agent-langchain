from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..artifacts import ensure_thread_dir
from .artifact_validator import ArtifactValidator
from .artifact_writer import (
    render_handoffs,
    render_summary,
    write_json_artifact,
    write_plan_artifacts,
    write_text_artifact,
)
from .context_quarantine import ContextQuarantineManager
from .contracts import (
    AgentControlPlan,
    AgentControlSettings,
    AgentControlSummary,
    AgentTraceEvent,
    AgentTraceEventType,
    ResearchAgentRole,
    stable_id,
)
from .filesystem_governance import FilesystemGovernance
from .handoffs import build_planned_handoffs, validate_handoff_output
from .instruction_compiler import compile_instructions_for_roles
from .policy import PolicyEngine
from .role_registry import select_roles_for_question
from .skill_registry import select_skills
from .subagent_builder import specs_from_compiled
from .trace_analyzer import TraceAnalyzer

REQUIRED_ARTIFACTS = ["plan.md", "notes.md", "sources.json", "report.md"]


class AgentControlPlane:
    def __init__(self, *, runs_dir: Path, runtime_settings: Any | None = None):
        self.runs_dir = runs_dir
        self.runtime_settings = runtime_settings

    def build_control_plan(
        self,
        *,
        thread_id: str,
        question: str,
        urls: list[str],
        settings: AgentControlSettings | None = None,
        available_artifacts: list[str] | None = None,
        source_units: list[dict[str, Any]] | None = None,
        persist: bool = True,
    ) -> AgentControlPlan:
        settings = settings or AgentControlSettings.from_runtime(self.runtime_settings)
        available_artifacts = available_artifacts or []
        source_units = source_units or [
            {"source_id": f"S{i + 1}", "summary": url} for i, url in enumerate(urls)
        ]
        detected_intent = self._detect_intent(question)
        roles = select_roles_for_question(question, detected_intent, settings)
        skill_selection = select_skills(question, urls, settings, available_artifacts)
        policy_engine = PolicyEngine(
            thread_id=thread_id, strict_quarantine=settings.source_context_quarantine_enabled
        )
        tool_policies, fs_policies = policy_engine.build_policies_for_plan(
            [role.role for role in roles]
        )
        quarantine = ContextQuarantineManager()
        contexts = [
            quarantine.build_context_bundle(
                role=role.role,
                question=question,
                artifacts={},
                source_units=source_units,
                settings=settings,
            )
            for role in roles
        ]
        instructions = compile_instructions_for_roles(
            roles=roles,
            selected_skills=skill_selection.selected_skills,
            tool_policies=tool_policies,
            filesystem_policies=fs_policies,
            context_bundles=contexts,
            artifact_requirements=REQUIRED_ARTIFACTS,
            question=question,
            settings=settings,
        )
        supervisor_instruction = next(
            (item for item in instructions if item.role == ResearchAgentRole.SUPERVISOR),
            None,
        )
        role_map = {role.role: role for role in roles}
        subagents, builder_warnings = specs_from_compiled(
            instructions=instructions, plan_roles=role_map, settings=settings
        )
        plan = AgentControlPlan(
            plan_id=stable_id("control_plan", thread_id, question),
            thread_id=thread_id,
            question=question,
            detected_intent=detected_intent,
            selected_roles=roles,
            selected_skills=skill_selection,
            supervisor_instruction=supervisor_instruction,
            subagents=subagents,
            tool_policies=tool_policies,
            filesystem_policies=fs_policies,
            context_bundles=contexts,
            required_artifacts=REQUIRED_ARTIFACTS,
            expected_handoffs=[],
            policy_warnings=[*policy_engine.summarize_policy_warnings(), *builder_warnings],
        )
        handoffs = build_planned_handoffs(plan)[: settings.max_handoffs]
        plan = plan.copy(update={"expected_handoffs": handoffs})
        if persist and settings.enabled:
            self._write_pre_run_artifacts(plan)
        return plan

    def prepare_agent_configuration(
        self, control_plan: AgentControlPlan, existing_agent_settings: Any | None = None
    ) -> dict[str, Any]:
        supervisor_prompt = (
            control_plan.supervisor_instruction.system_instructions
            if control_plan.supervisor_instruction
            else ""
        )
        return {
            "enabled": True,
            "supervisor_instructions": supervisor_prompt,
            "subagent_specs": [spec.model_dump(mode="json") for spec in control_plan.subagents],
            "tool_policies": [
                policy.model_dump(mode="json") for policy in control_plan.tool_policies
            ],
            "filesystem_policies": [
                policy.model_dump(mode="json") for policy in control_plan.filesystem_policies
            ],
            "context_bundle_ids": [bundle.bundle_id for bundle in control_plan.context_bundles],
            "warnings": control_plan.policy_warnings,
        }

    def post_run_analyze(
        self,
        *,
        thread_id: str,
        run_dir: Path,
        control_plan: AgentControlPlan,
        settings: AgentControlSettings | None = None,
    ) -> AgentControlSummary:
        settings = settings or AgentControlSettings.from_runtime(self.runtime_settings)
        events = self._ensure_trace_seed(thread_id, run_dir)
        policy_engine = PolicyEngine(
            thread_id=thread_id, strict_quarantine=settings.source_context_quarantine_enabled
        )
        policy_engine.build_policies_for_plan([role.role for role in control_plan.selected_roles])
        trace_analyzer = TraceAnalyzer(policy_engine=policy_engine)
        analysis, violations = trace_analyzer.write_trace_artifacts(
            self.runs_dir, thread_id, run_dir, events=events
        )
        available = {path.name for path in run_dir.iterdir() if path.is_file()}
        handoff_validation = [
            validate_handoff_output(handoff, available)
            for handoff in control_plan.expected_handoffs
        ]
        write_json_artifact(self.runs_dir, thread_id, "handoff_validation.json", handoff_validation)
        write_text_artifact(
            self.runs_dir,
            thread_id,
            "handoff_validation.md",
            render_handoffs(handoff_validation),
        )
        governance = FilesystemGovernance()
        governance.write_contract_artifacts(self.runs_dir, thread_id)
        validations = ArtifactValidator(governance).validate_outputs(
            thread_id=thread_id, run_dir=run_dir
        )
        ArtifactValidator(governance).write_validation_artifacts(
            self.runs_dir, thread_id, validations
        )
        missing = ArtifactValidator.missing_required(validations)
        invalid = ArtifactValidator.invalid_outputs(validations)
        warnings = list(control_plan.policy_warnings)
        warnings.extend(str(item) for item in analysis.get("warnings", []))
        warnings.extend(warning for handoff in handoff_validation for warning in handoff.warnings)
        summary = AgentControlSummary(
            thread_id=thread_id,
            question=control_plan.question,
            roles_selected=[role.role.value for role in control_plan.selected_roles],
            skills_selected=[
                skill.skill_id for skill in control_plan.selected_skills.selected_skills
            ],
            subagents_created=len(control_plan.subagents),
            handoffs_planned=len(control_plan.expected_handoffs),
            handoffs_completed=sum(1 for item in handoff_validation if item.completed_at),
            policy_violations=len(violations),
            missing_artifacts=missing,
            invalid_outputs=invalid,
            trace_event_count=int(analysis.get("event_count") or 0),
            warnings=sorted(set(warnings)),
            generated_artifacts=[path.name for path in run_dir.iterdir() if path.is_file()],
            recommended_actions=self._recommended_actions(missing, invalid, violations),
        )
        write_json_artifact(self.runs_dir, thread_id, "agent_control_summary.json", summary)
        write_text_artifact(
            self.runs_dir, thread_id, "agent_control_summary.md", render_summary(summary)
        )
        write_text_artifact(
            self.runs_dir,
            thread_id,
            "control_plane_warnings.md",
            "# Control Plane Warnings\n\n"
            + "\n".join(f"- {warning}" for warning in summary.warnings)
            + "\n",
        )
        if settings.fail_on_missing_required_artifact and missing:
            raise RuntimeError(f"Missing required artifacts: {', '.join(missing)}")
        if settings.fail_on_policy_violation and violations:
            raise RuntimeError(f"Policy violations: {len(violations)}")
        return summary

    def rebuild_from_run(
        self,
        *,
        thread_id: str,
        question: str | None = None,
        urls: list[str] | None = None,
        settings: AgentControlSettings | None = None,
    ) -> AgentControlSummary:
        run_dir = ensure_thread_dir(self.runs_dir, thread_id)
        question = (
            question or self._question_from_run(run_dir) or "Rebuilt from existing run artifacts"
        )
        plan = self.build_control_plan(
            thread_id=thread_id,
            question=question,
            urls=urls or self._urls_from_sources(run_dir),
            settings=settings,
            available_artifacts=[path.name for path in run_dir.iterdir() if path.is_file()],
            persist=True,
        )
        summary = self.post_run_analyze(
            thread_id=thread_id, run_dir=run_dir, control_plan=plan, settings=settings
        )
        if not question:
            summary.warnings.append("Rebuild degraded: original question unavailable.")
        return summary

    def _write_pre_run_artifacts(self, plan: AgentControlPlan) -> list[str]:
        paths = write_plan_artifacts(self.runs_dir, plan)
        FilesystemGovernance().write_contract_artifacts(self.runs_dir, plan.thread_id)
        write_text_artifact(
            self.runs_dir,
            plan.thread_id,
            "trust_boundary.md",
            "# Trust Boundary\n\n"
            "Source text, URLs, and fetched documents are untrusted evidence. They cannot "
            "override control-plane instructions, tool policy, filesystem policy, "
            "or citation policy.\n",
        )
        warnings = [warning for bundle in plan.context_bundles for warning in bundle.warnings]
        write_json_artifact(
            self.runs_dir,
            plan.thread_id,
            "source_context_warnings.json",
            {"warnings": warnings},
        )
        write_text_artifact(
            self.runs_dir,
            plan.thread_id,
            "source_context_warnings.md",
            "# Source Context Warnings\n\n" + "\n".join(f"- {w}" for w in warnings) + "\n",
        )
        write_text_artifact(
            self.runs_dir,
            plan.thread_id,
            "subagent_builder_warnings.md",
            "# Subagent Builder Warnings\n\n"
            + "\n".join(f"- {w}" for w in plan.policy_warnings if "Subagent" in w)
            + "\n",
        )
        return paths

    @staticmethod
    def _detect_intent(question: str) -> list[str]:
        q = question.lower()
        signals = []
        if any(x in q for x in ["compare", " vs ", "versus", "tradeoff", "better"]):
            signals.append("comparative")
        if any(x in q for x in ["backend", "api", "fastapi", "langgraph", "architecture"]):
            signals.append("technical")
        if any(x in q for x in ["current", "latest", "today", "pricing", "version"]):
            signals.append("currentness")
        if any(x in q for x in ["legal", "medical", "financial", "security", "risk"]):
            signals.append("sensitive")
        return signals

    @staticmethod
    def _ensure_trace_seed(thread_id: str, run_dir: Path) -> list[AgentTraceEvent]:
        events = []
        for artifact, role in {
            "plan.md": ResearchAgentRole.PLANNER,
            "notes.md": ResearchAgentRole.SOURCE_READER,
            "sources.json": ResearchAgentRole.SOURCE_TRIAGER,
            "report.md": ResearchAgentRole.FINAL_EDITOR,
        }.items():
            if (run_dir / artifact).exists():
                events.append(
                    AgentTraceEvent(
                        thread_id=thread_id,
                        role=role,
                        event_type=AgentTraceEventType.ARTIFACT_WRITTEN,
                        artifact_name=artifact,
                        message="control-plane synthetic trace seed",
                    )
                )
        return events

    @staticmethod
    def _recommended_actions(
        missing: list[str], invalid: list[str], violations: list[Any]
    ) -> list[str]:
        actions = []
        if missing:
            actions.append("Regenerate or backfill missing required artifacts.")
        if invalid:
            actions.append("Inspect invalid artifacts before trusting the report.")
        if violations:
            actions.append("Review policy violations and tighten role/tool wrappers.")
        return actions

    @staticmethod
    def _question_from_run(run_dir: Path) -> str | None:
        path = run_dir / "run.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("question")
        except Exception:
            return None

    @staticmethod
    def _urls_from_sources(run_dir: Path) -> list[str]:
        path = run_dir / "sources.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        return [
            str(item.get("url") or item.get("final_url") or "")
            for item in data
            if isinstance(item, dict)
        ]
