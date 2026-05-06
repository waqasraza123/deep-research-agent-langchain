from __future__ import annotations

from .artifact_writer import render_handoffs, write_json_artifact, write_text_artifact
from .contracts import (
    AgentControlPlan,
    AgentHandoff,
    HandoffStatus,
    ResearchAgentRole,
    now_iso_utc,
    stable_id,
)


def _has(plan: AgentControlPlan, role: ResearchAgentRole) -> bool:
    return any(item.role == role for item in plan.selected_roles)


def _handoff(
    thread_id: str,
    from_role: ResearchAgentRole,
    to_role: ResearchAgentRole,
    reason: str,
    expected: str,
    artifacts: list[str],
) -> AgentHandoff:
    return AgentHandoff(
        handoff_id=stable_id("handoff", thread_id, from_role.value, to_role.value, reason),
        thread_id=thread_id,
        from_role=from_role,
        to_role=to_role,
        reason=reason,
        input_summary=f"{from_role.value} output",
        expected_output=expected,
        required_artifacts=artifacts,
    )


def build_planned_handoffs(control_plan: AgentControlPlan) -> list[AgentHandoff]:
    tid = control_plan.thread_id
    edges = [
        (
            ResearchAgentRole.SUPERVISOR,
            ResearchAgentRole.PLANNER,
            "start governed planning",
            "plan structure and evidence needs",
            ["plan.md"],
        ),
        (
            ResearchAgentRole.PLANNER,
            ResearchAgentRole.SOURCE_TRIAGER,
            "triage evidence sources",
            "source priority and safety warnings",
            ["sources.json"],
        ),
        (
            ResearchAgentRole.SOURCE_TRIAGER,
            ResearchAgentRole.SOURCE_READER,
            "read prioritized safe sources",
            "bounded extracted facts",
            ["notes.md"],
        ),
        (
            ResearchAgentRole.SOURCE_READER,
            ResearchAgentRole.EVIDENCE_EXTRACTOR,
            "convert facts into evidence",
            "evidence units",
            ["evidence_ledger.json"],
        ),
        (
            ResearchAgentRole.EVIDENCE_EXTRACTOR,
            ResearchAgentRole.SKEPTICAL_REVIEWER,
            "challenge evidence and claims",
            "critique warnings",
            ["contradictions.md"],
        ),
        (
            ResearchAgentRole.SKEPTICAL_REVIEWER,
            ResearchAgentRole.CITATION_AUDITOR,
            "audit citation readiness",
            "citation gaps",
            ["citation_audit.md"],
        ),
        (
            ResearchAgentRole.SKEPTICAL_REVIEWER,
            ResearchAgentRole.SYNTHESIS_WRITER,
            "send critique to synthesis",
            "draft report",
            ["report.raw.md"],
        ),
        (
            ResearchAgentRole.SYNTHESIS_WRITER,
            ResearchAgentRole.FINAL_EDITOR,
            "polish draft without adding facts",
            "final report",
            ["report.md"],
        ),
        (
            ResearchAgentRole.FINAL_EDITOR,
            ResearchAgentRole.SUPERVISOR,
            "final validation",
            "artifact checklist",
            ["report.md"],
        ),
    ]
    if _has(control_plan, ResearchAgentRole.COMPARISON_ANALYST):
        edges.insert(
            5,
            (
                ResearchAgentRole.EVIDENCE_EXTRACTOR,
                ResearchAgentRole.COMPARISON_ANALYST,
                "comparative evidence analysis",
                "comparison matrix",
                ["comparison_matrix.md"],
            ),
        )
        edges.insert(
            6,
            (
                ResearchAgentRole.COMPARISON_ANALYST,
                ResearchAgentRole.SYNTHESIS_WRITER,
                "include balanced tradeoffs",
                "comparison-aware synthesis notes",
                ["comparison_matrix.md"],
            ),
        )
    if _has(control_plan, ResearchAgentRole.TECHNICAL_ANALYST):
        edges.insert(
            5,
            (
                ResearchAgentRole.EVIDENCE_EXTRACTOR,
                ResearchAgentRole.TECHNICAL_ANALYST,
                "technical due diligence",
                "technical risks and limits",
                ["technical_analysis.md"],
            ),
        )
        edges.insert(
            6,
            (
                ResearchAgentRole.TECHNICAL_ANALYST,
                ResearchAgentRole.SYNTHESIS_WRITER,
                "include technical assessment",
                "technical synthesis notes",
                ["technical_analysis.md"],
            ),
        )
    if _has(control_plan, ResearchAgentRole.RISK_REVIEWER):
        edges.insert(
            -2,
            (
                ResearchAgentRole.RISK_REVIEWER,
                ResearchAgentRole.SYNTHESIS_WRITER,
                "include risk register",
                "risk-aware synthesis notes",
                ["risk_register.md"],
            ),
        )
    selected = {role.role for role in control_plan.selected_roles}
    return [
        _handoff(tid, from_role, to_role, reason, expected, artifacts)
        for from_role, to_role, reason, expected, artifacts in edges
        if from_role in selected and to_role in selected
    ]


def record_handoff_started(handoff: AgentHandoff) -> AgentHandoff:
    return handoff.copy(update={"status": HandoffStatus.STARTED})


def record_handoff_completed(handoff: AgentHandoff) -> AgentHandoff:
    return handoff.copy(update={"status": HandoffStatus.COMPLETED, "completed_at": now_iso_utc()})


def validate_handoff_output(handoff: AgentHandoff, available_artifacts: set[str]) -> AgentHandoff:
    missing = [item for item in handoff.required_artifacts if item not in available_artifacts]
    if missing:
        return handoff.copy(
            update={
                "status": HandoffStatus.INVALID,
                "warnings": [*handoff.warnings, f"missing artifacts: {', '.join(missing)}"],
            }
        )
    return handoff


def detect_missing_handoffs(planned: list[AgentHandoff], actual: list[AgentHandoff]) -> list[str]:
    completed = {
        (item.from_role, item.to_role) for item in actual if item.status == HandoffStatus.COMPLETED
    }
    return [
        f"{item.from_role.value}->{item.to_role.value}"
        for item in planned
        if (item.from_role, item.to_role) not in completed
    ]


def summarize_handoffs(handoffs: list[AgentHandoff]) -> dict[str, int]:
    return {
        "planned": len(handoffs),
        "completed": sum(1 for item in handoffs if item.status == HandoffStatus.COMPLETED),
        "invalid": sum(1 for item in handoffs if item.status == HandoffStatus.INVALID),
    }


def write_handoff_artifacts(runs_dir, thread_id: str, handoffs: list[AgentHandoff]) -> list[str]:
    return [
        write_json_artifact(runs_dir, thread_id, "agent_handoffs.json", handoffs),
        write_text_artifact(runs_dir, thread_id, "agent_handoffs.md", render_handoffs(handoffs)),
    ]
