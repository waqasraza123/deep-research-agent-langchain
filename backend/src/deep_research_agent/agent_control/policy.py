from __future__ import annotations

from pathlib import Path

from .contracts import (
    AgentControlPlan,
    FilesystemPolicy,
    FilesystemWriteMode,
    PolicyViolation,
    ResearchAgentRole,
    ToolPolicy,
    stable_id,
)

TOOL_CATEGORIES = {
    "fetch_and_store": "source_fetch",
    "fetch_document": "source_fetch",
    "artifact_read": "artifact_read",
    "artifact_write": "artifact_write",
    "filesystem_read": "filesystem_read",
    "filesystem_write": "filesystem_write",
    "model_call": "model_call",
    "subagent_call": "subagent_call",
}


def _base_allowed(role: ResearchAgentRole) -> tuple[list[str], bool, bool, bool, int]:
    if role in {ResearchAgentRole.SOURCE_TRIAGER, ResearchAgentRole.SOURCE_READER}:
        return ["source_fetch", "artifact_read", "artifact_write"], True, True, True, 12
    if role == ResearchAgentRole.SUPERVISOR:
        return (
            ["subagent_call", "artifact_read", "artifact_write", "model_call"],
            False,
            True,
            True,
            4,
        )
    if role in {
        ResearchAgentRole.PLANNER,
        ResearchAgentRole.EVIDENCE_EXTRACTOR,
        ResearchAgentRole.SKEPTICAL_REVIEWER,
        ResearchAgentRole.TECHNICAL_ANALYST,
        ResearchAgentRole.COMPARISON_ANALYST,
        ResearchAgentRole.RISK_REVIEWER,
        ResearchAgentRole.CITATION_AUDITOR,
        ResearchAgentRole.SYNTHESIS_WRITER,
        ResearchAgentRole.FINAL_EDITOR,
    }:
        return ["artifact_read", "artifact_write", "model_call"], False, True, True, 2
    return ["artifact_read"], False, True, False, 0


def _role_required_artifacts(role: ResearchAgentRole) -> list[str]:
    mapping = {
        ResearchAgentRole.PLANNER: ["plan.md"],
        ResearchAgentRole.SOURCE_READER: ["notes.md"],
        ResearchAgentRole.SOURCE_TRIAGER: ["sources.json"],
        ResearchAgentRole.SYNTHESIS_WRITER: ["report.raw.md"],
        ResearchAgentRole.FINAL_EDITOR: ["report.md"],
        ResearchAgentRole.SUPERVISOR: ["plan.md", "notes.md", "sources.json", "report.md"],
    }
    return mapping.get(role, [])


def _role_optional_artifacts(role: ResearchAgentRole) -> list[str]:
    mapping = {
        ResearchAgentRole.SOURCE_TRIAGER: ["source_rankings.json", "source_warnings.md"],
        ResearchAgentRole.SOURCE_READER: ["source_notes.md"],
        ResearchAgentRole.EVIDENCE_EXTRACTOR: [
            "evidence_ledger.json",
            "evidence_ledger.md",
            "citation_map.json",
        ],
        ResearchAgentRole.SKEPTICAL_REVIEWER: [
            "contradictions.md",
            "claim_rewrite_suggestions.json",
        ],
        ResearchAgentRole.TECHNICAL_ANALYST: ["technical_analysis.md"],
        ResearchAgentRole.COMPARISON_ANALYST: ["comparison_matrix.json", "comparison_matrix.md"],
        ResearchAgentRole.RISK_REVIEWER: ["risk_register.json", "risk_register.md"],
        ResearchAgentRole.CITATION_AUDITOR: ["citation_audit.md", "citation_readiness.json"],
        ResearchAgentRole.SYNTHESIS_WRITER: ["synthesis_output.json", "report.raw.md"],
        ResearchAgentRole.FINAL_EDITOR: ["report.md"],
    }
    return mapping.get(role, [])


class PolicyEngine:
    def __init__(self, *, thread_id: str = "preview", strict_quarantine: bool = True):
        self.thread_id = thread_id
        self.strict_quarantine = strict_quarantine
        self.tool_policies: dict[ResearchAgentRole, ToolPolicy] = {}
        self.filesystem_policies: dict[ResearchAgentRole, FilesystemPolicy] = {}
        self.violations: list[PolicyViolation] = []

    def build_policies_for_plan(
        self, selected_roles: list[ResearchAgentRole] | AgentControlPlan
    ) -> tuple[list[ToolPolicy], list[FilesystemPolicy]]:
        if isinstance(selected_roles, AgentControlPlan):
            roles = [role.role for role in selected_roles.selected_roles]
            self.thread_id = selected_roles.thread_id
        else:
            roles = selected_roles
        tool_policies: list[ToolPolicy] = []
        fs_policies: list[FilesystemPolicy] = []
        for role in roles:
            allowed, network, file_read, file_write, max_fetches = _base_allowed(role)
            denied = ["unsafe_network", "secret_access"]
            if role in {ResearchAgentRole.SKEPTICAL_REVIEWER, ResearchAgentRole.FINAL_EDITOR}:
                denied.append("source_fetch")
            tool_policy = ToolPolicy(
                policy_id=stable_id("tool_policy", self.thread_id, role.value),
                role=role,
                allowed_tools=allowed,
                denied_tools=denied,
                allow_network=network,
                allow_file_read=file_read,
                allow_file_write=file_write,
                allow_artifact_download=file_read,
                allow_source_fetch=network,
                max_fetches=max_fetches,
                max_tool_calls=48 if role == ResearchAgentRole.SUPERVISOR else 24,
                warnings=[],
            )
            writable = [*_role_required_artifacts(role), *_role_optional_artifacts(role)]
            readable = ["*"]
            forbidden = ["../", "/etc", ".env", "raw_secrets", "sources/raw_unsafe"]
            if self.strict_quarantine and role == ResearchAgentRole.FINAL_EDITOR:
                forbidden.extend(["sources/", "sanitized_sources/"])
            fs_policy = FilesystemPolicy(
                policy_id=stable_id("fs_policy", self.thread_id, role.value),
                role=role,
                readable_paths=readable,
                writable_paths=sorted(set(writable)),
                forbidden_paths=forbidden,
                required_artifacts=_role_required_artifacts(role),
                optional_artifacts=_role_optional_artifacts(role),
                write_mode=FilesystemWriteMode.CONTROLLED_ARTIFACTS_ONLY,
            )
            tool_policies.append(tool_policy)
            fs_policies.append(fs_policy)
            self.tool_policies[role] = tool_policy
            self.filesystem_policies[role] = fs_policy
        return tool_policies, fs_policies

    def _tool_policy(self, role: ResearchAgentRole) -> ToolPolicy:
        return self.tool_policies.get(role) or self.build_policies_for_plan([role])[0][0]

    def _fs_policy(self, role: ResearchAgentRole) -> FilesystemPolicy:
        if role not in self.filesystem_policies:
            self.build_policies_for_plan([role])
        return self.filesystem_policies[role]

    def is_tool_allowed(
        self, role: ResearchAgentRole, tool_name: str, tool_category: str | None = None
    ) -> bool:
        category = tool_category or TOOL_CATEGORIES.get(tool_name, tool_name)
        policy = self._tool_policy(role)
        if category in policy.denied_tools or tool_name in policy.denied_tools:
            return False
        if category == "source_fetch" and not policy.allow_source_fetch:
            return False
        return category in policy.allowed_tools or tool_name in policy.allowed_tools

    def is_artifact_read_allowed(self, role: ResearchAgentRole, artifact_name: str) -> bool:
        policy = self._fs_policy(role)
        if self._has_forbidden_path(policy, artifact_name):
            return False
        return "*" in policy.readable_paths or artifact_name in policy.readable_paths

    def is_artifact_write_allowed(self, role: ResearchAgentRole, artifact_name: str) -> bool:
        policy = self._fs_policy(role)
        if self._has_forbidden_path(policy, artifact_name):
            return False
        if policy.write_mode == FilesystemWriteMode.DISABLED:
            return False
        return artifact_name in policy.writable_paths

    def validate_filesystem_path(self, role: ResearchAgentRole, path: str | Path) -> bool:
        value = str(path).replace("\\", "/")
        if value.startswith("/") or ".." in value:
            self.record_violation(
                role=role,
                policy_type="filesystem",
                message="Path traversal or absolute path blocked.",
                attempted_path=value,
                severity="error",
            )
            return False
        policy = self._fs_policy(role)
        if self._has_forbidden_path(policy, value):
            self.record_violation(
                role=role,
                policy_type="filesystem",
                message="Forbidden filesystem path blocked.",
                attempted_path=value,
                severity="error",
            )
            return False
        return True

    def record_violation(
        self,
        *,
        role: ResearchAgentRole,
        policy_type: str,
        message: str,
        severity: str = "warning",
        attempted_tool: str | None = None,
        attempted_path: str | None = None,
        attempted_artifact: str | None = None,
        recommended_action: str = "Keep the role inside its declared policy.",
    ) -> PolicyViolation:
        violation = PolicyViolation(
            thread_id=self.thread_id,
            role=role,
            policy_type=policy_type,
            severity=severity,
            message=message,
            attempted_tool=attempted_tool,
            attempted_path=attempted_path,
            attempted_artifact=attempted_artifact,
            recommended_action=recommended_action,
        )
        self.violations.append(violation)
        return violation

    def check_tool(
        self, role: ResearchAgentRole, tool_name: str, tool_category: str | None = None
    ) -> bool:
        allowed = self.is_tool_allowed(role, tool_name, tool_category)
        if not allowed:
            self.record_violation(
                role=role,
                policy_type="tool",
                message=f"Denied tool use: {tool_name}",
                attempted_tool=tool_name,
            )
        return allowed

    def summarize_policy_warnings(self) -> list[str]:
        warnings = []
        for violation in self.violations:
            warnings.append(f"{violation.role.value}: {violation.message}")
        return warnings

    @staticmethod
    def _has_forbidden_path(policy: FilesystemPolicy, value: str) -> bool:
        normalized = value.replace("\\", "/")
        return any(fragment and fragment in normalized for fragment in policy.forbidden_paths)
