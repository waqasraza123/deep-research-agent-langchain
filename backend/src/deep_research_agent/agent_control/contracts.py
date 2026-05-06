from __future__ import annotations

import json
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_id(prefix: str, *parts: Any) -> str:
    import hashlib

    raw = "|".join(str(part) for part in parts if part is not None)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def model_to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [model_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [model_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): model_to_plain(item) for key, item in value.items()}
    return value


def json_text(value: Any) -> str:
    return json.dumps(model_to_plain(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


class RoleIsolationLevel(str, Enum):
    NONE = "none"
    LIGHT = "light"
    STRICT = "strict"
    QUARANTINE = "quarantine"


class ResearchAgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    SOURCE_TRIAGER = "source_triager"
    SOURCE_READER = "source_reader"
    EVIDENCE_EXTRACTOR = "evidence_extractor"
    SKEPTICAL_REVIEWER = "skeptical_reviewer"
    TECHNICAL_ANALYST = "technical_analyst"
    COMPARISON_ANALYST = "comparison_analyst"
    RISK_REVIEWER = "risk_reviewer"
    CITATION_AUDITOR = "citation_auditor"
    SYNTHESIS_WRITER = "synthesis_writer"
    FINAL_EDITOR = "final_editor"
    UNKNOWN = "unknown"


class AgentSkillType(str, Enum):
    PLANNING = "planning"
    SOURCE_TRIAGE = "source_triage"
    SOURCE_READING = "source_reading"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    CITATION_REVIEW = "citation_review"
    CONTRADICTION_DETECTION = "contradiction_detection"
    SYNTHESIS = "synthesis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    COMPARISON_ANALYSIS = "comparison_analysis"
    RISK_REVIEW = "risk_review"
    TEMPORAL_REVIEW = "temporal_review"
    QUANTITATIVE_REVIEW = "quantitative_review"
    SOURCE_SAFETY_REVIEW = "source_safety_review"
    FINAL_EDITING = "final_editing"
    ARTIFACT_VALIDATION = "artifact_validation"


class FilesystemWriteMode(str, Enum):
    DISABLED = "disabled"
    APPEND_ONLY = "append_only"
    OVERWRITE_ALLOWED = "overwrite_allowed"
    CONTROLLED_ARTIFACTS_ONLY = "controlled_artifacts_only"


class ContextTrustLevel(str, Enum):
    TRUSTED_INSTRUCTION = "trusted_instruction"
    TRUSTED_POLICY = "trusted_policy"
    TRUSTED_ARTIFACT = "trusted_artifact"
    UNTRUSTED_SOURCE = "untrusted_source"
    UNTRUSTED_USER_SUPPLIED = "untrusted_user_supplied"
    GENERATED_DRAFT = "generated_draft"


class HandoffStatus(str, Enum):
    PLANNED = "planned"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INVALID = "invalid"


class AgentTraceEventType(str, Enum):
    ROLE_STARTED = "role_started"
    ROLE_COMPLETED = "role_completed"
    TOOL_ALLOWED = "tool_allowed"
    TOOL_DENIED = "tool_denied"
    ARTIFACT_READ = "artifact_read"
    ARTIFACT_WRITTEN = "artifact_written"
    HANDOFF_STARTED = "handoff_started"
    HANDOFF_COMPLETED = "handoff_completed"
    POLICY_WARNING = "policy_warning"
    POLICY_VIOLATION = "policy_violation"
    CONTEXT_TRUNCATED = "context_truncated"
    ROLE_CONFUSION_DETECTED = "role_confusion_detected"
    OUTPUT_VALIDATION_FAILED = "output_validation_failed"
    OUTPUT_VALIDATION_PASSED = "output_validation_passed"


class AgentControlSettings(BaseModel):
    enabled: bool = True
    offline_mode: bool = True
    strict_role_isolation: bool = True
    source_context_quarantine_enabled: bool = True
    tool_governance_enabled: bool = True
    filesystem_governance_enabled: bool = True
    skill_selection_enabled: bool = True
    subagent_planning_enabled: bool = True
    trace_analysis_enabled: bool = True
    artifact_validation_enabled: bool = True
    max_compiled_instruction_chars: int = 12_000
    max_subagents: int = 8
    max_handoffs: int = 32
    max_context_chars_per_role: int = 16_000
    fail_on_policy_violation: bool = False
    fail_on_missing_required_artifact: bool = False
    allow_mock_subagents: bool = True
    produce_markdown_artifacts: bool = True
    produce_json_artifacts: bool = True

    @classmethod
    def from_runtime(cls, runtime_settings: Any, **overrides: Any) -> "AgentControlSettings":
        data = {
            "enabled": getattr(runtime_settings, "agent_control_enabled", True),
            "offline_mode": getattr(runtime_settings, "intelligence_offline_mode", True),
            "strict_role_isolation": getattr(
                runtime_settings, "agent_control_strict_role_isolation", True
            ),
            "source_context_quarantine_enabled": getattr(
                runtime_settings, "agent_control_source_context_quarantine_enabled", True
            ),
            "tool_governance_enabled": getattr(
                runtime_settings, "agent_control_tool_governance_enabled", True
            ),
            "filesystem_governance_enabled": getattr(
                runtime_settings, "agent_control_filesystem_governance_enabled", True
            ),
            "skill_selection_enabled": getattr(
                runtime_settings, "agent_control_skill_selection_enabled", True
            ),
            "subagent_planning_enabled": getattr(
                runtime_settings, "agent_control_subagent_planning_enabled", True
            ),
            "trace_analysis_enabled": getattr(
                runtime_settings, "agent_control_trace_analysis_enabled", True
            ),
            "artifact_validation_enabled": getattr(
                runtime_settings, "agent_control_artifact_validation_enabled", True
            ),
            "max_compiled_instruction_chars": getattr(
                runtime_settings, "agent_control_max_compiled_instruction_chars", 12_000
            ),
            "max_subagents": getattr(runtime_settings, "agent_control_max_subagents", 8),
            "max_handoffs": getattr(runtime_settings, "agent_control_max_handoffs", 32),
            "max_context_chars_per_role": getattr(
                runtime_settings, "agent_control_max_context_chars_per_role", 16_000
            ),
            "fail_on_policy_violation": getattr(
                runtime_settings, "agent_control_fail_on_policy_violation", False
            ),
            "fail_on_missing_required_artifact": getattr(
                runtime_settings, "agent_control_fail_on_missing_required_artifact", False
            ),
            "allow_mock_subagents": getattr(
                runtime_settings, "agent_control_allow_mock_subagents", True
            ),
            "produce_markdown_artifacts": getattr(
                runtime_settings, "agent_control_produce_markdown_artifacts", True
            ),
            "produce_json_artifacts": getattr(
                runtime_settings, "agent_control_produce_json_artifacts", True
            ),
        }
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


class AgentRoleDefinition(BaseModel):
    role: ResearchAgentRole
    name: str
    description: str
    purpose: str
    responsibilities: list[str] = Field(default_factory=list)
    forbidden_behaviors: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    readable_artifacts: list[str] = Field(default_factory=list)
    writable_artifacts: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    max_context_chars: int = 8000
    isolation_level: RoleIsolationLevel = RoleIsolationLevel.STRICT
    system_instruction_template: str = ""
    handoff_contracts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AgentSkillDefinition(BaseModel):
    skill_id: str
    name: str
    skill_type: AgentSkillType
    description: str
    trigger_signals: list[str] = Field(default_factory=list)
    required_role: ResearchAgentRole | None = None
    allowed_roles: list[ResearchAgentRole] = Field(default_factory=list)
    instruction_block: str
    required_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    optional_artifacts: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    priority: int = 50
    enabled_by_default: bool = True


class SkillSelection(BaseModel):
    question: str
    selected_skills: list[AgentSkillDefinition] = Field(default_factory=list)
    skipped_skills: list[str] = Field(default_factory=list)
    selection_reasons: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ToolPolicy(BaseModel):
    policy_id: str
    role: ResearchAgentRole
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    allowed_domains: list[str] = Field(default_factory=list)
    denied_domains: list[str] = Field(default_factory=list)
    allow_network: bool = False
    allow_file_read: bool = False
    allow_file_write: bool = False
    allow_artifact_download: bool = False
    allow_source_fetch: bool = False
    max_tool_calls: int = 32
    max_fetches: int = 0
    max_write_bytes: int = 1_000_000
    warnings: list[str] = Field(default_factory=list)


class FilesystemPolicy(BaseModel):
    policy_id: str
    role: ResearchAgentRole
    readable_paths: list[str] = Field(default_factory=list)
    writable_paths: list[str] = Field(default_factory=list)
    forbidden_paths: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    optional_artifacts: list[str] = Field(default_factory=list)
    write_mode: FilesystemWriteMode = FilesystemWriteMode.CONTROLLED_ARTIFACTS_ONLY
    path_traversal_protection: bool = True
    max_artifact_bytes: int = 5_000_000
    warnings: list[str] = Field(default_factory=list)


class ContextBundle(BaseModel):
    bundle_id: str
    role: ResearchAgentRole
    question: str
    trusted_context: str = ""
    untrusted_source_context: str = ""
    artifact_context: str = ""
    policy_context: str = ""
    max_chars: int = 16_000
    truncation_applied: bool = False
    warnings: list[str] = Field(default_factory=list)


class CompiledInstruction(BaseModel):
    instruction_id: str
    role: ResearchAgentRole
    system_instructions: str
    role_instructions: str
    skill_instructions: str
    policy_instructions: str
    source_boundary_instructions: str
    artifact_requirements: list[str] = Field(default_factory=list)
    forbidden_behaviors: list[str] = Field(default_factory=list)
    expected_output_format: str = ""
    total_chars: int = 0
    warnings: list[str] = Field(default_factory=list)


class SubagentSpec(BaseModel):
    subagent_id: str
    role: ResearchAgentRole
    name: str
    description: str
    system_prompt: str
    tools: list[str] = Field(default_factory=list)
    readable_artifacts: list[str] = Field(default_factory=list)
    writable_artifacts: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    context_limits: dict[str, int] = Field(default_factory=dict)
    isolation_level: RoleIsolationLevel = RoleIsolationLevel.STRICT
    expected_outputs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AgentHandoff(BaseModel):
    handoff_id: str
    thread_id: str
    from_role: ResearchAgentRole
    to_role: ResearchAgentRole
    reason: str
    input_summary: str
    expected_output: str
    required_artifacts: list[str] = Field(default_factory=list)
    status: HandoffStatus = HandoffStatus.PLANNED
    created_at: str = Field(default_factory=now_iso_utc)
    completed_at: str | None = None
    warnings: list[str] = Field(default_factory=list)


class AgentTraceEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    thread_id: str
    role: ResearchAgentRole = ResearchAgentRole.UNKNOWN
    event_type: AgentTraceEventType
    timestamp: str = Field(default_factory=now_iso_utc)
    tool_name: str | None = None
    artifact_name: str | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PolicyViolation(BaseModel):
    violation_id: str = Field(default_factory=lambda: f"viol_{uuid.uuid4().hex[:12]}")
    thread_id: str
    role: ResearchAgentRole
    policy_type: str
    severity: str = "warning"
    message: str
    attempted_tool: str | None = None
    attempted_path: str | None = None
    attempted_artifact: str | None = None
    recommended_action: str = ""
    created_at: str = Field(default_factory=now_iso_utc)


class AgentOutputValidation(BaseModel):
    validation_id: str = Field(default_factory=lambda: f"val_{uuid.uuid4().hex[:12]}")
    thread_id: str
    role: ResearchAgentRole = ResearchAgentRole.UNKNOWN
    artifact_name: str
    expected: bool = True
    exists: bool = False
    valid: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso_utc)


class AgentControlPlan(BaseModel):
    plan_id: str
    thread_id: str
    question: str
    detected_intent: list[str] = Field(default_factory=list)
    selected_roles: list[AgentRoleDefinition] = Field(default_factory=list)
    selected_skills: SkillSelection
    supervisor_instruction: CompiledInstruction | None = None
    subagents: list[SubagentSpec] = Field(default_factory=list)
    tool_policies: list[ToolPolicy] = Field(default_factory=list)
    filesystem_policies: list[FilesystemPolicy] = Field(default_factory=list)
    context_bundles: list[ContextBundle] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    expected_handoffs: list[AgentHandoff] = Field(default_factory=list)
    policy_warnings: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=now_iso_utc)


class AgentControlSummary(BaseModel):
    thread_id: str
    question: str
    roles_selected: list[str] = Field(default_factory=list)
    skills_selected: list[str] = Field(default_factory=list)
    subagents_created: int = 0
    handoffs_planned: int = 0
    handoffs_completed: int = 0
    policy_violations: int = 0
    missing_artifacts: list[str] = Field(default_factory=list)
    invalid_outputs: list[str] = Field(default_factory=list)
    trace_event_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    generated_artifacts: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


class ArtifactContract(BaseModel):
    artifact_name: str
    producer_role: ResearchAgentRole
    required: bool = True
    format: str = "markdown"
    min_size_bytes: int = 1
    max_size_bytes: int = 5_000_000
    must_contain_sections: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)
    validation_rules: list[str] = Field(default_factory=list)
    description: str = ""
