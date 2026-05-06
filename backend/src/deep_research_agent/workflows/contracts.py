from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def model_to_plain(value: Any) -> Any:
    if isinstance(value, list):
        return [model_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [model_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): model_to_plain(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return value.value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(value, BaseModel):
        return value.dict()
    return value


class WorkflowModel(BaseModel):
    def to_json_dict(self) -> dict[str, Any]:
        return model_to_plain(self)

    def to_markdown(self) -> str:
        lines = [f"# {self.__class__.__name__}", ""]
        for key, value in self.to_json_dict().items():
            if isinstance(value, (dict, list)):
                lines.extend([f"## {key}", "", "```json"])
                lines.append(json.dumps(value, indent=2, sort_keys=True))
                lines.extend(["```", ""])
            else:
                lines.append(f"- **{key}**: {value}")
        return "\n".join(lines).rstrip() + "\n"


class WorkflowMode(str, Enum):
    quick_brief = "quick_brief"
    deep_research = "deep_research"
    technical_due_diligence = "technical_due_diligence"
    framework_comparison = "framework_comparison"
    vendor_evaluation = "vendor_evaluation"
    legal_policy_review = "legal_policy_review"
    market_research = "market_research"
    implementation_planning = "implementation_planning"
    source_audit_only = "source_audit_only"
    evidence_extraction_only = "evidence_extraction_only"
    report_generation_only = "report_generation_only"
    verification_only = "verification_only"
    evaluation_only = "evaluation_only"
    adversarial_source_review = "adversarial_source_review"
    offline_benchmark = "offline_benchmark"
    rebuild_from_artifacts = "rebuild_from_artifacts"
    custom = "custom"


class WorkflowInputType(str, Enum):
    question = "question"
    urls = "urls"
    existing_run = "existing_run"
    existing_artifacts = "existing_artifacts"
    local_sources = "local_sources"
    settings = "settings"
    metadata = "metadata"


class WorkflowStageType(str, Enum):
    input_snapshot = "input_snapshot"
    request_analysis = "request_analysis"
    protocol_selection = "protocol_selection"
    source_discovery = "source_discovery"
    source_fetching = "source_fetching"
    source_safety = "source_safety"
    document_intelligence = "document_intelligence"
    source_audit = "source_audit"
    retrieval_indexing = "retrieval_indexing"
    context_pack_building = "context_pack_building"
    agent_control_planning = "agent_control_planning"
    agent_execution = "agent_execution"
    artifact_backfill = "artifact_backfill"
    evidence_extraction = "evidence_extraction"
    hypothesis_testing = "hypothesis_testing"
    temporal_analysis = "temporal_analysis"
    quantitative_analysis = "quantitative_analysis"
    synthesis = "synthesis"
    verification = "verification"
    evaluation = "evaluation"
    quality_gate = "quality_gate"
    provenance = "provenance"
    finalization = "finalization"
    custom = "custom"


class WorkflowFailureBehavior(str, Enum):
    fail_workflow = "fail_workflow"
    warn_and_continue = "warn_and_continue"
    skip_downstream = "skip_downstream"
    mark_degraded = "mark_degraded"
    require_review = "require_review"


class ArtifactType(str, Enum):
    markdown = "markdown"
    json = "json"
    jsonl = "jsonl"
    text = "text"
    csv = "csv"
    binary = "binary"
    directory = "directory"
    unknown = "unknown"


class WorkflowStageStatus(str, Enum):
    pending = "pending"
    ready = "ready"
    running = "running"
    completed = "completed"
    skipped = "skipped"
    failed = "failed"
    degraded = "degraded"


class WorkflowExecutionStatus(str, Enum):
    compiled = "compiled"
    running = "running"
    completed = "completed"
    completed_with_warnings = "completed_with_warnings"
    degraded = "degraded"
    failed = "failed"
    cancelled = "cancelled"
    skipped = "skipped"


class WorkflowReadinessStatus(str, Enum):
    ready = "ready"
    ready_with_warnings = "ready_with_warnings"
    degraded = "degraded"
    blocked = "blocked"
    failed = "failed"


class WarningSeverity(str, Enum):
    info = "info"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ArtifactValidationStatus(str, Enum):
    passed = "passed"
    failed = "failed"
    warning = "warning"
    skipped = "skipped"


class WorkflowWarning(WorkflowModel):
    warning_id: str = Field(default_factory=lambda: f"ww-{uuid.uuid4().hex[:10]}")
    code: str
    severity: WarningSeverity = WarningSeverity.medium
    stage_id: str | None = None
    message: str
    affected_artifacts: list[str] = Field(default_factory=list)
    affected_stages: list[str] = Field(default_factory=list)
    recommended_action: str = ""


class WorkflowInputRequirement(WorkflowModel):
    name: str
    required: bool = True
    input_type: WorkflowInputType
    description: str = ""
    default: Any = None
    validation_rules: dict[str, Any] = Field(default_factory=dict)


class WorkflowStageDefinition(WorkflowModel):
    stage_id: str
    name: str
    stage_type: WorkflowStageType
    description: str = ""
    required: bool = True
    enabled_by_default: bool = True
    depends_on: list[str] = Field(default_factory=list)
    optional_depends_on: list[str] = Field(default_factory=list)
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    consumes_runtime_services: list[str] = Field(default_factory=list)
    produces_operator_warnings: bool = False
    retryable: bool = False
    timeout_seconds: int | None = Field(default=None, ge=1)
    max_attempts: int = Field(default=1, ge=1)
    skip_if_artifacts_exist: list[str] = Field(default_factory=list)
    skip_conditions: list[str] = Field(default_factory=list)
    failure_behavior: WorkflowFailureBehavior = WorkflowFailureBehavior.fail_workflow
    settings_overrides: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("stage_id")
    @classmethod
    def _validate_stage_id(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value or ".." in value:
            raise ValueError("stage_id must be a safe identifier")
        return value


class WorkflowArtifactContract(WorkflowModel):
    artifact_name: str
    artifact_type: ArtifactType = ArtifactType.unknown
    required: bool = True
    producer_stage: str = ""
    consumer_stages: list[str] = Field(default_factory=list)
    format: str = ""
    min_size_bytes: int = Field(default=0, ge=0)
    max_size_bytes: int | None = Field(default=None, ge=1)
    must_parse_as_json: bool = False
    must_be_nonempty: bool = False
    allowed_missing_when: list[str] = Field(default_factory=list)
    validation_rules: dict[str, Any] = Field(default_factory=dict)
    description: str = ""

    @field_validator("artifact_name")
    @classmethod
    def _validate_artifact_name(cls, value: str) -> str:
        if not value or value.startswith("/") or "\\" in value or ".." in value.split("/"):
            raise ValueError("artifact_name must be relative and safe")
        return value


class WorkflowPolicy(WorkflowModel):
    policy_id: str
    name: str
    description: str = ""
    strict_citations: bool = False
    freshness_required: bool = False
    source_discovery_allowed: bool = True
    external_network_allowed: bool = True
    model_required: bool = True
    mock_allowed: bool = False
    review_required: bool = False
    sensitive_domain: bool = False
    fail_on_critical_warning: bool = False
    run_quality_gate: bool = False
    quality_gate_id: str | None = None
    max_sources: int | None = Field(default=None, ge=0)
    max_artifacts: int | None = Field(default=None, ge=1)
    max_runtime_seconds: int | None = Field(default=None, ge=1)
    max_model_calls: int | None = Field(default=None, ge=0)
    max_output_chars: int | None = Field(default=None, ge=0)
    allowed_stage_types: list[WorkflowStageType] = Field(default_factory=list)
    denied_stage_types: list[WorkflowStageType] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowTemplate(WorkflowModel):
    template_id: str
    name: str
    description: str
    mode: WorkflowMode
    version: str = "1.0"
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)
    default_settings: dict[str, Any] = Field(default_factory=dict)
    required_inputs: list[WorkflowInputRequirement] = Field(default_factory=list)
    optional_inputs: list[WorkflowInputRequirement] = Field(default_factory=list)
    stages: list[WorkflowStageDefinition] = Field(default_factory=list)
    artifact_contracts: list[WorkflowArtifactContract] = Field(default_factory=list)
    policies: list[WorkflowPolicy] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    failure_policy: WorkflowFailureBehavior = WorkflowFailureBehavior.fail_workflow
    created_at: str = Field(default_factory=now_iso_utc)
    updated_at: str = Field(default_factory=now_iso_utc)


class WorkflowInput(WorkflowModel):
    question: str = ""
    urls: list[str] = Field(default_factory=list)
    thread_id: str | None = None
    mode: WorkflowMode | None = None
    template_id: str | None = None
    settings_overrides: dict[str, Any] = Field(default_factory=dict)
    requested_artifacts: list[str] = Field(default_factory=list)
    existing_run_id: str | None = None
    existing_thread_id: str | None = None
    dry_run: bool = False
    run_now: bool = True
    run_quality_gate: bool = False
    quality_gate_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowDependencyGraph(WorkflowModel):
    nodes: list[str] = Field(default_factory=list)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    cycles_detected: list[list[str]] = Field(default_factory=list)
    missing_dependencies: list[dict[str, str]] = Field(default_factory=list)
    execution_layers: list[list[str]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CompiledWorkflow(WorkflowModel):
    workflow_id: str = Field(default_factory=lambda: f"wf-{uuid.uuid4().hex[:12]}")
    thread_id: str
    template_id: str
    mode: WorkflowMode
    question: str = ""
    urls: list[str] = Field(default_factory=list)
    stages: list[WorkflowStageDefinition] = Field(default_factory=list)
    dependency_graph: WorkflowDependencyGraph = Field(default_factory=WorkflowDependencyGraph)
    artifact_contracts: list[WorkflowArtifactContract] = Field(default_factory=list)
    policies: list[WorkflowPolicy] = Field(default_factory=list)
    settings_snapshot: dict[str, Any] = Field(default_factory=dict)
    execution_order: list[str] = Field(default_factory=list)
    skipped_stages: dict[str, str] = Field(default_factory=dict)
    warnings: list[WorkflowWarning] = Field(default_factory=list)
    generated_at: str = Field(default_factory=now_iso_utc)


class WorkflowStageExecution(WorkflowModel):
    execution_id: str = Field(default_factory=lambda: f"wse-{uuid.uuid4().hex[:12]}")
    workflow_id: str
    thread_id: str
    stage_id: str
    stage_type: WorkflowStageType
    status: WorkflowStageStatus = WorkflowStageStatus.pending
    started_at: str | None = None
    completed_at: str | None = None
    skipped_at: str | None = None
    failed_at: str | None = None
    attempts: int = 0
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class WorkflowReadiness(WorkflowModel):
    status: WorkflowReadinessStatus
    usable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    required_human_review: bool = False


class WorkflowExecutionResult(WorkflowModel):
    workflow_id: str
    thread_id: str
    mode: WorkflowMode
    status: WorkflowExecutionStatus
    started_at: str
    completed_at: str | None = None
    stages: list[WorkflowStageExecution] = Field(default_factory=list)
    generated_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    warnings: list[WorkflowWarning] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    degraded: bool = False
    readiness: WorkflowReadiness | None = None
    quality_gate_result: dict[str, Any] | None = None
    summary: str = ""


class WorkflowPreview(WorkflowModel):
    mode: WorkflowMode
    template_id: str
    question: str = ""
    selected_template: dict[str, Any] = Field(default_factory=dict)
    stages: list[WorkflowStageDefinition] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    policies: list[WorkflowPolicy] = Field(default_factory=list)
    estimated_complexity: str = "low"
    skipped_stages: dict[str, str] = Field(default_factory=dict)
    warnings: list[WorkflowWarning] = Field(default_factory=list)


class WorkflowManifest(WorkflowModel):
    workflow_id: str
    thread_id: str
    template_id: str
    mode: WorkflowMode
    input_fingerprint: str
    settings_fingerprint: str
    stage_count: int = 0
    artifact_contract_count: int = 0
    generated_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    completed_stages: list[str] = Field(default_factory=list)
    failed_stages: list[str] = Field(default_factory=list)
    skipped_stages: list[str] = Field(default_factory=list)
    degraded_stages: list[str] = Field(default_factory=list)
    quality_gate_status: str | None = None
    created_at: str = Field(default_factory=now_iso_utc)
    updated_at: str = Field(default_factory=now_iso_utc)
    warnings: list[str] = Field(default_factory=list)


class ArtifactValidationResult(WorkflowModel):
    artifact_name: str
    status: ArtifactValidationStatus
    required: bool
    exists: bool
    size_bytes: int = 0
    messages: list[str] = Field(default_factory=list)


class ArtifactValidationReport(WorkflowModel):
    status: ArtifactValidationStatus
    results: list[ArtifactValidationResult] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WorkflowRebuildRequest(WorkflowModel):
    thread_id: str
    mode: WorkflowMode = WorkflowMode.rebuild_from_artifacts
    stages: list[str] = Field(default_factory=list)
    force: bool = False
    allow_model_calls: bool = False
    allow_source_refetch: bool = False
    settings_overrides: dict[str, Any] = Field(default_factory=dict)


class WorkflowRebuildResult(WorkflowModel):
    thread_id: str
    mode: WorkflowMode
    stages_requested: list[str] = Field(default_factory=list)
    stages_rebuilt: list[str] = Field(default_factory=list)
    stages_skipped: list[str] = Field(default_factory=list)
    stages_failed: list[str] = Field(default_factory=list)
    generated_artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
