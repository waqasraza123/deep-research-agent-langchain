from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DependencyKind = Literal["input", "source", "model", "settings", "artifact", "subsystem"]
ReproducibilityStatus = Literal["replayable", "partially_replayable", "not_replayable"]


class ArtifactDependency(BaseModel):
    dependency_type: DependencyKind
    identifier: str
    path: str | None = None
    content_hash: str | None = None
    relationship: str = ""
    required_for_replay: bool = True


class SourceFingerprint(BaseModel):
    source_id: str
    url: str
    normalized_url: str = ""
    title: str | None = None
    fetched_at: str | None = None
    local_path: str | None = None
    content_hash: str | None = None
    metadata_hash: str
    live_dependency: bool = True
    may_have_changed: bool = True


class ModelInvocationFingerprint(BaseModel):
    provider: str
    model_name: str
    purpose: str = ""
    config_hash: str
    redacted_config: dict[str, Any] = Field(default_factory=dict)
    nondeterministic: bool = True
    credentials_required: list[str] = Field(default_factory=list)


class SubsystemInvocation(BaseModel):
    subsystem_name: str
    subsystem_version: str = "1.0"
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    config_hash: str | None = None
    warnings: list[str] = Field(default_factory=list)


class RunInputFingerprint(BaseModel):
    question_hash: str
    urls_hash: str
    combined_hash: str
    question_length: int = 0
    urls: list[str] = Field(default_factory=list)
    normalized_urls: list[str] = Field(default_factory=list)


class ArtifactMetadata(BaseModel):
    artifact_name: str
    artifact_path: str
    artifact_type: str
    created_at: str
    updated_at: str
    content_hash: str
    size_bytes: int
    producer_subsystem: str
    producer_version: str = "1.0"
    input_dependencies: list[ArtifactDependency] = Field(default_factory=list)
    source_dependencies: list[ArtifactDependency] = Field(default_factory=list)
    model_dependencies: list[ArtifactDependency] = Field(default_factory=list)
    artifact_dependencies: list[ArtifactDependency] = Field(default_factory=list)
    settings_fingerprint: str
    warnings: list[str] = Field(default_factory=list)


class ArtifactManifest(BaseModel):
    manifest_version: str = "1.0"
    thread_id: str
    generated_at: str
    run_input: RunInputFingerprint | None = None
    settings_fingerprint: str
    artifacts: list[ArtifactMetadata] = Field(default_factory=list)
    sources: list[SourceFingerprint] = Field(default_factory=list)
    model_invocations: list[ModelInvocationFingerprint] = Field(default_factory=list)
    subsystem_invocations: list[SubsystemInvocation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ArtifactDependencyGraph(BaseModel):
    graph_version: str = "1.0"
    thread_id: str
    generated_at: str
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, str]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ProvenanceRecord(BaseModel):
    thread_id: str
    generated_at: str
    manifest: ArtifactManifest
    dependency_graph: ArtifactDependencyGraph


class ReproducibilityReport(BaseModel):
    report_version: str = "1.0"
    thread_id: str
    generated_at: str
    status: ReproducibilityStatus
    can_replay: bool
    replayable_offline_artifacts: list[str] = Field(default_factory=list)
    live_source_dependent_artifacts: list[str] = Field(default_factory=list)
    model_nondeterministic_artifacts: list[str] = Field(default_factory=list)
    sources_may_have_changed: list[SourceFingerprint] = Field(default_factory=list)
    credentials_required: list[str] = Field(default_factory=list)
    providers_required: list[str] = Field(default_factory=list)
    settings_used: dict[str, Any] = Field(default_factory=dict)
    not_reproducible: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReplayPlan(BaseModel):
    plan_version: str = "1.0"
    thread_id: str
    generated_at: str
    ordered_steps: list[dict[str, Any]] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    optional_artifacts: list[str] = Field(default_factory=list)
    expected_output_hashes: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ArtifactDiffSummary(BaseModel):
    added_artifacts: list[str] = Field(default_factory=list)
    removed_artifacts: list[str] = Field(default_factory=list)
    changed_artifacts: list[str] = Field(default_factory=list)
    unchanged_artifacts: list[str] = Field(default_factory=list)
    changed_hashes: dict[str, dict[str, str | None]] = Field(default_factory=dict)
    changed_sizes: dict[str, dict[str, int | None]] = Field(default_factory=dict)
    changed_producer_subsystems: dict[str, dict[str, str | None]] = Field(default_factory=dict)
