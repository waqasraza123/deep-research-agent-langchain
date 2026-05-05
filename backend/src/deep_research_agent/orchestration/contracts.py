from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResearchTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"


class ResearchTaskType(str, Enum):
    QUESTION_NORMALIZATION = "question_normalization"
    SOURCE_TRIAGE = "source_triage"
    SOURCE_EXTRACTION = "source_extraction"
    EVIDENCE_COLLECTION = "evidence_collection"
    SUBQUESTION_ANSWERING = "subquestion_answering"
    CONTRADICTION_SCAN = "contradiction_scan"
    RISK_SCAN = "risk_scan"
    SYNTHESIS = "synthesis"
    CITATION_REVIEW = "citation_review"
    FINAL_REPORT_REVIEW = "final_report_review"


class SpecialistRole(str, Enum):
    PLANNER = "planner"
    SOURCE_TRIAGER = "source_triager"
    EVIDENCE_COLLECTOR = "evidence_collector"
    SKEPTICAL_REVIEWER = "skeptical_reviewer"
    DOMAIN_ANALYST = "domain_analyst"
    SYNTHESIS_WRITER = "synthesis_writer"
    CITATION_AUDITOR = "citation_auditor"
    RISK_REVIEWER = "risk_reviewer"


def model_to_json_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


class StageError(BaseModel):
    error_type: str
    message: str
    recoverable: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class StageInput(BaseModel):
    thread_id: str
    question: str
    urls: list[str] = Field(default_factory=list)
    task_type: ResearchTaskType
    specialist_role: SpecialistRole
    strategy: dict[str, Any] | None = None
    available_source_count: int = 0
    follow_links: bool = False
    max_links_per_source: int = 0
    previous_outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageOutput(BaseModel):
    node_id: str
    task_type: ResearchTaskType
    specialist_role: SpecialistRole
    status: ResearchTaskStatus = ResearchTaskStatus.SUCCEEDED
    stage_summary: str
    findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_next_steps: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    artifact_updates: dict[str, Any] = Field(default_factory=dict)
    model_instruction_block: str = ""
    skipped_reason: str | None = None
    error: StageError | None = None


class ExecutionDecision(BaseModel):
    node_id: str
    task_type: ResearchTaskType
    specialist_role: SpecialistRole
    should_run: bool
    reason: str
    confidence_policy: str = "standard"
    signals: list[str] = Field(default_factory=list)


class ResearchTaskNode(BaseModel):
    id: str
    task_type: ResearchTaskType
    specialist_role: SpecialistRole
    title: str
    description: str
    status: ResearchTaskStatus = ResearchTaskStatus.PENDING
    depends_on: list[str] = Field(default_factory=list)
    decision: ExecutionDecision | None = None
    output: StageOutput | None = None
    error: StageError | None = None
    skipped_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchTaskEdge(BaseModel):
    from_node: str
    to_node: str
    reason: str = ""
    required: bool = True


class ResearchTaskGraph(BaseModel):
    graph_id: str
    thread_id: str
    question: str
    urls: list[str] = Field(default_factory=list)
    nodes: list[ResearchTaskNode] = Field(default_factory=list)
    edges: list[ResearchTaskEdge] = Field(default_factory=list)
    created_at: str
    routing_signals: list[str] = Field(default_factory=list)
    confidence_policy: str = "standard"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def node_by_id(self) -> dict[str, ResearchTaskNode]:
        return {node.id: node for node in self.nodes}

    def execution_order(self) -> list[ResearchTaskNode]:
        by_id = self.node_by_id()
        remaining = set(by_id)
        ordered: list[ResearchTaskNode] = []
        while remaining:
            ready = sorted(
                node_id
                for node_id in remaining
                if all(dep not in remaining for dep in by_id[node_id].depends_on)
            )
            if not ready:
                raise ValueError("Task graph contains a dependency cycle")
            for node_id in ready:
                ordered.append(by_id[node_id])
                remaining.remove(node_id)
        return ordered

    def to_json_dict(self) -> dict[str, Any]:
        return model_to_json_dict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent, ensure_ascii=False) + "\n"


class OrchestrationSummary(BaseModel):
    thread_id: str
    graph_id: str
    status: ResearchTaskStatus
    executed_nodes: list[str] = Field(default_factory=list)
    skipped_nodes: dict[str, str] = Field(default_factory=dict)
    failed_nodes: dict[str, StageError] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    required_next_steps: list[str] = Field(default_factory=list)
    agent_instruction_block: str = ""
    confidence_policy: str = "standard"
    artifact_paths: list[str] = Field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return model_to_json_dict(self)
