from __future__ import annotations

from .contracts import (
    ArtifactType,
    WorkflowArtifactContract,
    WorkflowFailureBehavior,
    WorkflowInputRequirement,
    WorkflowInputType,
    WorkflowMode,
    WorkflowPolicy,
    WorkflowStageDefinition,
    WorkflowStageType,
    WorkflowTemplate,
)


def _req(name: str, input_type: WorkflowInputType, description: str) -> WorkflowInputRequirement:
    return WorkflowInputRequirement(name=name, input_type=input_type, description=description)


def _stage(
    stage_type: WorkflowStageType,
    *,
    required: bool = True,
    depends_on: list[str] | None = None,
    optional_depends_on: list[str] | None = None,
    outputs: list[str] | None = None,
    services: list[str] | None = None,
    failure: WorkflowFailureBehavior | None = None,
    skip_if: list[str] | None = None,
    metadata: dict | None = None,
) -> WorkflowStageDefinition:
    stage_id = stage_type.value
    return WorkflowStageDefinition(
        stage_id=stage_id,
        name=stage_id.replace("_", " ").title(),
        stage_type=stage_type,
        required=required,
        depends_on=depends_on or [],
        optional_depends_on=optional_depends_on or [],
        output_artifacts=outputs or [],
        consumes_runtime_services=services or [],
        failure_behavior=failure
        or (
            WorkflowFailureBehavior.fail_workflow
            if required
            else WorkflowFailureBehavior.mark_degraded
        ),
        skip_if_artifacts_exist=skip_if or [],
        metadata=metadata or {},
    )


def _contract(
    name: str,
    artifact_type: ArtifactType,
    *,
    required: bool,
    producer: str,
    min_size: int = 1,
    must_json: bool = False,
    description: str = "",
) -> WorkflowArtifactContract:
    return WorkflowArtifactContract(
        artifact_name=name,
        artifact_type=artifact_type,
        required=required,
        producer_stage=producer,
        min_size_bytes=min_size,
        must_parse_as_json=must_json,
        must_be_nonempty=artifact_type in {ArtifactType.markdown, ArtifactType.text},
        description=description,
    )


BASE_CONTRACTS = [
    _contract("plan.md", ArtifactType.markdown, required=True, producer="agent_execution"),
    _contract("notes.md", ArtifactType.markdown, required=True, producer="agent_execution"),
    _contract(
        "sources.json",
        ArtifactType.json,
        required=True,
        producer="source_fetching",
        must_json=True,
    ),
    _contract("report.md", ArtifactType.markdown, required=True, producer="agent_execution"),
    _contract(
        "workflow_manifest.json",
        ArtifactType.json,
        required=False,
        producer="finalization",
        must_json=True,
    ),
    _contract(
        "workflow_readiness.json",
        ArtifactType.json,
        required=False,
        producer="finalization",
        must_json=True,
    ),
]


OPTIONAL_CONTRACTS = {
    WorkflowStageType.source_safety: _contract(
        "source_safety.json",
        ArtifactType.json,
        required=False,
        producer="source_safety",
        must_json=True,
    ),
    WorkflowStageType.document_intelligence: _contract(
        "document_profiles.json",
        ArtifactType.json,
        required=False,
        producer="document_intelligence",
        must_json=True,
    ),
    WorkflowStageType.retrieval_indexing: _contract(
        "retrieval_index.json",
        ArtifactType.json,
        required=False,
        producer="retrieval_indexing",
        must_json=True,
    ),
    WorkflowStageType.evidence_extraction: _contract(
        "evidence_ledger.json",
        ArtifactType.json,
        required=False,
        producer="evidence_extraction",
        must_json=True,
    ),
    WorkflowStageType.verification: _contract(
        "verification_results.json",
        ArtifactType.json,
        required=False,
        producer="verification",
        must_json=True,
    ),
    WorkflowStageType.evaluation: _contract(
        "evaluation.json",
        ArtifactType.json,
        required=False,
        producer="evaluation",
        must_json=True,
    ),
    WorkflowStageType.quality_gate: _contract(
        "workflow_quality_gate_result.json",
        ArtifactType.json,
        required=False,
        producer="quality_gate",
        must_json=True,
    ),
}


def _contracts_for(stages: list[WorkflowStageDefinition]) -> list[WorkflowArtifactContract]:
    contracts = list(BASE_CONTRACTS)
    present = {stage.stage_type for stage in stages}
    for stage_type, contract in OPTIONAL_CONTRACTS.items():
        if stage_type in present:
            contracts.append(contract)
    return contracts


def _policy(
    mode: WorkflowMode,
    *,
    strict: bool = False,
    freshness: bool = False,
    network: bool = True,
    model: bool = True,
    mock: bool = False,
    review: bool = False,
    sensitive: bool = False,
    gate: bool = False,
    gate_id: str | None = None,
    denied: list[WorkflowStageType] | None = None,
    metadata: dict | None = None,
) -> WorkflowPolicy:
    return WorkflowPolicy(
        policy_id=f"{mode.value}_policy",
        name=f"{mode.value.replace('_', ' ').title()} Policy",
        strict_citations=strict,
        freshness_required=freshness,
        external_network_allowed=network,
        model_required=model,
        mock_allowed=mock,
        review_required=review,
        sensitive_domain=sensitive,
        run_quality_gate=gate,
        quality_gate_id=gate_id,
        denied_stage_types=denied or [],
        metadata=metadata or {},
    )


def _template(
    mode: WorkflowMode,
    description: str,
    stages: list[WorkflowStageDefinition],
    *,
    tags: list[str],
    required_inputs: list[WorkflowInputRequirement] | None = None,
    optional_inputs: list[WorkflowInputRequirement] | None = None,
    policies: list[WorkflowPolicy] | None = None,
    default_settings: dict | None = None,
    success: list[str] | None = None,
    failure: WorkflowFailureBehavior = WorkflowFailureBehavior.fail_workflow,
) -> WorkflowTemplate:
    return WorkflowTemplate(
        template_id=mode.value,
        name=mode.value.replace("_", " ").title(),
        description=description,
        mode=mode,
        tags=tags,
        default_settings=default_settings or {},
        required_inputs=required_inputs
        or [_req("question", WorkflowInputType.question, "Research question")],
        optional_inputs=optional_inputs
        or [_req("urls", WorkflowInputType.urls, "User supplied source URLs")],
        stages=stages,
        artifact_contracts=_contracts_for(stages),
        policies=policies or [_policy(mode)],
        success_criteria=success
        or [
            "Required artifacts exist and validate.",
            "Final readiness is not blocked or failed.",
        ],
        failure_policy=failure,
    )


def _quality_gate_dep(base_dep: str = "evaluation") -> WorkflowStageDefinition:
    return _stage(
        WorkflowStageType.quality_gate,
        required=False,
        depends_on=[base_dep],
        services=["evaluation_lab_quality_gates"],
        failure=WorkflowFailureBehavior.mark_degraded,
    )


def built_in_templates() -> list[WorkflowTemplate]:
    quick = _template(
        WorkflowMode.quick_brief,
        "Answer a focused question from supplied URLs with minimal post-processing.",
        [
            _stage(WorkflowStageType.input_snapshot, outputs=["workflow_input_snapshot.json"]),
            _stage(WorkflowStageType.source_safety, required=False, depends_on=["input_snapshot"]),
            _stage(WorkflowStageType.source_fetching, depends_on=["input_snapshot"]),
            _stage(
                WorkflowStageType.agent_execution,
                depends_on=["source_fetching"],
                optional_depends_on=["source_safety"],
                outputs=["plan.md", "notes.md", "sources.json", "report.md"],
                services=["agent"],
            ),
            _stage(WorkflowStageType.artifact_backfill, depends_on=["agent_execution"]),
            _stage(
                WorkflowStageType.verification,
                required=False,
                depends_on=["artifact_backfill"],
                failure=WorkflowFailureBehavior.warn_and_continue,
            ),
            _quality_gate_dep("artifact_backfill"),
            _stage(WorkflowStageType.finalization, depends_on=["artifact_backfill"]),
        ],
        tags=["brief", "default"],
        policies=[_policy(WorkflowMode.quick_brief, mock=True)],
        default_settings={"complexity": "low"},
    )

    deep = _template(
        WorkflowMode.deep_research,
        (
            "Thorough research report with source analysis, evidence, synthesis, "
            "verification, and evaluation."
        ),
        [
            _stage(WorkflowStageType.input_snapshot),
            _stage(WorkflowStageType.request_analysis, depends_on=["input_snapshot"]),
            _stage(WorkflowStageType.source_fetching, depends_on=["request_analysis"]),
            _stage(WorkflowStageType.source_safety, required=False, depends_on=["source_fetching"]),
            _stage(
                WorkflowStageType.document_intelligence,
                required=False,
                depends_on=["source_fetching"],
            ),
            _stage(
                WorkflowStageType.retrieval_indexing,
                required=False,
                depends_on=["document_intelligence"],
            ),
            _stage(WorkflowStageType.agent_execution, depends_on=["source_fetching"]),
            _stage(WorkflowStageType.artifact_backfill, depends_on=["agent_execution"]),
            _stage(
                WorkflowStageType.evidence_extraction,
                required=False,
                depends_on=["artifact_backfill"],
            ),
            _stage(WorkflowStageType.synthesis, required=False, depends_on=["evidence_extraction"]),
            _stage(
                WorkflowStageType.verification, required=False, depends_on=["artifact_backfill"]
            ),
            _stage(WorkflowStageType.evaluation, required=False, depends_on=["artifact_backfill"]),
            _stage(WorkflowStageType.provenance, required=False, depends_on=["artifact_backfill"]),
            _quality_gate_dep("evaluation"),
            _stage(WorkflowStageType.finalization, depends_on=["artifact_backfill"]),
        ],
        tags=["research", "broad"],
        policies=[_policy(WorkflowMode.deep_research, mock=True)],
        default_settings={"complexity": "high"},
    )

    tdd = _template(
        WorkflowMode.technical_due_diligence,
        "Evaluate a technical stack, framework, API, library, or architecture.",
        [
            _stage(WorkflowStageType.request_analysis),
            _stage(WorkflowStageType.source_fetching, depends_on=["request_analysis"]),
            _stage(WorkflowStageType.source_audit, required=False, depends_on=["source_fetching"]),
            _stage(
                WorkflowStageType.document_intelligence,
                required=False,
                depends_on=["source_fetching"],
            ),
            _stage(
                WorkflowStageType.retrieval_indexing,
                required=False,
                depends_on=["document_intelligence"],
            ),
            _stage(
                WorkflowStageType.agent_control_planning,
                required=False,
                depends_on=["request_analysis"],
            ),
            _stage(WorkflowStageType.agent_execution, depends_on=["source_fetching"]),
            _stage(WorkflowStageType.artifact_backfill, depends_on=["agent_execution"]),
            _stage(
                WorkflowStageType.evidence_extraction,
                required=False,
                depends_on=["artifact_backfill"],
            ),
            _stage(WorkflowStageType.synthesis, required=False, depends_on=["evidence_extraction"]),
            _stage(
                WorkflowStageType.verification, required=False, depends_on=["artifact_backfill"]
            ),
            _stage(WorkflowStageType.evaluation, required=False, depends_on=["artifact_backfill"]),
            _stage(WorkflowStageType.finalization, depends_on=["artifact_backfill"]),
        ],
        tags=["technical", "due-diligence"],
        policies=[
            _policy(
                WorkflowMode.technical_due_diligence,
                mock=True,
                metadata={
                    "require_tradeoffs": True,
                    "require_implementation_risks": True,
                    "require_failure_modes": True,
                    "require_uncertainty": True,
                },
            )
        ],
    )

    comparison = _template(
        WorkflowMode.framework_comparison,
        "Compare two or more frameworks, tools, or vendors across explicit dimensions.",
        [
            _stage(WorkflowStageType.request_analysis),
            _stage(WorkflowStageType.source_fetching, depends_on=["request_analysis"]),
            _stage(WorkflowStageType.agent_execution, depends_on=["source_fetching"]),
            _stage(WorkflowStageType.artifact_backfill, depends_on=["agent_execution"]),
            _stage(WorkflowStageType.synthesis, required=False, depends_on=["artifact_backfill"]),
            _stage(
                WorkflowStageType.verification, required=False, depends_on=["artifact_backfill"]
            ),
            _stage(WorkflowStageType.evaluation, required=False, depends_on=["artifact_backfill"]),
            _quality_gate_dep("evaluation"),
            _stage(WorkflowStageType.finalization, depends_on=["artifact_backfill"]),
        ],
        tags=["comparison"],
        policies=[
            _policy(
                WorkflowMode.framework_comparison,
                mock=True,
                metadata={
                    "require_both_sides": True,
                    "require_comparison_dimensions": True,
                    "require_counterarguments": True,
                    "forbid_absolute_best_without_evidence": True,
                },
            )
        ],
    )

    vendor = _template(
        WorkflowMode.vendor_evaluation,
        "Evaluate vendor claims and adoption risk.",
        [
            _stage(WorkflowStageType.source_fetching),
            _stage(WorkflowStageType.source_safety, required=False, depends_on=["source_fetching"]),
            _stage(WorkflowStageType.source_audit, required=False, depends_on=["source_fetching"]),
            _stage(WorkflowStageType.agent_execution, depends_on=["source_fetching"]),
            _stage(WorkflowStageType.artifact_backfill, depends_on=["agent_execution"]),
            _stage(
                WorkflowStageType.verification, required=False, depends_on=["artifact_backfill"]
            ),
            _stage(WorkflowStageType.evaluation, required=False, depends_on=["artifact_backfill"]),
            _stage(WorkflowStageType.finalization, depends_on=["artifact_backfill"]),
        ],
        tags=["vendor", "risk"],
        policies=[
            _policy(
                WorkflowMode.vendor_evaluation,
                mock=True,
                metadata={"warn_marketing_only": True, "require_risk_register": True},
            )
        ],
    )

    legal = _template(
        WorkflowMode.legal_policy_review,
        "Conservative review of law, policy, or regulation sources.",
        [
            _stage(WorkflowStageType.source_fetching),
            _stage(WorkflowStageType.source_safety, required=False, depends_on=["source_fetching"]),
            _stage(
                WorkflowStageType.temporal_analysis, required=False, depends_on=["source_fetching"]
            ),
            _stage(WorkflowStageType.agent_execution, depends_on=["source_fetching"]),
            _stage(WorkflowStageType.artifact_backfill, depends_on=["agent_execution"]),
            _stage(
                WorkflowStageType.verification, required=False, depends_on=["artifact_backfill"]
            ),
            _stage(WorkflowStageType.evaluation, required=False, depends_on=["artifact_backfill"]),
            _stage(WorkflowStageType.finalization, depends_on=["artifact_backfill"]),
        ],
        tags=["legal", "policy", "sensitive"],
        policies=[
            _policy(
                WorkflowMode.legal_policy_review,
                strict=True,
                freshness=True,
                mock=True,
                review=True,
                sensitive=True,
                metadata={
                    "no_professional_legal_advice": True,
                    "warn_if_no_primary_legal_source": True,
                },
            )
        ],
    )

    implementation = _template(
        WorkflowMode.implementation_planning,
        "Turn research into an implementation plan with risks and tests.",
        [
            _stage(WorkflowStageType.request_analysis),
            _stage(
                WorkflowStageType.source_fetching,
                required=False,
                depends_on=["request_analysis"],
                failure=WorkflowFailureBehavior.warn_and_continue,
            ),
            _stage(WorkflowStageType.agent_execution, depends_on=["request_analysis"]),
            _stage(WorkflowStageType.artifact_backfill, depends_on=["agent_execution"]),
            _stage(WorkflowStageType.synthesis, required=False, depends_on=["artifact_backfill"]),
            _stage(
                WorkflowStageType.verification,
                required=False,
                depends_on=["artifact_backfill"],
                failure=WorkflowFailureBehavior.warn_and_continue,
            ),
            _stage(WorkflowStageType.finalization, depends_on=["artifact_backfill"]),
        ],
        tags=["implementation"],
        policies=[
            _policy(
                WorkflowMode.implementation_planning,
                mock=True,
                metadata={
                    "require_architecture_steps": True,
                    "require_edge_cases": True,
                    "require_risks": True,
                    "require_test_plan": True,
                },
            )
        ],
    )

    source_audit = _template(
        WorkflowMode.source_audit_only,
        "Fetch and analyze source quality without making final claims beyond audit summary.",
        [
            _stage(WorkflowStageType.source_fetching),
            _stage(WorkflowStageType.source_safety, required=False, depends_on=["source_fetching"]),
            _stage(WorkflowStageType.source_audit, required=False, depends_on=["source_fetching"]),
            _stage(
                WorkflowStageType.artifact_backfill, required=False, depends_on=["source_fetching"]
            ),
            _stage(WorkflowStageType.finalization, depends_on=["artifact_backfill"]),
        ],
        tags=["source-audit"],
        policies=[
            _policy(
                WorkflowMode.source_audit_only,
                model=False,
                mock=True,
                metadata={"report_may_be_audit_summary": True, "no_final_claims": True},
            )
        ],
    )

    evidence = _template(
        WorkflowMode.evidence_extraction_only,
        "Convert sources and artifacts into evidence units.",
        [
            _stage(WorkflowStageType.source_fetching, required=False),
            _stage(
                WorkflowStageType.document_intelligence,
                required=False,
                depends_on=["source_fetching"],
            ),
            _stage(
                WorkflowStageType.evidence_extraction,
                required=False,
                depends_on=["source_fetching"],
            ),
            _stage(
                WorkflowStageType.artifact_backfill,
                required=False,
                depends_on=["evidence_extraction"],
            ),
            _stage(WorkflowStageType.finalization, depends_on=["artifact_backfill"]),
        ],
        tags=["evidence"],
        policies=[_policy(WorkflowMode.evidence_extraction_only, model=False, mock=True)],
    )

    verification = _template(
        WorkflowMode.verification_only,
        (
            "Rebuild verification from existing run artifacts without refetching "
            "or rerunning the agent."
        ),
        [
            _stage(
                WorkflowStageType.artifact_backfill,
                skip_if=["plan.md", "notes.md", "sources.json", "report.md"],
            ),
            _stage(WorkflowStageType.verification, depends_on=["artifact_backfill"]),
            _stage(WorkflowStageType.finalization, depends_on=["verification"]),
        ],
        tags=["verification", "rebuild"],
        required_inputs=[
            _req("existing_run", WorkflowInputType.existing_run, "Existing run/thread id")
        ],
        policies=[
            _policy(
                WorkflowMode.verification_only,
                network=False,
                model=False,
                mock=True,
                denied=[WorkflowStageType.source_fetching, WorkflowStageType.agent_execution],
            )
        ],
    )

    evaluation = _template(
        WorkflowMode.evaluation_only,
        "Evaluate existing run artifacts without refetching or rerunning the agent.",
        [
            _stage(
                WorkflowStageType.artifact_backfill,
                skip_if=["plan.md", "notes.md", "sources.json", "report.md"],
            ),
            _stage(WorkflowStageType.evaluation, depends_on=["artifact_backfill"]),
            _stage(WorkflowStageType.finalization, depends_on=["evaluation"]),
        ],
        tags=["evaluation", "rebuild"],
        required_inputs=[
            _req("existing_run", WorkflowInputType.existing_run, "Existing run/thread id")
        ],
        policies=[
            _policy(
                WorkflowMode.evaluation_only,
                network=False,
                model=False,
                mock=True,
                denied=[WorkflowStageType.source_fetching, WorkflowStageType.agent_execution],
            )
        ],
    )

    adversarial = _template(
        WorkflowMode.adversarial_source_review,
        "Inspect malicious, prompt-injection, and source-poisoning risk.",
        [
            _stage(WorkflowStageType.source_fetching),
            _stage(WorkflowStageType.source_safety, required=False, depends_on=["source_fetching"]),
            _stage(
                WorkflowStageType.agent_execution, required=False, depends_on=["source_fetching"]
            ),
            _stage(
                WorkflowStageType.artifact_backfill, required=False, depends_on=["agent_execution"]
            ),
            _stage(
                WorkflowStageType.verification, required=False, depends_on=["artifact_backfill"]
            ),
            _stage(WorkflowStageType.evaluation, required=False, depends_on=["artifact_backfill"]),
            _quality_gate_dep("evaluation"),
            _stage(WorkflowStageType.finalization, depends_on=["artifact_backfill"]),
        ],
        tags=["adversarial", "source-safety"],
        policies=[
            _policy(
                WorkflowMode.adversarial_source_review,
                mock=True,
                gate=True,
                gate_id="adversarial",
                metadata={"source_quarantine_strict": True},
            )
        ],
    )

    offline = _template(
        WorkflowMode.offline_benchmark,
        "Run a benchmark or gate-compatible workflow in offline/mock mode.",
        [
            _stage(WorkflowStageType.input_snapshot),
            _stage(WorkflowStageType.agent_execution, depends_on=["input_snapshot"]),
            _stage(WorkflowStageType.artifact_backfill, depends_on=["agent_execution"]),
            _stage(WorkflowStageType.quality_gate, depends_on=["artifact_backfill"]),
            _stage(WorkflowStageType.finalization, depends_on=["quality_gate"]),
        ],
        tags=["offline", "benchmark"],
        policies=[
            _policy(
                WorkflowMode.offline_benchmark,
                network=False,
                model=False,
                mock=True,
                gate=True,
                gate_id="smoke",
                denied=[WorkflowStageType.source_discovery],
            )
        ],
        default_settings={"mock_mode": True},
    )

    rebuild = _template(
        WorkflowMode.rebuild_from_artifacts,
        "Rebuild downstream workflow artifacts from an existing run directory.",
        [
            _stage(
                WorkflowStageType.artifact_backfill,
                skip_if=["plan.md", "notes.md", "sources.json", "report.md"],
            ),
            _stage(
                WorkflowStageType.verification, required=False, depends_on=["artifact_backfill"]
            ),
            _stage(WorkflowStageType.evaluation, required=False, depends_on=["artifact_backfill"]),
            _stage(WorkflowStageType.provenance, required=False, depends_on=["artifact_backfill"]),
            _stage(WorkflowStageType.finalization, depends_on=["artifact_backfill"]),
        ],
        tags=["rebuild"],
        required_inputs=[
            _req("existing_run", WorkflowInputType.existing_run, "Existing run/thread id")
        ],
        policies=[
            _policy(
                WorkflowMode.rebuild_from_artifacts,
                network=False,
                model=False,
                mock=True,
                denied=[WorkflowStageType.source_fetching, WorkflowStageType.agent_execution],
            )
        ],
    )

    return [
        quick,
        deep,
        tdd,
        comparison,
        vendor,
        legal,
        implementation,
        source_audit,
        evidence,
        verification,
        evaluation,
        adversarial,
        offline,
        rebuild,
    ]
