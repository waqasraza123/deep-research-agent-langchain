from __future__ import annotations

from .contracts import (
    KernelWarning,
    PassType,
    ResearchBlueprint,
    ResearchComplexity,
    ResearchIntent,
    ResearchKernelInput,
    ResearchKernelSettings,
    stable_id,
    write_json,
)

EXPECTED_ARTIFACTS = [
    "kernel_blueprint.json",
    "kernel_blueprint.md",
    "kernel_passes.json",
    "kernel_passes.md",
    "source_inventory.json",
    "source_inventory.md",
    "source_units.json",
    "source_units.md",
    "evidence_units.json",
    "evidence_units.md",
    "evidence_coverage.json",
    "evidence_coverage.md",
    "claims.json",
    "claims.md",
    "critique_findings.json",
    "critique_findings.md",
    "operator_warnings.json",
    "operator_warnings.md",
    "verification_tasks.json",
    "verification_tasks.md",
    "verification_results.json",
    "verification_results.md",
    "confidence_calibration.json",
    "confidence_calibration.md",
    "kernel_artifact_registry.json",
    "kernel_artifact_registry.md",
    "kernel_summary.json",
    "kernel_summary.md",
    "research_readiness.md",
]


def _base_passes(
    settings: ResearchKernelSettings,
) -> tuple[list[PassType], list[PassType], dict[str, str]]:
    required: list[PassType] = [
        "request_analysis",
        "blueprint_generation",
        "source_inventory",
        "final_kernel_summary",
    ]
    optional: list[PassType] = []
    skipped: dict[str, str] = {
        "agent_execution": "The kernel wraps existing agent execution and does not rerun it during rebuild."
    }
    if settings.source_reasoning_enabled:
        required.extend(["source_unitization", "evidence_unitization"])
    else:
        skipped["source_unitization"] = "Source reasoning is disabled."
        skipped["evidence_unitization"] = "Source reasoning is disabled."
    if settings.critique_enabled:
        optional.append("report_critique")
    else:
        skipped["report_critique"] = "Critique is disabled."
    if settings.verification_enabled:
        optional.append("claim_verification")
    else:
        skipped["claim_verification"] = "Verification is disabled."
    if settings.confidence_enabled:
        optional.append("confidence_calibration")
    else:
        skipped["confidence_calibration"] = "Confidence calibration is disabled."
    return required, optional, skipped


def generate_blueprint(
    kernel_input: ResearchKernelInput,
    intent: ResearchIntent,
    complexity: ResearchComplexity,
    settings: ResearchKernelSettings | None = None,
    analyzer_warnings: list[KernelWarning] | None = None,
) -> ResearchBlueprint:
    settings = settings or ResearchKernelSettings()
    analyzer_warnings = analyzer_warnings or []
    required, optional, skipped = _base_passes(settings)
    label = intent.label
    source_requirements = [
        "Use supplied or already-fetched sources; preserve existing sources.json."
    ]
    evidence_requirements = ["Extract evidence with source linkage and citation readiness."]
    verification_requirements = ["Verify high-risk claims against local evidence only."]
    citation_policy = {
        "strict": bool(settings.strict_citation_mode),
        "require_source_ids": True,
        "unknown_source_claims": "warn",
    }
    freshness_policy = {
        "freshness_sensitive": False,
        "require_dates_for_latest_claims": False,
        "warn_when_dates_unknown": True,
    }
    safety_policy = {
        "sensitive_domain_review_required": settings.sensitive_domain_review_required,
        "fail_on_critical_warnings": settings.fail_on_critical_warnings,
        "prompt_injection_source_review": True,
    }
    synthesis_policy = {
        "must_state_uncertainty": True,
        "must_preserve_guaranteed_artifacts": ["plan.md", "notes.md", "sources.json", "report.md"],
        "avoid_professional_advice_tone": False,
    }
    warnings = list(analyzer_warnings)

    if label in {"comparative_analysis", "technical_due_diligence", "vendor_evaluation"}:
        for pass_type in ("report_critique", "claim_verification", "confidence_calibration"):
            if pass_type in optional:
                optional.remove(pass_type)  # type: ignore[arg-type]
                required.append(pass_type)  # type: ignore[arg-type]
        source_requirements.extend(
            [
                "Official docs for each compared framework or vendor.",
                "GitHub repositories, releases, or changelogs when available.",
                "Production, deployment, persistence, tool-calling, and failure-mode evidence.",
            ]
        )
        evidence_requirements.extend(
            [
                "orchestration model",
                "persistence/checkpointing",
                "streaming",
                "tool calling",
                "deployment",
                "observability",
                "ecosystem maturity",
                "failure modes",
            ]
        )
        verification_requirements.extend(
            [
                "Verify comparative claims.",
                "Verify recommendation claims.",
                "Verify current docs and version/date-sensitive claims.",
            ]
        )
        synthesis_policy["require_tradeoffs"] = True
    if label in {"implementation_planning", "source_code_research", "library_or_framework_review"}:
        source_requirements.extend(
            ["Implementation docs, API references, and source-code examples when available."]
        )
        evidence_requirements.extend(
            ["API behavior", "integration constraints", "operational risks"]
        )
    if label in {"legal_policy_review", "medical_health_review", "financial_risk_review"}:
        citation_policy["strict"] = True
        citation_policy["unknown_source_claims"] = "degrade_confidence"
        safety_policy["human_review_required"] = True
        synthesis_policy["avoid_professional_advice_tone"] = True
        verification_requirements.extend(
            [
                "Verify every material sensitive-domain claim.",
                "Flag missing primary or reputable sources.",
            ]
        )
        source_requirements.extend(
            ["Primary legal/regulatory, clinical, or financial sources as appropriate."]
        )
        warnings.append(
            KernelWarning(
                warning_id=stable_id("warn", "blueprint", "human_review", kernel_input.thread_id),
                subsystem="blueprint",
                code="human_review_required",
                severity="high",
                message="Sensitive-domain request requires conservative output and human review.",
                recommended_action="Do not treat deterministic kernel output as professional advice.",
            )
        )
    if label == "news_or_current_review" or "current" in intent.signals:
        freshness_policy["freshness_sensitive"] = True
        freshness_policy["require_dates_for_latest_claims"] = True
        verification_requirements.append(
            "Verify temporal and latest/current claims against dated evidence."
        )
        warnings.append(
            KernelWarning(
                warning_id=stable_id("warn", "blueprint", "freshness", kernel_input.thread_id),
                subsystem="blueprint",
                code="freshness_required",
                severity="medium",
                message="The request is currentness-sensitive; undated sources reduce confidence.",
                recommended_action="Prefer dated primary sources and release notes.",
            )
        )
    if not kernel_input.urls:
        skipped["source_discovery"] = (
            "No source discovery pass exists in this kernel rebuild; supplied or existing sources are required."
        )

    seen: set[str] = set()
    ordered_required: list[PassType] = []
    for pass_type in required:
        if pass_type not in seen:
            ordered_required.append(pass_type)
            seen.add(pass_type)
    optional = [p for p in optional if p not in seen]
    return ResearchBlueprint(
        blueprint_id=stable_id("blueprint", kernel_input.thread_id, intent.label, complexity.level),
        thread_id=kernel_input.thread_id,
        question=kernel_input.question,
        normalized_question=" ".join(kernel_input.question.lower().split()),
        intent=intent,
        complexity=complexity,
        required_passes=ordered_required,
        optional_passes=optional,
        skipped_passes=skipped,
        source_requirements=source_requirements,
        evidence_requirements=evidence_requirements,
        verification_requirements=verification_requirements,
        citation_policy=citation_policy,
        freshness_policy=freshness_policy,
        safety_policy=safety_policy,
        synthesis_policy=synthesis_policy,
        expected_artifacts=list(EXPECTED_ARTIFACTS),
        operator_warnings=warnings,
    )


def render_blueprint_markdown(blueprint: ResearchBlueprint) -> str:
    lines = [
        "# Research Intelligence Blueprint",
        "",
        f"Thread: `{blueprint.thread_id}`",
        f"Intent: `{blueprint.intent.label}` ({blueprint.intent.confidence_score:.2f})",
        f"Complexity: `{blueprint.complexity.level}` ({blueprint.complexity.score:.2f})",
        "",
        "## Required Passes",
        "",
        *[f"- `{p}`" for p in blueprint.required_passes],
        "",
        "## Optional Passes",
        "",
        *([f"- `{p}`" for p in blueprint.optional_passes] or ["- None"]),
        "",
        "## Source Requirements",
        "",
        *[f"- {item}" for item in blueprint.source_requirements],
        "",
        "## Evidence Requirements",
        "",
        *[f"- {item}" for item in blueprint.evidence_requirements],
        "",
        "## Verification Requirements",
        "",
        *[f"- {item}" for item in blueprint.verification_requirements],
        "",
        "## Operator Warnings",
        "",
    ]
    if blueprint.operator_warnings:
        lines.extend(
            f"- **{w.severity}** `{w.code}`: {w.message}" for w in blueprint.operator_warnings
        )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def write_blueprint_artifacts(run_dir, blueprint: ResearchBlueprint) -> list[str]:
    write_json(run_dir / "kernel_blueprint.json", blueprint)
    (run_dir / "kernel_blueprint.md").write_text(
        render_blueprint_markdown(blueprint), encoding="utf-8"
    )
    return ["kernel_blueprint.json", "kernel_blueprint.md"]
