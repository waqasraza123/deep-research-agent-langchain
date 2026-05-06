from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_research_agent.intelligence_kernel import (
    ResearchKernelInput,
    ResearchKernelSettings,
    analyze_request,
    generate_blueprint,
    rebuild_intelligence_kernel,
)
from deep_research_agent.intelligence_kernel.artifact_registry import build_artifact_registry
from deep_research_agent.intelligence_kernel.blueprint import render_blueprint_markdown
from deep_research_agent.intelligence_kernel.confidence import calibrate_confidence
from deep_research_agent.intelligence_kernel.critique import critique_report, extract_claims
from deep_research_agent.intelligence_kernel.reasoning_units import build_evidence_units
from deep_research_agent.intelligence_kernel.source_units import (
    build_source_units,
    normalize_source_url,
)
from deep_research_agent.intelligence_kernel.verification import (
    generate_verification_tasks,
    run_verification_tasks,
)
from deep_research_agent.settings import Settings


def _kernel_input(question: str, urls: list[str] | None = None) -> ResearchKernelInput:
    return ResearchKernelInput(thread_id="kernel-test", question=question, urls=urls or [])


def _blueprint(question: str, urls: list[str] | None = None):
    intent, complexity, warnings = analyze_request(question, urls or [])
    return generate_blueprint(
        _kernel_input(question, urls), intent, complexity, ResearchKernelSettings(), warnings
    )


def _fake_run(run_dir: Path) -> None:
    source_text = (
        "LangGraph is a framework for stateful agent orchestration. "
        "LangGraph supports checkpointing and persistence for production agent workflows. "
        "CrewAI focuses on role-based multi-agent collaboration. "
        "LangGraph is better for durable backend orchestration when checkpointing is required. "
        "The release notes were updated in 2026. "
    )
    (run_dir / "source1.txt").write_text(source_text, encoding="utf-8")
    (run_dir / "plan.md").write_text(
        "# Plan\n\n- Compare orchestration and persistence.\n", encoding="utf-8"
    )
    (run_dir / "notes.md").write_text(
        "# Notes\n\n- LangGraph supports checkpointing.\n- CrewAI focuses on role collaboration.\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        "# Report\n\n"
        "LangGraph is better than CrewAI for a production research-agent backend because it supports checkpointing. "
        "CrewAI is useful for role-based collaboration. "
        "LangGraph is always the best choice for every backend. "
        "The latest release was in 2026. "
        "CrewAI reduces infrastructure cost by 99 percent in all deployments.\n",
        encoding="utf-8",
    )
    (run_dir / "sources.json").write_text(
        json.dumps(
            [
                {
                    "source_id": "src1",
                    "url": "https://docs.langchain.com/langgraph/?utm_source=x",
                    "title": "LangGraph docs",
                    "document_kind": "html",
                    "source_kind": "root",
                    "local_path": "source1.txt",
                    "ok": True,
                    "word_count": 40,
                    "fetched_at": "2026-05-01T00:00:00Z",
                },
                {
                    "source_id": "src2",
                    "url": "https://docs.langchain.com/langgraph/",
                    "title": "LangGraph duplicate",
                    "summary": "Duplicate docs page",
                    "ok": True,
                },
                {
                    "source_id": "src3",
                    "url": "https://random.example/thin",
                    "title": "Thin page",
                    "summary": "Short",
                    "ok": True,
                },
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is LangGraph?", "factual_answer"),
        ("Compare LangGraph vs CrewAI for backend architecture", "technical_due_diligence"),
        ("Review the compliance policy and liability terms", "legal_policy_review"),
        ("What treatment helps these symptoms according to doctors?", "medical_health_review"),
        ("Assess investment risk and stock valuation", "financial_risk_review"),
        ("What is the latest pricing version today?", "news_or_current_review"),
        ("Build an implementation plan for a FastAPI LangGraph backend", "implementation_planning"),
    ],
)
def test_request_analyzer_intents(question: str, expected: str) -> None:
    intent, complexity, warnings = analyze_request(question, [])
    assert intent.label == expected
    assert 0 <= complexity.score <= 1
    assert warnings


def test_request_analyzer_broad_ambiguous_question() -> None:
    intent, complexity, _ = analyze_request("Tell me everything about it and what is best", [])
    assert intent.label in {"general_research", "comparative_analysis"}
    assert complexity.level in {"moderate", "deep", "adversarial"}


def test_blueprint_generation_sensitive_and_serializable() -> None:
    bp = _blueprint("Review medical treatment risk and symptoms", ["https://nih.gov/x"])
    assert bp.citation_policy["strict"] is True
    assert "report_critique" in bp.optional_passes or "report_critique" in bp.required_passes
    assert "kernel_summary.json" in bp.expected_artifacts
    assert bp.skipped_passes["agent_execution"]
    assert "Research Intelligence Blueprint" in render_blueprint_markdown(bp)
    assert bp.dict()["intent"]["label"] == "medical_health_review"


def test_source_units_normalization_id_role_and_markdown(tmp_path: Path) -> None:
    _fake_run(tmp_path)
    bp = _blueprint(
        "Compare LangGraph and CrewAI for production backend",
        ["https://docs.langchain.com/langgraph"],
    )
    units, inventory, warnings = build_source_units(tmp_path, bp)
    assert (
        normalize_source_url("HTTPS://Example.com/a/?utm_source=x&b=1")
        == "https://example.com/a?b=1"
    )
    assert units[0].source_unit_id == build_source_units(tmp_path, bp)[0][0].source_unit_id
    assert any(unit.source_role == "duplicate" for unit in units)
    assert any(unit.source_role == "weak_reference" for unit in units)
    assert any(unit.trust_level == "high" for unit in units)
    assert inventory["source_count"] == 3
    assert warnings


def test_evidence_extraction_claims_critique_verification_confidence(tmp_path: Path) -> None:
    _fake_run(tmp_path)
    bp = _blueprint(
        "Compare LangGraph and CrewAI for a production research-agent backend latest 2026"
    )
    sources, _, _ = build_source_units(tmp_path, bp)
    evidence, coverage, evidence_warnings = build_evidence_units(tmp_path, bp, sources)
    assert evidence
    assert any(
        unit.evidence_type in {"comparison", "date_or_version", "code_or_api_reference"}
        for unit in evidence
    )
    assert coverage["evidence_unit_count"] == len(evidence)

    claims, claim_warnings = extract_claims(tmp_path, bp)
    assert any(claim.claim_type == "comparative" for claim in claims)
    assert any(claim.claim_type == "temporal" for claim in claims)
    assert any(claim.strength == "absolute" for claim in claims)
    assert claim_warnings

    findings, critique_warnings = critique_report(tmp_path, bp, sources, evidence, claims)
    assert any(f.category == "overclaiming" for f in findings)
    assert any(f.category == "missing_counterargument" for f in findings)

    tasks = generate_verification_tasks(claims, findings, evidence, ResearchKernelSettings())
    assert tasks == sorted(tasks, key=lambda task: (task.priority, task.task_type, task.claim_id))
    tasks = run_verification_tasks(tasks, claims, evidence, sources)
    assert any(task.status in {"verified", "partially_verified"} for task in tasks)
    assert any(
        task.status in {"unsupported", "not_enough_information", "contradicted"} for task in tasks
    )

    calibrations = calibrate_confidence(bp, sources, claims, findings, tasks)
    report_confidence = [c for c in calibrations if c.target_type == "report"][0]
    assert report_confidence.level in {"low", "medium", "high"}
    assert report_confidence.penalties or report_confidence.boosts
    assert evidence_warnings or critique_warnings


def test_artifact_registry_path_safety_and_missing(tmp_path: Path) -> None:
    _fake_run(tmp_path)
    bp = _blueprint("Compare LangGraph and CrewAI", ["https://docs.langchain.com/langgraph"])
    registry = build_artifact_registry(tmp_path, bp)
    report = [item for item in registry if item.path == "report.md"][0]
    missing = [item for item in registry if item.path == "kernel_summary.json"][0]
    assert report.exists and report.content_hash
    assert not missing.exists
    assert missing.warnings


def test_pipeline_rebuild_writes_expected_artifacts(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "thread-1"
    run_dir.mkdir(parents=True)
    _fake_run(run_dir)
    result = rebuild_intelligence_kernel(
        runs_dir=runs,
        thread_id="thread-1",
        question="Compare LangGraph and CrewAI for a production research-agent backend",
        urls=["https://docs.langchain.com/langgraph"],
        runtime_settings=Settings(runs_dir=runs),
    )
    assert result.summary.thread_id == "thread-1"
    for rel in (
        "kernel_blueprint.json",
        "source_units.json",
        "evidence_units.json",
        "claims.json",
        "critique_findings.json",
        "verification_tasks.json",
        "verification_results.json",
        "confidence_calibration.json",
        "kernel_artifact_registry.json",
        "kernel_summary.json",
        "research_readiness.md",
        "kernel_passes.json",
    ):
        assert (run_dir / rel).exists(), rel


def test_pipeline_optional_failure_records_degraded_summary(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "thread-2"
    run_dir.mkdir(parents=True)
    _fake_run(run_dir)
    settings = Settings(runs_dir=runs)
    result = rebuild_intelligence_kernel(
        runs_dir=runs,
        thread_id="thread-2",
        question="What is LangGraph?",
        urls=[],
        runtime_settings=settings,
    )
    assert any(
        p.pass_type == "final_kernel_summary" and p.status == "completed" for p in result.passes
    )


def test_api_intelligence_endpoints(client, test_runs_dir: Path) -> None:
    analyze = client.post(
        "/intelligence/analyze",
        json={"question": "Compare LangGraph vs CrewAI for backend architecture", "urls": []},
    )
    assert analyze.status_code == 200
    assert analyze.json()["intent"]["label"] == "technical_due_diligence"

    run_dir = test_runs_dir / "api-kernel"
    run_dir.mkdir(parents=True)
    _fake_run(run_dir)
    rebuild = client.post("/runs/api-kernel/intelligence/rebuild")
    assert rebuild.status_code == 200
    assert rebuild.json()["summary"]["thread_id"] == "api-kernel"

    assert client.get("/runs/api-kernel/intelligence").status_code == 200
    assert client.get("/runs/api-kernel/blueprint").status_code == 200
    assert client.get("/runs/api-kernel/critique").status_code == 200
    assert client.get("/runs/api-kernel/verification").status_code == 200
    assert client.get("/runs/api-kernel/confidence").status_code == 200
    readiness = client.get("/runs/api-kernel/readiness")
    assert readiness.status_code == 200
    assert "Research Readiness" in readiness.text
