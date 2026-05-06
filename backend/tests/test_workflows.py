from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_research_agent.artifacts import ensure_thread_dir
from deep_research_agent.settings import Settings
from deep_research_agent.workflows import (
    WorkflowCompiler,
    WorkflowExecutionContext,
    WorkflowInput,
    WorkflowMode,
    WorkflowRebuildRequest,
    execute_workflow,
    preview_workflow,
    rebuild_workflow,
)
from deep_research_agent.workflows.artifact_contracts import (
    validate_artifact_contracts,
)
from deep_research_agent.workflows.contracts import (
    ArtifactType,
    WorkflowArtifactContract,
    WorkflowStageDefinition,
    WorkflowStageType,
)
from deep_research_agent.workflows.dependency_resolver import (
    build_dependency_graph,
    explain_dependency_path,
    find_downstream_stages,
    find_upstream_stages,
    topological_sort,
)
from deep_research_agent.workflows.policies import redact_secrets
from deep_research_agent.workflows.registry import (
    WorkflowTemplateRegistry,
    infer_mode_from_question,
    load_custom_templates,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        runs_dir=tmp_path / "runs",
        model_provider="mock",
        workflows_allow_mock_agent=True,
        evaluation_lab_gate_runs_dir=tmp_path / "gate-runs",
        evaluation_lab_runs_dir=tmp_path / "lab-runs",
        evaluation_lab_baselines_dir=tmp_path / "baselines",
    )


def test_built_in_templates_validate_and_select():
    registry = WorkflowTemplateRegistry()
    templates = registry.list_templates()
    assert {template.mode for template in templates} >= {
        WorkflowMode.quick_brief,
        WorkflowMode.deep_research,
        WorkflowMode.legal_policy_review,
        WorkflowMode.offline_benchmark,
    }
    quick = registry.get_template_for_mode(WorkflowMode.quick_brief)
    deep = registry.get_template_for_mode(WorkflowMode.deep_research)
    verification = registry.get_template_for_mode(WorkflowMode.verification_only)
    rebuild = registry.get_template_for_mode(WorkflowMode.rebuild_from_artifacts)
    legal = registry.get_template_for_mode(WorkflowMode.legal_policy_review)
    offline = registry.get_template_for_mode(WorkflowMode.offline_benchmark)

    assert len(quick.stages) < len(deep.stages)
    assert WorkflowStageType.agent_execution not in {
        stage.stage_type for stage in verification.stages
    }
    assert WorkflowStageType.source_fetching not in {stage.stage_type for stage in rebuild.stages}
    assert legal.policies[0].strict_citations is True
    assert offline.policies[0].external_network_allowed is False


@pytest.mark.parametrize(
    ("question", "mode"),
    [
        ("Compare LangGraph vs CrewAI", WorkflowMode.framework_comparison),
        ("Build an implementation plan", WorkflowMode.implementation_planning),
        ("Review legal compliance policy", WorkflowMode.legal_policy_review),
        ("Detect prompt injection in source", WorkflowMode.adversarial_source_review),
    ],
)
def test_infer_mode(question: str, mode: WorkflowMode):
    assert infer_mode_from_question(WorkflowInput(question=question)) == mode


def test_artifact_contract_validation(tmp_path: Path):
    run_dir = ensure_thread_dir(tmp_path / "runs", "contract")
    (run_dir / "plan.md").write_text("# Plan\n\n- Real step\n", encoding="utf-8")
    (run_dir / "sources.json").write_text('[{"url":"https://example.com"}]\n', encoding="utf-8")
    required = WorkflowArtifactContract(
        artifact_name="plan.md",
        artifact_type=ArtifactType.markdown,
        required=True,
        min_size_bytes=1,
        must_be_nonempty=True,
    )
    optional = WorkflowArtifactContract(
        artifact_name="optional.json",
        artifact_type=ArtifactType.json,
        required=False,
        must_parse_as_json=True,
    )
    invalid = WorkflowArtifactContract(
        artifact_name="broken.json",
        artifact_type=ArtifactType.json,
        required=True,
        must_parse_as_json=True,
    )
    (run_dir / "broken.json").write_text("{bad", encoding="utf-8")
    report = validate_artifact_contracts(run_dir, [required, optional, invalid])
    statuses = {result.artifact_name: result.status.value for result in report.results}
    assert statuses["plan.md"] == "passed"
    assert statuses["optional.json"] == "passed"
    assert statuses["broken.json"] == "failed"
    with pytest.raises(ValueError):
        WorkflowArtifactContract(artifact_name="../bad", artifact_type=ArtifactType.text)


def test_dependency_resolver():
    stages = [
        WorkflowStageDefinition(
            stage_id="a", name="A", stage_type=WorkflowStageType.input_snapshot
        ),
        WorkflowStageDefinition(
            stage_id="b",
            name="B",
            stage_type=WorkflowStageType.request_analysis,
            depends_on=["a"],
        ),
        WorkflowStageDefinition(
            stage_id="c",
            name="C",
            stage_type=WorkflowStageType.agent_execution,
            depends_on=["b"],
        ),
    ]
    assert topological_sort(stages) == ["a", "b", "c"]
    graph = build_dependency_graph(stages)
    assert graph.execution_layers == [["a"], ["b"], ["c"]]
    assert find_downstream_stages("a", stages) == ["b", "c"]
    assert find_upstream_stages("c", stages) == ["b", "a"]
    assert explain_dependency_path("a", "c", stages) == ["a", "b", "c"]


def test_compile_preview_and_policy_warnings(tmp_path: Path):
    settings = _settings(tmp_path)
    compiler = WorkflowCompiler(settings)
    compiled = compiler.compile(
        WorkflowInput(
            question="Compare LangGraph vs CrewAI", mode=WorkflowMode.framework_comparison
        ),
        write_artifacts=True,
    )
    assert compiled.mode == WorkflowMode.framework_comparison
    assert "agent_execution" in compiled.execution_order
    assert (settings.runs_dir / compiled.thread_id / "workflow_execution_plan.md").exists()

    preview = preview_workflow(
        WorkflowInput(question="Review legal policy", mode=WorkflowMode.legal_policy_review),
        settings=settings,
    )
    assert preview.mode == WorkflowMode.legal_policy_review
    assert any(warning.code == "human_review_required" for warning in preview.warnings)
    assert not (settings.runs_dir / preview.template_id).exists()


def test_compile_rebuild_and_verification_no_refetch(tmp_path: Path):
    settings = _settings(tmp_path)
    compiled = WorkflowCompiler(settings).compile(
        WorkflowInput(
            question="Verify existing run",
            thread_id="existing",
            existing_run_id="existing",
            mode=WorkflowMode.verification_only,
        )
    )
    stage_types = {stage.stage_type for stage in compiled.stages}
    assert WorkflowStageType.source_fetching not in stage_types
    assert WorkflowStageType.agent_execution not in stage_types
    assert any(policy.external_network_allowed is False for policy in compiled.policies)


def test_execute_quick_brief_mock_writes_required_artifacts(tmp_path: Path):
    settings = _settings(tmp_path)
    compiled = WorkflowCompiler(settings).compile(
        WorkflowInput(
            question="Summarize provided material",
            urls=["https://example.invalid"],
            mode=WorkflowMode.quick_brief,
            settings_overrides={"model_provider": "mock", "mock_mode": True},
        ),
        write_artifacts=True,
    )
    result = execute_workflow(
        compiled,
        WorkflowExecutionContext(compiled, settings=settings, dry_run=False),
    )
    run_dir = settings.runs_dir / compiled.thread_id
    assert result.status.value in {"completed", "completed_with_warnings", "degraded"}
    assert (run_dir / "plan.md").exists()
    assert "MOCK OUTPUT" in (run_dir / "report.md").read_text(encoding="utf-8")
    assert (run_dir / "workflow_manifest.json").exists()
    assert (run_dir / "workflow_readiness.json").exists()
    assert (run_dir / "workflow_stage_results.json").exists()


def test_dry_run_does_not_write_report(tmp_path: Path):
    settings = _settings(tmp_path)
    compiled = WorkflowCompiler(settings).compile(
        WorkflowInput(question="Dry run only", mode=WorkflowMode.quick_brief)
    )
    result = execute_workflow(
        compiled,
        WorkflowExecutionContext(compiled, settings=settings, dry_run=True),
    )
    assert all(stage.status.value == "skipped" for stage in result.stages)
    assert not (settings.runs_dir / compiled.thread_id / "report.md").exists()


def test_rebuild_verification_only(tmp_path: Path):
    settings = _settings(tmp_path)
    run_dir = ensure_thread_dir(settings.runs_dir, "reb")
    (run_dir / "plan.md").write_text("# Plan\n\n- Existing\n", encoding="utf-8")
    (run_dir / "notes.md").write_text("# Notes\n\n- Existing\n", encoding="utf-8")
    (run_dir / "sources.json").write_text("[]\n", encoding="utf-8")
    (run_dir / "report.md").write_text("# Report\n\nExisting report.\n", encoding="utf-8")
    result = rebuild_workflow(
        WorkflowRebuildRequest(thread_id="reb", mode=WorkflowMode.verification_only),
        settings=settings,
    )
    assert "verification" in result.stages_rebuilt
    assert "source_fetching" not in result.stages_rebuilt


def test_offline_benchmark_quality_gate_stage(tmp_path: Path):
    settings = _settings(tmp_path)
    compiled = WorkflowCompiler(settings).compile(
        WorkflowInput(
            question="Run offline benchmark gate",
            mode=WorkflowMode.offline_benchmark,
            run_quality_gate=True,
            settings_overrides={"model_provider": "mock", "mock_mode": True},
        ),
        write_artifacts=True,
    )
    result = execute_workflow(compiled, WorkflowExecutionContext(compiled, settings=settings))
    run_dir = settings.runs_dir / compiled.thread_id
    assert (run_dir / "workflow_quality_gate_result.json").exists()
    gate = json.loads((run_dir / "workflow_quality_gate_result.json").read_text(encoding="utf-8"))
    assert gate["status"] == "passed"
    assert result.quality_gate_result["status"] == "passed"


def test_api_workflow_endpoints(client):
    templates = client.get("/workflows/templates")
    assert templates.status_code == 200
    assert any(item["template_id"] == "quick_brief" for item in templates.json())

    preview = client.post(
        "/workflows/preview",
        json={"question": "Compare A vs B", "mode": "framework_comparison"},
    )
    assert preview.status_code == 200
    assert preview.json()["mode"] == "framework_comparison"

    compiled = client.post(
        "/workflows/compile",
        json={"question": "Compile workflow", "mode": "quick_brief"},
    )
    assert compiled.status_code == 200
    compiled_tid = compiled.json()["thread_id"]
    assert client.get(f"/runs/{compiled_tid}/workflow/plan").status_code == 200

    run = client.post(
        "/workflows/run",
        json={
            "question": "Workflow mock run",
            "mode": "quick_brief",
            "settings_overrides": {"model_provider": "mock", "mock_mode": True},
        },
    )
    assert run.status_code == 200
    tid = run.json()["thread_id"]
    assert client.get(f"/runs/{tid}/workflow/manifest").status_code == 200
    assert client.get(f"/runs/{tid}/workflow/readiness").status_code == 200
    assert client.get(f"/runs/{tid}/workflow/stages").status_code == 200
    assert client.get(f"/runs/{tid}/workflow/plan").status_code == 200

    rebuild = client.post(
        f"/runs/{tid}/workflows/rebuild",
        json={"thread_id": tid, "mode": "verification_only"},
    )
    assert rebuild.status_code == 200
    assert "verification" in rebuild.json()["stages_rebuilt"]

    compat = client.post(
        "/run",
        json={
            "question": "Workflow through compat endpoint",
            "workflow_mode": "quick_brief",
            "mock_mode": True,
        },
    )
    assert compat.status_code == 200
    assert compat.json()["workflow_id"].startswith("wf-")


def test_path_safety_and_secret_redaction(tmp_path: Path):
    settings = _settings(tmp_path)
    compiler = WorkflowCompiler(settings)
    with pytest.raises(ValueError):
        compiler.compile(WorkflowInput(question="bad", thread_id="../escape"))
    redacted = redact_secrets(
        {"openai_api_key": "sk-test", "nested": {"authorization": "Bearer abc"}, "safe": "ok"}
    )
    assert redacted["openai_api_key"] == "[REDACTED]"
    assert redacted["nested"]["authorization"] == "[REDACTED]"
    assert redacted["safe"] == "ok"


def test_custom_template_loading_requires_enablement(tmp_path: Path):
    registry = WorkflowTemplateRegistry()
    template = registry.get_template_for_mode(WorkflowMode.quick_brief)
    custom_dir = tmp_path / "templates"
    custom_dir.mkdir()
    (custom_dir / "quick.json").write_text(
        json.dumps(template.to_json_dict(), indent=2),
        encoding="utf-8",
    )
    assert load_custom_templates(custom_dir, enabled=False) == []
    loaded = load_custom_templates(custom_dir, enabled=True)
    assert loaded[0].template_id == "quick_brief"
