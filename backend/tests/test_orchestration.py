from __future__ import annotations

from pathlib import Path

from deep_research_agent.orchestration import (
    OrchestrationExecutor,
    ResearchTaskStatus,
    ResearchTaskType,
    StageInput,
    StageOutput,
    build_research_task_graph,
)
from deep_research_agent.orchestration.specialists import SpecialistStage, default_specialists


def _decision(graph, task_type: ResearchTaskType):
    node = next(node for node in graph.nodes if node.task_type == task_type)
    assert node.decision is not None
    return node.decision


def test_task_graph_creation_has_typed_nodes_and_valid_order():
    graph = build_research_task_graph(
        thread_id="graph-test",
        question="Compare LangGraph and CrewAI for production research workflows",
        urls=["https://example.com/a", "https://example.com/b"],
        available_source_count=2,
    )

    assert graph.thread_id == "graph-test"
    assert {node.task_type for node in graph.nodes} == set(ResearchTaskType)
    order = [node.task_type for node in graph.execution_order()]
    assert order.index(ResearchTaskType.QUESTION_NORMALIZATION) < order.index(
        ResearchTaskType.SYNTHESIS
    )
    assert order.index(ResearchTaskType.SYNTHESIS) < order.index(
        ResearchTaskType.CITATION_REVIEW
    )


def test_adaptive_routing_simple_summary_skips_heavy_review():
    graph = build_research_task_graph(
        thread_id="simple",
        question="Summarize this article",
        urls=["https://example.com/article"],
        available_source_count=1,
    )

    assert _decision(graph, ResearchTaskType.SYNTHESIS).should_run is True
    assert _decision(graph, ResearchTaskType.SOURCE_EXTRACTION).should_run is True
    assert _decision(graph, ResearchTaskType.CONTRADICTION_SCAN).should_run is False
    assert _decision(graph, ResearchTaskType.RISK_SCAN).should_run is False
    assert _decision(graph, ResearchTaskType.FINAL_REPORT_REVIEW).should_run is False


def test_adaptive_routing_comparison_requires_evidence_synthesis_and_citations():
    graph = build_research_task_graph(
        thread_id="comparison",
        question="Compare Vendor A vs Vendor B for API reliability and pricing",
        urls=["https://example.com/a", "https://example.com/b"],
        available_source_count=2,
    )

    assert _decision(graph, ResearchTaskType.SOURCE_TRIAGE).should_run is True
    assert _decision(graph, ResearchTaskType.EVIDENCE_COLLECTION).should_run is True
    assert _decision(graph, ResearchTaskType.SUBQUESTION_ANSWERING).should_run is True
    assert _decision(graph, ResearchTaskType.SYNTHESIS).should_run is True
    assert _decision(graph, ResearchTaskType.CITATION_REVIEW).should_run is True


def test_adaptive_routing_technical_due_diligence_requires_failure_review():
    graph = build_research_task_graph(
        thread_id="technical",
        question=(
            "Evaluate the production architecture, implementation risks, and failure modes "
            "for a backend API deployment"
        ),
        urls=["https://example.com/docs"],
        available_source_count=1,
    )

    assert graph.confidence_policy == "technical_due_diligence"
    assert _decision(graph, ResearchTaskType.RISK_SCAN).should_run is True
    assert _decision(graph, ResearchTaskType.FINAL_REPORT_REVIEW).should_run is True


def test_adaptive_routing_legal_question_uses_conservative_policy():
    graph = build_research_task_graph(
        thread_id="legal",
        question="What are the current compliance risks in this contract policy?",
        urls=["https://example.com/policy"],
        available_source_count=1,
    )

    assert graph.confidence_policy == "conservative"
    assert _decision(graph, ResearchTaskType.RISK_SCAN).should_run is True
    assert _decision(graph, ResearchTaskType.CONTRADICTION_SCAN).should_run is True
    assert _decision(graph, ResearchTaskType.FINAL_REPORT_REVIEW).should_run is True


def test_adaptive_routing_ambiguous_missing_urls_records_skips():
    graph = build_research_task_graph(
        thread_id="ambiguous",
        question="Research the best approach",
        urls=[],
        available_source_count=0,
    )

    assert "missing_urls" in graph.routing_signals
    assert _decision(graph, ResearchTaskType.SOURCE_TRIAGE).should_run is True
    assert _decision(graph, ResearchTaskType.SOURCE_EXTRACTION).should_run is False
    assert _decision(graph, ResearchTaskType.CITATION_REVIEW).should_run is False


def test_executor_records_valid_execution_order_and_skipped_reasons(tmp_path: Path):
    graph = build_research_task_graph(
        thread_id="execute-skips",
        question="Summarize this article",
        urls=["https://example.com/article"],
        available_source_count=1,
    )
    executor = OrchestrationExecutor(tmp_path)

    graph, outputs, summary = executor.execute(graph)

    assert summary.status == ResearchTaskStatus.SUCCEEDED
    assert outputs[0].task_type == ResearchTaskType.QUESTION_NORMALIZATION
    skipped = {
        output.task_type: output.skipped_reason
        for output in outputs
        if output.status == ResearchTaskStatus.SKIPPED
    }
    assert ResearchTaskType.RISK_SCAN in skipped
    assert skipped[ResearchTaskType.RISK_SCAN]


class FailingSynthesisSpecialist(SpecialistStage):
    task_type = ResearchTaskType.SYNTHESIS
    role = default_specialists()[ResearchTaskType.SYNTHESIS].role

    def _run(self, node_id: str, stage_input: StageInput) -> StageOutput:
        raise RuntimeError("forced synthesis failure")


def test_executor_failed_stage_handling_continues_with_typed_error(tmp_path: Path):
    graph = build_research_task_graph(
        thread_id="execute-failure",
        question="Compare A vs B for legal compliance",
        urls=["https://example.com/a", "https://example.com/b"],
        available_source_count=2,
    )
    specialists = default_specialists()
    specialists[ResearchTaskType.SYNTHESIS] = FailingSynthesisSpecialist()
    executor = OrchestrationExecutor(tmp_path, specialists=specialists)

    graph, outputs, summary = executor.execute(graph)

    assert summary.status == ResearchTaskStatus.FAILED
    failed = next(output for output in outputs if output.task_type == ResearchTaskType.SYNTHESIS)
    assert failed.error is not None
    assert failed.error.error_type == "RuntimeError"
    assert "forced synthesis failure" in failed.error.message
    assert ResearchTaskType.CITATION_REVIEW in {
        output.task_type for output in outputs if output.status == ResearchTaskStatus.SKIPPED
    }


def test_artifact_writing(tmp_path: Path):
    graph = build_research_task_graph(
        thread_id="artifact-run",
        question="Compare A vs B",
        urls=["https://example.com/a", "https://example.com/b"],
        available_source_count=2,
    )
    executor = OrchestrationExecutor(tmp_path)

    _, _, summary = executor.execute(graph)

    expected = {
        "task_graph.json",
        "task_graph.md",
        "stage_outputs.json",
        "specialist_findings.md",
        "orchestration_summary.json",
        "orchestration_summary.md",
    }
    assert expected.issubset(set(summary.artifact_paths))
    for rel_path in expected:
        assert (tmp_path / "artifact-run" / rel_path).exists()


def test_orchestration_preview_endpoint(client):
    r = client.post(
        "/orchestration/preview",
        json={
            "question": "Compare LangGraph and CrewAI for backend implementation risk",
            "urls": ["https://example.com/a", "https://example.com/b"],
            "available_source_count": 2,
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert body["task_graph"]["confidence_policy"] == "technical_due_diligence"
    assert body["summary"]["agent_instruction_block"]
    assert any(
        output["task_type"] == ResearchTaskType.RISK_SCAN.value
        and output["status"] == ResearchTaskStatus.SUCCEEDED.value
        for output in body["stage_outputs"]
    )
