from __future__ import annotations

import json
from pathlib import Path

from deep_research_agent.artifacts import ensure_thread_dir
from deep_research_agent.synthesis import (
    cluster_findings,
    extract_findings,
    is_comparative_question,
    rebuild_synthesis_artifacts,
)
from deep_research_agent.synthesis.argument_map import build_argument_map
from deep_research_agent.synthesis.comparison_matrix import build_comparison_matrix
from deep_research_agent.synthesis.contracts import SynthesisInput
from deep_research_agent.synthesis.decision_memo import build_decision_memo
from deep_research_agent.synthesis.report_assembler import (
    assemble_report,
    build_report_assembly_plan,
    choose_report_profile,
)
from deep_research_agent.synthesis.uncertainty import build_uncertainty_boundaries


def _input() -> SynthesisInput:
    question = (
        "Should we choose LangGraph vs CrewAI for a production research agent architecture?"
    )
    langgraph_finding = (
        "LangGraph supports graph-based orchestration for complex agent workflows."
    )
    langgraph_normalized = (
        "langgraph supports graph-based orchestration for complex agent workflows"
    )
    return SynthesisInput(
        thread_id="syn-1",
        question=question,
        generated_at="2026-01-01T00:00:00Z",
        notes_text=f"- {langgraph_finding}",
        report_text=f"# Report\n\n{langgraph_finding} [S1]",
        sources=[
            {"source_id": "S1", "url": "https://example.com/langgraph", "ok": True},
            {"source_id": "S2", "url": "https://example.com/crewai", "ok": True},
        ],
        evidence_ledger={
            "claims": [
                {
                    "claim_id": "C1",
                    "text": langgraph_finding,
                    "normalized_text": langgraph_normalized,
                    "claim_type": "factual",
                    "source_ids": ["S1"],
                    "support_level": "strong",
                    "confidence_score": 0.82,
                    "citations": [{"source_id": "S1", "score": 0.8}],
                    "needs_human_review": False,
                },
                {
                    "claim_id": "C2",
                    "text": "CrewAI is simpler for role-based multi-agent teams.",
                    "normalized_text": "crewai is simpler for role-based multi-agent teams",
                    "claim_type": "comparative",
                    "source_ids": ["S2"],
                    "support_level": "moderate",
                    "confidence_score": 0.61,
                    "citations": [{"source_id": "S2", "score": 0.6}],
                    "needs_human_review": True,
                },
                {
                    "claim_id": "C3",
                    "text": "LangGraph may add operational complexity for small teams.",
                    "normalized_text": "langgraph may add operational complexity for small teams",
                    "claim_type": "factual",
                    "source_ids": ["S1"],
                    "support_level": "weak",
                    "confidence_score": 0.38,
                    "citations": [{"source_id": "S1", "score": 0.3}],
                    "needs_human_review": True,
                },
                {
                    "claim_id": "C4",
                    "text": "Teams should validate reliability with primary docs before adoption.",
                    "normalized_text": (
                        "teams should validate reliability with primary docs before adoption"
                    ),
                    "claim_type": "recommendation",
                    "source_ids": [],
                    "support_level": "unsupported",
                    "confidence_score": 0.0,
                    "needs_human_review": True,
                },
            ]
        },
        strategy={
            "intent": "comparative_analysis",
            "subquestions": [
                {
                    "id": "SQ1",
                    "question": "Which framework has better production orchestration support?",
                }
            ],
        },
        subquestions=[
            {
                "id": "SQ1",
                "question": "Which framework has better production orchestration support?",
            }
        ],
        available_artifacts=["notes.md", "report.md", "sources.json", "evidence_ledger.json"],
    )


def test_finding_extraction_and_clustering_from_evidence():
    findings = extract_findings(_input())
    assert len(findings) >= 4
    assert findings[0].source_ids == ["S1"]
    assert findings[0].confidence_label == "strong"

    clusters = cluster_findings(findings)
    kinds = {cluster.cluster_kind for cluster in clusters}
    assert {"topic", "source", "claim_type", "confidence", "contradiction"} <= kinds
    assert any(cluster.label == "S1" for cluster in clusters)


def test_comparative_question_detection_and_matrix_creation():
    synthesis_input = _input()
    findings = extract_findings(synthesis_input)

    assert is_comparative_question(synthesis_input.question) is True

    matrix = build_comparison_matrix(synthesis_input, findings)
    assert matrix.detected is True
    assert matrix.options[:2] == ["LangGraph", "CrewAI"]
    assert matrix.dimensions
    assert any(cell.option == "LangGraph" for cell in matrix.cells)


def test_decision_memo_creation():
    synthesis_input = _input()
    findings = extract_findings(synthesis_input)

    memo = build_decision_memo(synthesis_input, findings)
    assert memo.detected is True
    assert memo.options[:2] == ["LangGraph", "CrewAI"]
    assert memo.recommendation is not None
    assert memo.next_validation_steps


def test_argument_map_creation():
    synthesis_input = _input()
    findings = extract_findings(synthesis_input)

    argument_map = build_argument_map(
        thread_id=synthesis_input.thread_id,
        question=synthesis_input.question,
        generated_at=synthesis_input.generated_at,
        findings=findings,
    )
    assert argument_map.main_answer
    assert any(node.kind == "supporting_claim" for node in argument_map.nodes)
    assert any(node.kind == "weak_evidence" for node in argument_map.nodes)
    assert argument_map.relations


def test_uncertainty_boundary_generation():
    synthesis_input = _input()
    findings = extract_findings(synthesis_input)

    boundary = build_uncertainty_boundaries(synthesis_input, findings)
    assert boundary.known
    assert boundary.not_verified
    assert boundary.requires_human_review
    assert boundary.open_questions


def test_report_assembly_with_missing_optional_artifacts():
    synthesis_input = SynthesisInput(
        thread_id="minimal",
        question="Summarize Acme Search",
        generated_at="2026-01-01T00:00:00Z",
        notes_text="- Acme Search supports private indexes for enterprise customers.",
        report_text="Acme Search supports private indexes for enterprise customers.",
        sources=[],
        available_artifacts=["notes.md", "report.md"],
    )
    findings = extract_findings(synthesis_input)
    argument_map = build_argument_map(
        thread_id=synthesis_input.thread_id,
        question=synthesis_input.question,
        generated_at=synthesis_input.generated_at,
        findings=findings,
    )
    matrix = build_comparison_matrix(synthesis_input, findings)
    memo = build_decision_memo(synthesis_input, findings)
    boundary = build_uncertainty_boundaries(synthesis_input, findings)
    profile = choose_report_profile(synthesis_input, matrix, memo)
    plan = build_report_assembly_plan(
        synthesis_input,
        profile=profile,
        findings=findings,
        comparison_matrix=matrix,
        decision_memo=memo,
        uncertainty=boundary,
    )
    report = assemble_report(
        synthesis_input,
        profile=profile,
        findings=findings,
        argument_map=argument_map,
        comparison_matrix=matrix,
        decision_memo=memo,
        uncertainty=boundary,
        plan=plan,
    )

    assert plan.safe_to_replace_report is True
    assert "GENERATED SYNTHESIS" in report
    assert "Acme Search" in report


def test_artifact_writing_preserves_raw_report(tmp_path: Path):
    run_dir = tmp_path / "runs" / "syn-artifacts"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "thread_id": "syn-artifacts",
                "question": "Compare LangGraph vs CrewAI for production agents",
                "input_snapshot": {
                    "question": "Compare LangGraph vs CrewAI for production agents"
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "notes.md").write_text(
        "- LangGraph supports graph-based orchestration for production agents.\n"
        "- CrewAI supports role-based agent teams.\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        "# Report\n\nLangGraph supports graph-based orchestration for production agents.",
        encoding="utf-8",
    )
    (run_dir / "sources.json").write_text("[]\n", encoding="utf-8")

    output = rebuild_synthesis_artifacts(run_dir, thread_id="syn-artifacts")

    assert output.findings
    assert (run_dir / "argument_map.json").exists()
    assert (run_dir / "comparison_matrix.md").exists()
    assert (run_dir / "decision_memo.json").exists()
    assert (run_dir / "uncertainty_boundaries.md").exists()
    assert (run_dir / "report.raw.md").exists()
    assert "GENERATED SYNTHESIS" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_synthesis_rebuild_endpoint(client, test_runs_dir: Path):
    tid = "synthesis-endpoint"
    td = ensure_thread_dir(test_runs_dir, tid)
    (td / "run.json").write_text(
        json.dumps(
            {
                "thread_id": tid,
                "question": "Should we choose LangGraph vs CrewAI for a research backend?",
                "input_snapshot": {
                    "question": "Should we choose LangGraph vs CrewAI for a research backend?"
                },
            }
        ),
        encoding="utf-8",
    )
    (td / "notes.md").write_text(
        "- LangGraph supports graph-based orchestration for research agents.\n",
        encoding="utf-8",
    )
    (td / "report.md").write_text(
        "LangGraph supports graph-based orchestration for research agents.",
        encoding="utf-8",
    )
    (td / "sources.json").write_text("[]\n", encoding="utf-8")

    r = client.post(f"/runs/{tid}/synthesis/rebuild")
    assert r.status_code == 200
    body = r.json()
    assert body["thread_id"] == tid
    assert body["finding_count"] >= 1
    assert any(a["path"] == "argument_map.md" for a in body["artifacts"])

    ar = client.get(f"/runs/{tid}/argument-map")
    assert ar.status_code == 200
    assert ar.json()["thread_id"] == tid
