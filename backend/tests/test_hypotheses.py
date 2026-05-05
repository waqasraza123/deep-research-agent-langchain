from __future__ import annotations

import json
from pathlib import Path

from deep_research_agent.evidence.citation_mapper import SourceDocument
from deep_research_agent.evidence.contracts import EvidenceSource
from deep_research_agent.hypotheses.artifact_writer import rebuild_hypothesis_artifacts
from deep_research_agent.hypotheses.confidence import confidence_update_for_hypothesis
from deep_research_agent.hypotheses.contracts import (
    HypothesisSet,
    HypothesisStatus,
    HypothesisSummary,
    HypothesisTestResult,
    HypothesisType,
    ResearchHypothesis,
)
from deep_research_agent.hypotheses.contradiction_mapper import map_contradictions
from deep_research_agent.hypotheses.evidence_tester import test_hypothesis as run_hypothesis_test
from deep_research_agent.hypotheses.generator import (
    HypothesisBuildInput,
    generate_hypotheses,
)


def test_hypothesis_generation_from_comparative_question():
    build_input = HypothesisBuildInput(
        thread_id="t1",
        question="Is LangGraph better than CrewAI for a production research-agent backend?",
        generated_at="2026-05-05T00:00:00Z",
        sources=[{"id": "S1"}],
    )

    hypothesis_set = generate_hypotheses(build_input)

    texts = [item.text for item in hypothesis_set.hypotheses]
    assert len(texts) >= 5
    assert any("LangGraph" in text and "CrewAI" in text for text in texts)
    assert any(
        item.hypothesis_type == HypothesisType.TECHNICAL
        for item in hypothesis_set.hypotheses
    )
    assert any(
        item.hypothesis_type == HypothesisType.RECOMMENDATION
        for item in hypothesis_set.hypotheses
    )


def test_hypothesis_generation_from_technical_question():
    build_input = HypothesisBuildInput(
        thread_id="t2",
        question="Can this API backend support durable checkpointing and observability?",
        generated_at="2026-05-05T00:00:00Z",
        subquestions=[
            {
                "id": "SQ1",
                "question": "Does the backend persist state for resumability?",
            }
        ],
        sources=[{"id": "S1"}],
    )

    hypothesis_set = generate_hypotheses(build_input)

    assert any(
        item.hypothesis_type == HypothesisType.TECHNICAL
        for item in hypothesis_set.hypotheses
    )
    assert any("operational risks" in item.text for item in hypothesis_set.hypotheses)
    assert any(item.subquestion_ids == ["SQ1"] for item in hypothesis_set.hypotheses)


def test_hypothesis_evidence_matching():
    hypothesis = ResearchHypothesis(
        hypothesis_id="H1",
        text="LangGraph offers stronger checkpointing for durable agent execution.",
        normalized_text="langgraph offers stronger checkpointing for durable agent execution",
        hypothesis_type=HypothesisType.TECHNICAL,
    )
    source = SourceDocument(
        source=EvidenceSource(source_id="S1", quality_score=0.9),
        text="LangGraph offers checkpointing and durable execution for stateful agents.",
    )

    result = run_hypothesis_test(hypothesis, source_docs=[source], claims=[], source_audit={})

    assert result.supporting_evidence
    assert result.support_score >= 0.34
    assert result.status in {
        HypothesisStatus.PARTIALLY_SUPPORTED,
        HypothesisStatus.INCONCLUSIVE,
        HypothesisStatus.SUPPORTED,
    }


def test_contradiction_detection_between_hypotheses():
    hypothesis_set = HypothesisSet(
        hypothesis_set_id="HS1",
        thread_id="t1",
        question="Compare tools",
        generated_at="2026-05-05T00:00:00Z",
        hypotheses=[
            ResearchHypothesis(
                hypothesis_id="H1",
                text="LangGraph is better than CrewAI for production orchestration.",
                normalized_text="langgraph is better than crewai for production orchestration",
                hypothesis_type=HypothesisType.COMPARATIVE,
            ),
            ResearchHypothesis(
                hypothesis_id="H2",
                text="LangGraph is worse than CrewAI for production orchestration.",
                normalized_text="langgraph is worse than crewai for production orchestration",
                hypothesis_type=HypothesisType.COMPARATIVE,
            ),
        ],
        summary=HypothesisSummary(thread_id="t1", generated_at="2026-05-05T00:00:00Z"),
    )

    mapped = map_contradictions(
        hypothesis_set,
        HypothesisBuildInput(
            thread_id="t1",
            question="Compare tools",
            generated_at="2026-05-05T00:00:00Z",
        ),
    )

    assert mapped.contradictions
    assert mapped.hypotheses[0].competing_hypothesis_ids == ["H2"]


def test_confidence_update_scoring():
    hypothesis = ResearchHypothesis(
        hypothesis_id="H1",
        text="LangGraph supports durable checkpointing.",
        normalized_text="langgraph supports durable checkpointing",
        hypothesis_type=HypothesisType.TECHNICAL,
        status=HypothesisStatus.SUPPORTED,
    )
    result = HypothesisTestResult(
        result_id="HT1",
        hypothesis_id="H1",
        status=HypothesisStatus.SUPPORTED,
        support_score=0.8,
        supporting_source_ids=["S1", "S2"],
        source_diversity=2,
        primary_source_count=1,
        citation_ready_count=1,
    )

    update = confidence_update_for_hypothesis(
        hypothesis,
        result,
        HypothesisBuildInput(
            thread_id="t1",
            question="Can LangGraph support production durable checkpointing?",
            generated_at="2026-05-05T00:00:00Z",
        ),
    )

    assert update.posterior_score >= 0.5
    assert update.confidence_level in {"medium", "high", "very_high"}


def test_confidence_adjusts_for_temporal_quantitative_and_safety_risks():
    hypothesis = ResearchHypothesis(
        hypothesis_id="H1",
        text="ProductX currently has 99.9% uptime.",
        normalized_text="productx currently has 99.9 uptime",
        hypothesis_type=HypothesisType.TECHNICAL,
        status=HypothesisStatus.SUPPORTED,
    )
    result = HypothesisTestResult(
        result_id="HT1",
        hypothesis_id="H1",
        status=HypothesisStatus.SUPPORTED,
        support_score=0.78,
        supporting_source_ids=["S1"],
        source_diversity=1,
        primary_source_count=1,
        citation_ready_count=1,
    )

    clean_update = confidence_update_for_hypothesis(
        hypothesis,
        result,
        HypothesisBuildInput(
            thread_id="t1",
            question="Does ProductX currently have 99.9% uptime?",
            generated_at="2026-05-05T00:00:00Z",
        ),
    )
    risk_update = confidence_update_for_hypothesis(
        hypothesis,
        result,
        HypothesisBuildInput(
            thread_id="t1",
            question="Does ProductX currently have 99.9% uptime?",
            generated_at="2026-05-05T00:00:00Z",
            currentness_assessment={
                "freshness_required": True,
                "status": "stale",
                "stale_sources": ["S1"],
            },
            quantitative_profile={
                "warnings": [{"severity": "high", "message": "Unsupported numeric claim"}],
                "consistency_failures": 1,
                "evidence": {
                    "numeric_claims": [{"support_status": "unsupported"}],
                    "comparisons": [{"comparable": False}],
                },
            },
            source_safety={
                "assessments": [
                    {
                        "source_id": "S1",
                        "risk_score": {"risk_level": "high"},
                        "sanitized_content": {"agent_context_allowed": False},
                    }
                ]
            },
        ),
    )

    assert risk_update.posterior_score < clean_update.posterior_score
    assert any("stale" in penalty.lower() for penalty in risk_update.penalties)
    assert any("source-safety" in penalty.lower() for penalty in risk_update.penalties)
    assert any("quantitative" in penalty.lower() for penalty in risk_update.penalties)


def test_low_confidence_behavior_when_sources_are_weak(tmp_path: Path):
    run_dir = tmp_path / "runs" / "weak"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"question": "Is ProductX the best legal compliance option?"}),
        encoding="utf-8",
    )
    (run_dir / "plan.md").write_text("# Plan\n", encoding="utf-8")
    (run_dir / "notes.md").write_text("- ProductX is the best option.\n", encoding="utf-8")
    (run_dir / "report.md").write_text(
        "ProductX is the best legal compliance option.\n",
        encoding="utf-8",
    )
    (run_dir / "sources.json").write_text("[]\n", encoding="utf-8")

    hypothesis_set = rebuild_hypothesis_artifacts(run_dir, thread_id="weak")

    assert hypothesis_set.summary.average_confidence < 0.5
    assert any(
        update.needs_human_review and update.confidence_level in {"very_low", "low"}
        for update in hypothesis_set.confidence_updates
    )
    assert any(
        item.status == HypothesisStatus.NEEDS_MORE_EVIDENCE
        for item in hypothesis_set.hypotheses
    )


def test_hypothesis_artifact_writing(tmp_path: Path):
    run_dir = tmp_path / "runs" / "t1"
    sources_dir = run_dir / "sources"
    sources_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps({"question": "Is LangGraph viable for production checkpointing?"}),
        encoding="utf-8",
    )
    (sources_dir / "s1.txt").write_text(
        "LangGraph supports durable execution and checkpointing for stateful agent workflows.",
        encoding="utf-8",
    )
    (run_dir / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": "S1",
                    "url": "https://example.com/langgraph",
                    "title": "LangGraph docs",
                    "local_path": "runs/t1/sources/s1.txt",
                    "ok": True,
                    "word_count": 500,
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "plan.md").write_text("# Plan\n", encoding="utf-8")
    (run_dir / "notes.md").write_text(
        "LangGraph supports durable checkpointing for stateful agent workflows.",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        "LangGraph supports durable checkpointing for stateful agent workflows.",
        encoding="utf-8",
    )

    hypothesis_set = rebuild_hypothesis_artifacts(run_dir, thread_id="t1")

    assert hypothesis_set.hypotheses
    for name in (
        "hypotheses.json",
        "hypotheses.md",
        "hypothesis_tests.json",
        "hypothesis_tests.md",
        "hypothesis_graph.json",
        "hypothesis_graph.md",
        "confidence_updates.json",
        "confidence_updates.md",
    ):
        assert (run_dir / name).exists()
    graph = json.loads((run_dir / "hypothesis_graph.json").read_text(encoding="utf-8"))
    assert graph["nodes"]


def test_hypothesis_rebuild_endpoint(client):
    r = client.post("/run", json={"question": "Can LangGraph support durable checkpointing?"})
    tid = r.json()["thread_id"]

    hr = client.post(f"/runs/{tid}/hypotheses/rebuild")
    assert hr.status_code == 200
    body = hr.json()
    assert body["thread_id"] == tid
    assert body["summary"]["total_hypotheses"] >= 1
    assert any(a["path"] == "hypotheses.md" for a in body["artifacts"])

    graph = client.get(f"/runs/{tid}/hypothesis-graph")
    assert graph.status_code == 200
    assert graph.json()["nodes"]
