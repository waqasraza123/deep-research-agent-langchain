from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_research_agent.artifacts import write_strategy_artifacts
from deep_research_agent.intelligence import (
    ComplexityLevel,
    ResearchIntent,
    classify_research_intent,
    create_research_strategy,
)
from deep_research_agent.intelligence.decomposition import decompose_question
from deep_research_agent.intelligence.scoring import assess_complexity


def test_intent_classification_is_rule_assisted_for_comparison():
    intent, confidence, matches = classify_research_intent(
        "Compare LangGraph and CrewAI for a production research agent backend"
    )

    assert intent == ResearchIntent.COMPARATIVE_ANALYSIS
    assert confidence >= 0.6
    assert any("compare" in match for match in matches)


def test_decomposition_outputs_expected_technical_comparison_shape():
    result = decompose_question(
        "Compare LangGraph and CrewAI for a production research agent backend",
        ResearchIntent.COMPARATIVE_ANALYSIS,
    )

    joined = " ".join(sq.question.lower() for sq in result.subquestions)
    assert result.primary_question.endswith("?")
    assert len(result.subquestions) >= 5
    assert "orchestration" in joined
    assert "checkpointing" in joined
    assert "deployment" in joined
    assert result.required_definitions
    assert result.assumptions_to_verify
    assert result.missing_information_warnings


def test_complexity_scoring_detects_multi_domain_and_freshness():
    assessment = assess_complexity(
        "Assess current legal, financial, security, and market risks of deploying "
        "AI agents in 2026",
        urls=["https://example.com/a", "https://example.com/b"],
    )

    assert assessment.level == ComplexityLevel.MULTI_DOMAIN
    assert assessment.freshness_required is True
    assert assessment.domain_count >= 3
    assert assessment.numeric_score >= 45


def test_strategy_serializes_to_json_and_markdown():
    strategy = create_research_strategy(
        "Compare LangGraph and CrewAI for a production research agent backend",
        ["https://docs.langchain.com/"],
    )

    payload = json.loads(strategy.to_json())
    markdown = strategy.to_markdown()

    assert payload["intent"] == "comparative_analysis"
    assert payload["subquestions"]
    assert payload["verification_plan"]
    assert "Research Strategy" in markdown
    assert "Agent Instructions" in markdown


def test_strategy_artifact_writer_is_path_safe(tmp_path: Path):
    strategy = create_research_strategy("Summarize LangGraph persistence support")
    runs_dir = tmp_path / "runs"

    artifacts = write_strategy_artifacts(runs_dir, "safe-thread", strategy)
    paths = {artifact.path for artifact in artifacts}

    assert {"strategy.json", "strategy.md", "subquestions.json", "verification_plan.md"} == paths
    assert json.loads((runs_dir / "safe-thread" / "strategy.json").read_text(encoding="utf-8"))

    with pytest.raises(ValueError):
        write_strategy_artifacts(runs_dir, "../bad", strategy)
