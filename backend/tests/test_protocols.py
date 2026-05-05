from __future__ import annotations

import json
from pathlib import Path

from deep_research_agent.protocols import (
    ProtocolRegistry,
    built_in_profiles,
    select_protocol,
    write_protocol_artifacts,
)
from deep_research_agent.protocols.artifact_writer import PROTOCOL_ARTIFACTS
from deep_research_agent.protocols.instruction_builder import build_selection_instruction_block
from deep_research_agent.protocols.policy_packs import DEFAULT_PACK_PATH, load_policy_packs
from deep_research_agent.protocols.validators import load_policy_packs_from_json


def test_protocol_loading_has_required_domains():
    registry = ProtocolRegistry()
    ids = set(registry.ids())
    assert {
        "general_research",
        "technical_due_diligence",
        "software_framework_comparison",
        "implementation_planning",
        "source_code_or_library_review",
        "legal_policy_review",
        "market_research",
        "vendor_evaluation",
        "academic_literature_review",
        "financial_or_investment_risk_review",
        "medical_or_health_information_review",
        "news_or_current_events_review",
    } <= ids
    for protocol in registry.list_protocols():
        assert protocol.required_artifacts
        assert protocol.synthesis_profile.profile


def test_classifier_behavior_for_required_domains():
    cases = [
        (
            "Assess the technical due diligence risks for adopting this production architecture",
            [],
            "technical_due_diligence",
        ),
        (
            "Compare LangGraph vs CrewAI as a framework for a backend agent",
            [],
            "software_framework_comparison",
        ),
        (
            "Review GDPR compliance policy obligations for a SaaS privacy workflow",
            [],
            "legal_policy_review",
        ),
        (
            "What do clinical guidelines say about hypertension treatment options?",
            [],
            "medical_or_health_information_review",
        ),
        (
            "Review investment risk and SEC filing evidence for this public company",
            ["https://www.sec.gov/Archives/example"],
            "financial_or_investment_risk_review",
        ),
        (
            "Market research on customer demand, competitors, and TAM for AI search tools",
            [],
            "market_research",
        ),
        (
            "Academic literature review of retrieval augmented generation evaluation papers",
            ["https://arxiv.org/abs/2401.00000"],
            "academic_literature_review",
        ),
        ("Explain the history of research agents", [], "general_research"),
    ]
    for question, urls, expected in cases:
        selection = select_protocol(question=question, urls=urls)
        assert selection.selected_protocol.protocol_id == expected
        assert selection.confidence_score > 0
        assert selection.reasons


def test_sensitive_domain_selection_is_conservative():
    selection = select_protocol(question="Can this drug dosage be recommended for symptoms?")
    assert selection.selected_protocol.protocol_id == "medical_or_health_information_review"
    assert selection.review_recommended is True
    assert selection.effective_citation_policy.strictness == "primary_source_required"
    assert any("not medical advice" in warning.message.lower() for warning in selection.warnings)


def test_intelligence_profile_validation_and_defaults():
    profiles = built_in_profiles()
    assert profiles["fast_brief"].max_sources < profiles["deep_research"].max_sources
    assert profiles["primary_sources_only"].citation_strictness == "primary_source_required"
    assert profiles["offline_mock"].source_discovery_enabled is False
    for profile in profiles.values():
        assert profile.profile_id
        assert 0 <= profile.max_links_per_source_default <= 10


def test_policy_pack_validation_from_default_file():
    packs = load_policy_packs(DEFAULT_PACK_PATH)
    assert packs
    assert any(pack.pack_id == "sensitive_domain_conservative_language" for pack in packs)
    assert all(pack.applies_to_protocols for pack in packs)


def test_policy_pack_validation_rejects_invalid_file(tmp_path: Path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"packs": [{"pack_id": "bad"}]}), encoding="utf-8")
    try:
        load_policy_packs_from_json(invalid)
    except Exception as exc:
        assert "Invalid policy pack" in str(exc)
    else:
        raise AssertionError("invalid pack should fail validation")


def test_instruction_generation_mentions_required_policy():
    selection = select_protocol(question="Review legal compliance obligations for data retention")
    block = build_selection_instruction_block(selection)
    assert "legal_policy_review" in block
    assert "Do not present the output as professional advice" in block
    assert "Citation Rules" in block
    assert "Freshness Handling" in block


def test_artifact_writer_outputs_expected_files(tmp_path: Path):
    selection = select_protocol(question="Compare FastAPI vs Django for an API backend")
    written = write_protocol_artifacts(tmp_path, selection)
    assert set(PROTOCOL_ARTIFACTS) == set(written)
    payload = json.loads((tmp_path / "protocol_selection.json").read_text(encoding="utf-8"))
    assert payload["selected_protocol"]["protocol_id"] == "software_framework_comparison"
    assert (tmp_path / "protocol_instructions.md").read_text(encoding="utf-8").startswith(
        "# Protocol Instructions"
    )


def test_api_protocol_selection_route(client):
    response = client.post(
        "/protocols/select",
        json={"question": "Compare LangChain and LlamaIndex for implementation planning"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["selected_protocol"]["protocol_id"] in {
        "software_framework_comparison",
        "implementation_planning",
    }
    assert body["reasons"]


def test_run_integrates_protocol_settings_and_artifacts(client):
    response = client.post(
        "/run",
        json={
            "question": "Review legal compliance policy for data retention",
            "mock_mode": True,
            "thread_id": "protocol-run",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["protocol"]["selected_protocol"]["protocol_id"] == "legal_policy_review"
    assert body["run"]["review_status"] == "pending"
    paths = {artifact["path"] for artifact in body["artifacts"]}
    assert set(PROTOCOL_ARTIFACTS) <= paths

    protocol = client.get("/runs/protocol-run/protocol")
    assert protocol.status_code == 200
    payload = protocol.json()
    assert payload["selection"]["selected_protocol"]["protocol_id"] == "legal_policy_review"
    assert payload["requirements"]["review_recommended"] is True
