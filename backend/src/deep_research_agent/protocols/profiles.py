from __future__ import annotations

from .contracts import IntelligenceProfile
from .errors import UnknownProfileError


def built_in_profiles() -> dict[str, IntelligenceProfile]:
    profiles = [
        IntelligenceProfile(
            profile_id="fast_brief",
            name="Fast Brief",
            description="Short, low-latency research with minimal source expansion.",
            max_sources=2,
            max_chunks=8,
            follow_links_default=False,
            max_links_per_source_default=0,
            verification_strictness="standard",
            synthesis_enabled=True,
            evaluation_required=False,
            report_length_preference="brief",
        ),
        IntelligenceProfile(
            profile_id="balanced_research",
            name="Balanced Research",
            description="Default profile balancing speed, coverage, citations, and synthesis.",
            max_sources=3,
            max_chunks=30,
            follow_links_default=False,
            max_links_per_source_default=0,
            verification_strictness="standard",
            synthesis_enabled=True,
            evaluation_required=True,
            report_length_preference="standard",
        ),
        IntelligenceProfile(
            profile_id="deep_research",
            name="Deep Research",
            description="Broader source coverage and full downstream synthesis and evaluation.",
            max_sources=8,
            max_chunks=120,
            follow_links_default=True,
            max_links_per_source_default=3,
            verification_strictness="high",
            synthesis_enabled=True,
            evaluation_required=True,
            review_gate_recommended=True,
            report_length_preference="long",
        ),
        IntelligenceProfile(
            profile_id="conservative_verification",
            name="Conservative Verification",
            description="High-stakes review mode with strict source and citation requirements.",
            max_sources=6,
            max_chunks=80,
            follow_links_default=True,
            max_links_per_source_default=2,
            verification_strictness="very_high",
            synthesis_enabled=True,
            evaluation_required=True,
            review_gate_recommended=True,
            citation_strictness="primary_source_required",
            report_length_preference="standard",
        ),
        IntelligenceProfile(
            profile_id="technical_architect",
            name="Technical Architect",
            description="Technical decision profile emphasizing docs, source code, risks, and plans.",
            max_sources=6,
            max_chunks=80,
            follow_links_default=True,
            max_links_per_source_default=2,
            verification_strictness="high",
            synthesis_enabled=True,
            evaluation_required=True,
            citation_strictness="strict",
            report_length_preference="long",
        ),
        IntelligenceProfile(
            profile_id="citation_strict",
            name="Citation Strict",
            description="Requires claim-level citations and stronger source audit signals.",
            max_sources=5,
            max_chunks=60,
            follow_links_default=False,
            max_links_per_source_default=0,
            verification_strictness="high",
            synthesis_enabled=True,
            evaluation_required=True,
            review_gate_recommended=True,
            citation_strictness="strict",
        ),
        IntelligenceProfile(
            profile_id="primary_sources_only",
            name="Primary Sources Only",
            description="Prioritizes official, primary, regulatory, code, paper, or filing sources.",
            max_sources=5,
            max_chunks=60,
            follow_links_default=False,
            max_links_per_source_default=0,
            verification_strictness="very_high",
            synthesis_enabled=True,
            evaluation_required=True,
            review_gate_recommended=True,
            citation_strictness="primary_source_required",
        ),
        IntelligenceProfile(
            profile_id="offline_mock",
            name="Offline Mock",
            description="Deterministic offline profile for tests and dry runs.",
            max_sources=1,
            max_chunks=4,
            follow_links_default=False,
            max_links_per_source_default=0,
            verification_strictness="low",
            source_discovery_enabled=False,
            synthesis_enabled=True,
            evaluation_required=False,
            citation_strictness="standard",
            report_length_preference="brief",
        ),
    ]
    return {profile.profile_id: profile for profile in profiles}


def get_profile(profile_id: str | None) -> IntelligenceProfile:
    profiles = built_in_profiles()
    key = profile_id or "balanced_research"
    try:
        return profiles[key]
    except KeyError:
        raise UnknownProfileError(key) from None

