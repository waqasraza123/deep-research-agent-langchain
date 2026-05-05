from __future__ import annotations

from .contracts import IntelligenceProfile, PolicyPack, ProtocolSelection, ResearchProtocol


def _bullet(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items if item]


def build_protocol_instruction_block(
    *,
    protocol: ResearchProtocol,
    profile: IntelligenceProfile,
    policy_packs: list[PolicyPack] | None = None,
) -> str:
    packs = policy_packs or []
    source_lines: list[str] = []
    for requirement in protocol.required_source_types:
        required = "required" if requirement.required else "preferred"
        count = (
            f"minimum {requirement.minimum_count}" if requirement.minimum_count else "no minimum"
        )
        freshness = (
            f"; max age {requirement.freshness_days} days" if requirement.freshness_days else ""
        )
        source_lines.append(
            f"- `{requirement.source_type}` ({required}, {count}{freshness}): "
            f"{requirement.rationale}"
        )
    for pack in packs:
        for requirement in pack.source_requirements:
            required = "required" if requirement.required else "preferred"
            source_lines.append(
                f"- `{requirement.source_type}` from policy pack `{pack.pack_id}` "
                f"({required}): {requirement.rationale}"
            )

    citation = protocol.citation_requirements
    freshness = protocol.freshness_requirements
    verification = protocol.verification_strictness
    synthesis = protocol.synthesis_profile
    safety = protocol.safety_warnings

    lines = [
        "# Protocol Instructions",
        "",
        "## Research Protocol",
        "",
        f"- Protocol: `{protocol.protocol_id}` ({protocol.name})",
        f"- Description: {protocol.description}",
        f"- Intelligence profile: `{profile.profile_id}` ({profile.name})",
        f"- Verification strictness: `{verification.strictness}`",
        "",
        "## Source Requirements",
        "",
        *(source_lines or ["- Use the strongest available authoritative sources."]),
        "",
        "## Citation Rules",
        "",
        f"- Citation strictness: `{citation.strictness}`",
        f"- Inline citations required: `{citation.require_inline_citations}`",
        f"- Claim-level citations required: `{citation.require_claim_level_citations}`",
        f"- Source IDs required: `{citation.require_source_ids}`",
    ]
    if citation.primary_source_required_for:
        lines.extend(
            ["- Primary sources required for:"]
            + [f"  - {item}" for item in citation.primary_source_required_for]
        )
    if citation.disallowed_citation_sources:
        disallowed = ", ".join(f"`{item}`" for item in citation.disallowed_citation_sources)
        lines.append(f"- Do not cite these source types directly: {disallowed}")

    lines.extend(
        [
            "",
            "## Freshness Handling",
            "",
            f"- Freshness strictness: `{freshness.strictness}`",
            f"- Maximum source age days: `{freshness.max_age_days}`",
            f"- Publication dates required: `{freshness.require_publication_dates}`",
            f"- Retrieval date required: `{freshness.require_retrieval_date}`",
            f"- Stale-source handling: {freshness.stale_source_handling}",
            "",
            "## Verification Expectations",
            "",
            f"- Minimum independent sources: `{verification.minimum_independent_sources}`",
            "- Primary source required for decisive claims: "
            f"`{verification.require_primary_source_for_decisive_claims}`",
            f"- Contradiction scan required: `{verification.require_contradiction_scan}`",
            f"- Uncertainty boundaries required: `{verification.require_uncertainty_boundaries}`",
            "",
            "## Forbidden Overclaims",
            "",
            *(
                _bullet(synthesis.forbidden_overclaims)
                or ["- Do not make claims that are not supported by captured sources."]
            ),
            "",
            "## Required Uncertainty Language",
            "",
            *(
                _bullet(synthesis.required_uncertainty_language)
                or ["- State when evidence is incomplete, indirect, stale, or uncertain."]
            ),
            "",
            "## Expected Output Shape",
            "",
            f"- Synthesis profile: `{synthesis.profile}`",
        ]
    )
    if synthesis.required_sections:
        lines.extend([f"- Required section: {section}" for section in synthesis.required_sections])
    lines.extend(
        [
            f"- Include tradeoffs: `{synthesis.include_tradeoffs}`",
            f"- Include uncertainties: `{synthesis.include_uncertainties}`",
            f"- Include next steps: `{synthesis.include_next_steps}`",
            "",
            "## Review Conditions",
            "",
        ]
    )
    review_conditions = (
        protocol.operator_review_required_when
        + safety.operator_review_required_when
        + [rule.required_action for rule in protocol.rules if rule.severity in {"high", "critical"}]
    )
    for pack in packs:
        review_conditions.extend(
            warning.message for warning in pack.warnings if warning.review_recommended
        )
        review_conditions.extend(
            rule.required_action for rule in pack.rules if rule.severity in {"high", "critical"}
        )
    if safety.professional_advice_disclaimer:
        review_conditions.append("Do not present the output as professional advice.")
    lines.extend(_bullet(review_conditions) or ["- Standard operator review if evidence is weak."])

    if safety.safety_warnings:
        lines.extend(["", "## Safety Warnings", ""])
        lines.extend(_bullet(safety.safety_warnings))

    if packs:
        lines.extend(["", "## Policy Packs", ""])
        for pack in packs:
            lines.append(f"- `{pack.pack_id}`: {pack.name} ({pack.version})")

    return "\n".join(lines).rstrip() + "\n"


def build_selection_instruction_block(selection: ProtocolSelection) -> str:
    return build_protocol_instruction_block(
        protocol=selection.selected_protocol,
        profile=selection.intelligence_profile,
        policy_packs=selection.policy_packs,
    )
