from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from urllib.parse import urlsplit

from deep_research_agent.source_intelligence.dedupe import normalize_url

from .contracts import (
    SourceAcquisitionDecision,
    SourceAcquisitionPlan,
    SourceCandidate,
    SourceCandidateScore,
)

PROMOTIONAL_TERMS = {
    "best",
    "ultimate",
    "sponsored",
    "affiliate",
    "coupon",
    "top 10",
    "pricing",
    "buy",
}


def _domain(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _canonical_key(url: str) -> str:
    normalized = normalize_url(url)
    parts = urlsplit(normalized)
    path = re.sub(r"/(index|home)\.(html?|php)$", "/", parts.path or "/")
    return f"{parts.scheme}://{parts.netloc}{path}".rstrip("/")


def _similar_title(a: str, b: str) -> float:
    a_norm = " ".join(_tokens(a))
    b_norm = " ".join(_tokens(b))
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def dedupe_candidates(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    url_groups: dict[str, str] = {}
    domain_titles: list[tuple[str, str, str]] = []
    provider_query_seen: dict[tuple[str, str, str], str] = {}
    group_index = 1

    for candidate in candidates:
        keys = [
            normalize_url(candidate.url),
            _canonical_key(candidate.url),
        ]
        group_id = None
        for key in keys:
            if key in url_groups:
                group_id = url_groups[key]
                break
        if group_id is None:
            domain = candidate.domain or _domain(candidate.url)
            for existing_domain, existing_title, existing_group in domain_titles:
                if (
                    existing_domain == domain
                    and _similar_title(existing_title, candidate.title) >= 0.88
                ):
                    group_id = existing_group
                    candidate.warnings.append("Possible duplicate by domain and title similarity.")
                    break
        provider_key = (candidate.provider, candidate.query_id or "", normalize_url(candidate.url))
        if group_id is None and provider_key in provider_query_seen:
            group_id = provider_query_seen[provider_key]
            candidate.warnings.append(
                "Provider returned the same URL for this query more than once."
            )
        if group_id is None:
            group_id = f"dup-{group_index}"
            group_index += 1
        for key in keys:
            url_groups.setdefault(key, group_id)
        provider_query_seen.setdefault(provider_key, group_id)
        if candidate.title:
            domain_titles.append(
                (candidate.domain or _domain(candidate.url), candidate.title, group_id)
            )
        candidate.duplicate_group_id = group_id
    return candidates


def score_candidate(
    candidate: SourceCandidate, plan: SourceAcquisitionPlan
) -> SourceCandidateScore:
    question_terms = _tokens(plan.question)
    query_terms = _tokens(candidate.query)
    title_terms = _tokens(candidate.title)
    snippet_terms = _tokens(candidate.snippet)
    all_candidate_terms = title_terms | snippet_terms | _tokens(candidate.url)

    components: dict[str, float] = {}
    reasons: list[str] = []
    penalties: list[str] = []
    warnings: list[str] = list(candidate.warnings)

    if candidate.source_type_hint in plan.required_source_types:
        components["source_type_match"] = 1.0
        reasons.append(f"Matches required source type `{candidate.source_type_hint}`.")
    elif candidate.source_type_hint in plan.preferred_source_types:
        components["source_type_match"] = 0.75
        reasons.append(f"Matches preferred source type `{candidate.source_type_hint}`.")
    elif candidate.source_type_hint in plan.optional_source_types:
        components["source_type_match"] = 0.45
    else:
        components["source_type_match"] = 0.25

    entity_hits = question_terms & all_candidate_terms
    components["exact_entity_match"] = min(
        1.0, len(entity_hits) / max(1, min(len(question_terms), 6))
    )
    if entity_hits:
        reasons.append(
            "Title/snippet/url overlap with question terms: "
            + ", ".join(sorted(entity_hits)[:6])
            + "."
        )

    components["authority"] = candidate.authority_hint
    if candidate.authority_hint >= 0.75:
        reasons.append("Strong authority hint from domain or source type.")

    components["primary_source"] = candidate.primary_source_likelihood
    if candidate.primary_source_likelihood >= 0.7:
        reasons.append("Likely primary source.")

    freshness_score = 0.5
    if candidate.freshness_hint in {"current", "recent"}:
        freshness_score = 1.0
        reasons.append(f"Freshness hint is `{candidate.freshness_hint}`.")
    elif candidate.freshness_hint in {"stale", "possibly_stale"}:
        freshness_score = 0.2
        warnings.append("Candidate may be stale.")
    components["freshness"] = (
        freshness_score if plan.query_plan.freshness_required else max(0.45, freshness_score * 0.7)
    )

    query_overlap = query_terms & all_candidate_terms
    components["query_intent_match"] = min(
        1.0, len(query_overlap) / max(1, min(len(query_terms), 7))
    )
    if (
        candidate.query_intent == "official_documentation"
        and candidate.source_type_hint == "official_docs"
    ):
        components["query_intent_match"] = max(components["query_intent_match"], 0.9)
    if candidate.query_intent == "risk_failure_mode" and (
        "issue" in candidate.url.lower() or "failure" in candidate.snippet.lower()
    ):
        components["query_intent_match"] = max(components["query_intent_match"], 0.85)

    noise_penalty = 0.0
    haystack = f"{candidate.title} {candidate.snippet} {candidate.url}".lower()
    for term in PROMOTIONAL_TERMS:
        if term in haystack:
            noise_penalty += 0.06
    if noise_penalty:
        penalties.append("Promotional/noise wording penalty.")
    if candidate.source_type_hint == "forum_discussion" and not plan.settings.allow_forums:
        noise_penalty += 0.18
        penalties.append("Forum source is not allowed by settings.")
    if (
        candidate.source_type_hint
        not in {
            "official_docs",
            "source_code_repository",
            "release_notes",
            "government_or_policy",
            "legal_or_regulatory",
            "company_announcement",
            "academic_paper",
            "dataset",
        }
        and not plan.settings.allow_secondary_sources
    ):
        noise_penalty += 0.25
        penalties.append("Secondary sources are not allowed by settings.")

    weighted = (
        components["source_type_match"] * 0.22
        + components["exact_entity_match"] * 0.18
        + components["authority"] * 0.18
        + components["primary_source"] * 0.17
        + components["freshness"] * 0.12
        + components["query_intent_match"] * 0.13
    )
    total = max(0.0, min(1.0, weighted - noise_penalty))
    if math.isnan(total):
        total = 0.0
    return SourceCandidateScore(
        candidate_id=candidate.candidate_id,
        total_score=round(total, 4),
        components={k: round(v, 4) for k, v in components.items()},
        reasons=reasons,
        penalties=penalties,
        warnings=warnings,
    )


def rank_and_select(
    candidates: list[SourceCandidate],
    plan: SourceAcquisitionPlan,
) -> tuple[list[SourceCandidate], list[SourceAcquisitionDecision]]:
    candidates = dedupe_candidates(candidates)
    scored: list[SourceCandidate] = []
    for candidate in candidates:
        score = score_candidate(candidate, plan)
        candidate.ranking_score = score.total_score
        candidate.ranking_reasons = [*score.reasons, *score.penalties]
        candidate.warnings = score.warnings
        scored.append(candidate)

    sorted_candidates = sorted(
        scored,
        key=lambda c: (c.ranking_score, c.primary_source_likelihood, c.authority_hint, c.title),
        reverse=True,
    )

    selected: list[SourceCandidate] = []
    decisions: list[SourceAcquisitionDecision] = []
    seen_groups: set[str] = set()
    max_selected = plan.settings.max_selected_sources

    for candidate in sorted_candidates:
        reason = "Selected for acquisition."
        decision = "selected"
        rank = None
        group = candidate.duplicate_group_id
        if group and group in seen_groups:
            decision = "duplicate"
            reason = "Rejected as duplicate of a higher-ranked candidate."
        elif candidate.source_type_hint == "forum_discussion" and not plan.settings.allow_forums:
            decision = "skipped"
            reason = "Forum/discussion sources are disabled."
        elif (
            not plan.settings.allow_secondary_sources and candidate.primary_source_likelihood < 0.55
        ):
            decision = "skipped"
            reason = "Secondary sources are disabled."
        elif len(selected) >= max_selected:
            decision = "rejected"
            reason = "Selection limit reached."
        else:
            selected.append(candidate)
            if group:
                seen_groups.add(group)
            rank = len(selected)

        candidate.decision = decision  # type: ignore[assignment]
        decisions.append(
            SourceAcquisitionDecision(
                candidate_id=candidate.candidate_id,
                url=candidate.url,
                title=candidate.title,
                domain=candidate.domain,
                source_type_hint=candidate.source_type_hint,
                decision=decision,  # type: ignore[arg-type]
                score=candidate.ranking_score,
                rank=rank,
                reason=reason,
                duplicate_group_id=candidate.duplicate_group_id,
                warnings=candidate.warnings,
            )
        )

    if (
        plan.settings.require_primary_source_when_possible
        and selected
        and not any(c.primary_source_likelihood >= 0.7 for c in selected)
    ):
        primary = next(
            (
                c
                for c in sorted_candidates
                if c.primary_source_likelihood >= 0.7 and c.decision != "duplicate"
            ),
            None,
        )
        if primary and primary not in selected:
            replaced = selected[-1]
            replaced.decision = "rejected"
            primary.decision = "selected"
            selected[-1] = primary
            for idx, item in enumerate(decisions):
                if item.candidate_id == primary.candidate_id:
                    decisions[idx].decision = "selected"
                    decisions[idx].rank = len(selected)
                    decisions[idx].reason = "Selected to satisfy primary-source preference."
                elif item.candidate_id == replaced.candidate_id:
                    decisions[idx].decision = "rejected"
                    decisions[idx].rank = None
                    decisions[idx].reason = "Replaced by an available primary source."

    return sorted_candidates, decisions
