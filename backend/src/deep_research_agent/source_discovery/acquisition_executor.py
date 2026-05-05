from __future__ import annotations

from .acquisition_plan import build_acquisition_plan
from .candidate_ranker import rank_and_select
from .contracts import (
    SourceCandidate,
    SourceDiscoveryBatch,
    SourceDiscoveryRequest,
    SourceDiscoverySummary,
)
from .search_client import SearchClient


def execute_source_discovery(request: SourceDiscoveryRequest) -> SourceDiscoveryBatch:
    plan = build_acquisition_plan(request)
    provider_results = []
    candidates: list[SourceCandidate] = []
    selected: list[SourceCandidate] = []
    decisions = []
    skipped_reason = None

    if not plan.settings.discovery_enabled:
        skipped_reason = "Source discovery is disabled by settings."
    elif not plan.provider_config.enabled:
        skipped_reason = plan.provider_config.reason or "Source discovery provider is unavailable."
    elif not plan.query_plan.queries:
        skipped_reason = "No search queries were generated."
    else:
        client = SearchClient(plan.provider_config)
        provider_results = client.search_many(
            plan.query_plan.queries,
            max_candidates_per_query=plan.settings.max_candidates_per_query,
        )
        for result in provider_results:
            candidates.extend(result.candidates)
        candidates, decisions = rank_and_select(candidates, plan)
        selected_ids = {d.candidate_id for d in decisions if d.decision == "selected"}
        selected = [c for c in candidates if c.candidate_id in selected_ids]

    warnings = list(plan.warnings)
    for result in provider_results:
        warnings.extend(result.warnings)
    if skipped_reason:
        warnings.append(skipped_reason)

    summary = SourceDiscoverySummary(
        discovery_enabled=plan.settings.discovery_enabled,
        provider=plan.provider_config.provider,
        question=plan.question,
        query_count=len(plan.query_plan.queries),
        provider_result_count=len(provider_results),
        candidate_count=len(candidates),
        selected_count=len(selected),
        selected_urls=[c.url for c in selected],
        skipped_reason=skipped_reason,
        required_source_types=plan.required_source_types,
        coverage_notes=_coverage_notes(plan.required_source_types, selected),
        warnings=warnings,
    )
    return SourceDiscoveryBatch(
        request=request,
        plan=plan,
        provider_results=provider_results,
        candidates=candidates,
        decisions=decisions,
        selected_candidates=selected,
        summary=summary,
    )


def _coverage_notes(required_source_types: list[str], selected: list[SourceCandidate]) -> list[str]:
    selected_types = {c.source_type_hint for c in selected}
    notes: list[str] = []
    for source_type in required_source_types:
        if source_type in selected_types:
            notes.append(f"Covered required source type `{source_type}`.")
        else:
            notes.append(f"No selected candidate covered required source type `{source_type}`.")
    return notes
