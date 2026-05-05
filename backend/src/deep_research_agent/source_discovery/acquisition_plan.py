from __future__ import annotations

from .contracts import (
    SearchProviderConfig,
    SourceAcquisitionPlan,
    SourceDiscoveryRequest,
    SourceDiscoverySettings,
)
from .query_expander import expand_queries
from .source_types import plan_source_types


def provider_config_from_settings(settings: SourceDiscoverySettings) -> SearchProviderConfig:
    enabled = bool(settings.discovery_enabled and settings.provider != "disabled")
    reason = None
    if not settings.discovery_enabled:
        reason = "Source discovery is disabled by settings."
    elif settings.provider == "disabled":
        reason = "No source discovery provider is configured."
    return SearchProviderConfig(
        provider=settings.provider,
        enabled=enabled,
        reason=reason,
        static_results=settings.static_results,
    )


def build_acquisition_plan(request: SourceDiscoveryRequest) -> SourceAcquisitionPlan:
    settings = request.settings
    query_plan = expand_queries(
        request.question,
        max_queries=settings.max_queries,
        freshness_required=settings.freshness_required,
    )
    required, preferred, optional, rationale = plan_source_types(request.question, query_plan)
    provider_config = provider_config_from_settings(settings)
    warnings = list(query_plan.warnings)
    if not settings.discovery_enabled:
        warnings.append("Discovery disabled; only user-provided URLs will be used.")
    elif not provider_config.enabled:
        warnings.append(provider_config.reason or "Search provider unavailable.")
    return SourceAcquisitionPlan(
        question=request.question.strip(),
        required_source_types=required,
        preferred_source_types=preferred,
        optional_source_types=optional,
        query_plan=query_plan,
        provider_config=provider_config,
        user_urls=[u.strip() for u in request.user_urls if u and u.strip()],
        settings=settings,
        rationale=rationale,
        warnings=warnings,
    )
