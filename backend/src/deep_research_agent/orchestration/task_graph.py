from __future__ import annotations

from typing import Any

from deep_research_agent.intelligence import ResearchStrategy

from .contracts import ResearchTaskGraph
from .router import AdaptiveRouter, RoutingContext


def _strategy_payload(strategy: ResearchStrategy | dict[str, Any] | None) -> dict[str, Any] | None:
    if strategy is None:
        return None
    if isinstance(strategy, dict):
        return strategy
    if hasattr(strategy, "model_dump"):
        return strategy.model_dump(mode="json")
    return strategy.dict()


def build_research_task_graph(
    *,
    thread_id: str,
    question: str,
    urls: list[str] | None = None,
    strategy: ResearchStrategy | dict[str, Any] | None = None,
    follow_links: bool = False,
    max_links_per_source: int = 0,
    available_source_count: int = 0,
    router: AdaptiveRouter | None = None,
) -> ResearchTaskGraph:
    """Create a deterministic task graph for a research run."""
    return (router or AdaptiveRouter()).route(
        RoutingContext(
            thread_id=thread_id,
            question=question,
            urls=urls or [],
            follow_links=follow_links,
            max_links_per_source=max_links_per_source,
            available_source_count=available_source_count,
            strategy=_strategy_payload(strategy),
        )
    )
