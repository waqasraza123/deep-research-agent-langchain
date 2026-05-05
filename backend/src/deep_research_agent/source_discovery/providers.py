from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlsplit

from .contracts import (
    SearchProviderConfig,
    SearchProviderResult,
    SearchQuery,
    SourceCandidate,
    SourceType,
)


def _domain(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def _candidate_id(provider: str, query_id: str, url: str) -> str:
    digest = hashlib.sha1(f"{provider}|{query_id}|{url}".encode("utf-8")).hexdigest()[:12]
    return f"{provider}-{query_id}-{digest}"


def _infer_source_type(url: str, title: str, snippet: str = "") -> SourceType:
    text = f"{url} {title} {snippet}".lower()
    domain = _domain(url)
    if "github.com" in domain or "gitlab.com" in domain:
        return "source_code_repository"
    if domain.endswith(".gov") or "europa.eu" in domain:
        return "government_or_policy"
    if "law" in domain or "regulation" in text or "compliance" in text:
        return "legal_or_regulatory"
    if "docs." in domain or "/docs" in text or "documentation" in text:
        return "official_docs"
    if "release" in text or "changelog" in text or "releases" in text:
        return "release_notes"
    if domain.endswith(".edu") or "arxiv.org" in domain or "doi.org" in domain:
        return "academic_paper"
    if "benchmark" in text or "evaluation" in text:
        return "benchmark_report"
    if "forum" in domain or "reddit.com" in domain or "stackoverflow.com" in domain:
        return "forum_discussion"
    if "dataset" in text or "kaggle.com" in domain or "huggingface.co/datasets" in text:
        return "dataset"
    if "blog" in domain or "/blog" in text or "tutorial" in text:
        return "tutorial_or_blog"
    if "announcement" in text or "press" in text:
        return "company_announcement"
    return "unknown"


def _primary_likelihood(source_type: SourceType, url: str) -> float:
    domain = _domain(url)
    if source_type in {
        "official_docs",
        "source_code_repository",
        "release_notes",
        "government_or_policy",
        "legal_or_regulatory",
        "company_announcement",
        "dataset",
    }:
        return 0.8
    if domain.endswith(".edu") or source_type == "academic_paper":
        return 0.65
    if source_type == "forum_discussion":
        return 0.15
    return 0.35


def _authority(source_type: SourceType, url: str) -> float:
    domain = _domain(url)
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.9
    if source_type in {"official_docs", "source_code_repository", "release_notes"}:
        return 0.82
    if "github.com" in domain:
        return 0.75
    if source_type == "forum_discussion":
        return 0.25
    if source_type == "tutorial_or_blog":
        return 0.45
    return 0.5


class SearchProvider(ABC):
    name: str

    @abstractmethod
    def search(self, query: SearchQuery, *, max_results: int) -> SearchProviderResult:
        raise NotImplementedError


class DisabledSearchProvider(SearchProvider):
    name = "disabled"

    def __init__(self, reason: str = "Live search provider is disabled or not configured.") -> None:
        self.reason = reason

    def search(self, query: SearchQuery, *, max_results: int) -> SearchProviderResult:
        return SearchProviderResult(
            provider=self.name,
            query_id=query.query_id,
            query=query.text,
            ok=False,
            disabled_reason=self.reason,
            warnings=[self.reason],
        )


class StaticSearchProvider(SearchProvider):
    name = "static"

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results

    def search(self, query: SearchQuery, *, max_results: int) -> SearchProviderResult:
        candidates: list[SourceCandidate] = []
        q_terms = {t for t in re.findall(r"[a-z0-9]+", query.text.lower()) if len(t) > 2}
        for item in self.results:
            title = str(item.get("title") or "")
            snippet = str(item.get("snippet") or "")
            url = str(item.get("url") or "")
            if not url:
                continue
            haystack = f"{title} {snippet} {url}".lower()
            if q_terms and not any(term in haystack for term in q_terms):
                explicit_query = str(item.get("query") or "").lower()
                if explicit_query and explicit_query not in query.text.lower():
                    continue
            source_type = item.get("source_type_hint") or _infer_source_type(url, title, snippet)
            candidate = SourceCandidate(
                candidate_id=str(
                    item.get("candidate_id") or _candidate_id(self.name, query.query_id, url)
                ),
                url=url,
                title=title,
                snippet=snippet,
                domain=str(item.get("domain") or _domain(url)),
                provider=self.name,
                query=query.text,
                query_id=query.query_id,
                query_intent=query.intent,
                source_type_hint=source_type,
                primary_source_likelihood=float(
                    item.get("primary_source_likelihood") or _primary_likelihood(source_type, url)
                ),
                freshness_hint=str(item.get("freshness_hint") or "unknown"),
                authority_hint=float(item.get("authority_hint") or _authority(source_type, url)),
                metadata={"static": True, **dict(item.get("metadata") or {})},
            )
            candidates.append(candidate)
            if len(candidates) >= max_results:
                break
        return SearchProviderResult(
            provider=self.name,
            query_id=query.query_id,
            query=query.text,
            candidates=candidates,
        )


class MockSearchProvider(SearchProvider):
    name = "mock"

    def search(self, query: SearchQuery, *, max_results: int) -> SearchProviderResult:
        base = self._mock_items(query)
        candidates = []
        for item in base[: max(0, max_results)]:
            url = item["url"]
            source_type = item["source_type_hint"]
            candidates.append(
                SourceCandidate(
                    candidate_id=_candidate_id(self.name, query.query_id, url),
                    url=url,
                    title=item["title"],
                    snippet=item["snippet"],
                    domain=_domain(url),
                    provider=self.name,
                    query=query.text,
                    query_id=query.query_id,
                    query_intent=query.intent,
                    source_type_hint=source_type,
                    primary_source_likelihood=_primary_likelihood(source_type, url),
                    freshness_hint=item.get("freshness_hint", "unknown"),
                    authority_hint=_authority(source_type, url),
                    metadata={"mock": True},
                )
            )
        return SearchProviderResult(
            provider=self.name,
            query_id=query.query_id,
            query=query.text,
            candidates=candidates,
            metadata={"offline": True},
        )

    def _mock_items(self, query: SearchQuery) -> list[dict[str, Any]]:
        text = query.text.lower()
        items: list[dict[str, Any]] = []

        if "langgraph" in text:
            items.extend(
                [
                    {
                        "url": "https://langchain-ai.github.io/langgraph/",
                        "title": "LangGraph Documentation",
                        "snippet": (
                            "Official LangGraph docs covering persistence, "
                            "checkpointing, tools, and deployment."
                        ),
                        "source_type_hint": "official_docs",
                        "freshness_hint": "current",
                    },
                    {
                        "url": "https://github.com/langchain-ai/langgraph",
                        "title": "langchain-ai/langgraph",
                        "snippet": (
                            "Primary source repository for LangGraph issues, "
                            "releases, and implementation details."
                        ),
                        "source_type_hint": "source_code_repository",
                        "freshness_hint": "recent",
                    },
                ]
            )
        if "crewai" in text or "crew ai" in text:
            items.extend(
                [
                    {
                        "url": "https://docs.crewai.com/",
                        "title": "CrewAI Documentation",
                        "snippet": (
                            "Official CrewAI docs for agents, tools, memory, "
                            "flows, and deployment concepts."
                        ),
                        "source_type_hint": "official_docs",
                        "freshness_hint": "current",
                    },
                    {
                        "url": "https://github.com/crewAIInc/crewAI",
                        "title": "crewAIInc/crewAI",
                        "snippet": (
                            "Primary source repository for CrewAI issues, "
                            "releases, and implementation details."
                        ),
                        "source_type_hint": "source_code_repository",
                        "freshness_hint": "recent",
                    },
                ]
            )
        if query.intent == "risk_failure_mode":
            items.append(
                {
                    "url": "https://github.com/langchain-ai/langgraph/issues",
                    "title": "LangGraph GitHub Issues",
                    "snippet": (
                        "Issue tracker useful for production failure modes "
                        "and reliability concerns."
                    ),
                    "source_type_hint": "source_code_repository",
                    "freshness_hint": "recent",
                }
            )
        if query.intent == "benchmark_evaluation":
            items.append(
                {
                    "url": "https://arxiv.org/search/cs?query=agent+orchestration+evaluation",
                    "title": "Agent orchestration evaluation literature",
                    "snippet": (
                        "Academic search results for agent orchestration "
                        "evaluation and benchmarks."
                    ),
                    "source_type_hint": "academic_paper",
                    "freshness_hint": "unknown",
                }
            )
        if query.intent == "regulatory_legal":
            items.append(
                {
                    "url": "https://www.ftc.gov/business-guidance",
                    "title": "FTC Business Guidance",
                    "snippet": "Government guidance for business compliance and policy claims.",
                    "source_type_hint": "government_or_policy",
                    "freshness_hint": "current",
                }
            )

        if not items:
            slug = re.sub(r"[^a-z0-9]+", "-", query.text.lower()).strip("-")[:60] or "research"
            items.extend(
                [
                    {
                        "url": f"https://example.org/research/{slug}",
                        "title": f"Mock overview for {query.text}",
                        "snippet": (
                            "Offline mock search result for deterministic "
                            "development and tests."
                        ),
                        "source_type_hint": "tutorial_or_blog",
                        "freshness_hint": "unknown",
                    },
                    {
                        "url": f"https://docs.example.org/{slug}",
                        "title": f"Mock official documentation for {query.text}",
                        "snippet": "Offline mock official documentation result.",
                        "source_type_hint": "official_docs",
                        "freshness_hint": "unknown",
                    },
                ]
            )
        return items


def provider_from_config(config: SearchProviderConfig) -> SearchProvider:
    if not config.enabled:
        return DisabledSearchProvider(config.reason or "Source discovery provider is disabled.")
    if config.provider == "mock":
        return MockSearchProvider()
    if config.provider == "static":
        return StaticSearchProvider(config.static_results)
    return DisabledSearchProvider(
        config.reason or "No live source discovery provider is configured."
    )
