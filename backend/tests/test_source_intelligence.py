from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_research_agent.source_intelligence import crawl_sources
from deep_research_agent.source_intelligence.contracts import LinkCandidate
from deep_research_agent.source_intelligence.dedupe import DedupeIndex, normalize_url
from deep_research_agent.source_intelligence.link_extractor import normalize_and_validate_link
from deep_research_agent.source_intelligence.prioritizer import prioritize_links
from deep_research_agent.tools import FetchResult


def _text(label: str) -> str:
    return (f"{label} research documentation source reference citation 2025. " * 80).strip()


def _fetch_result(
    url: str,
    *,
    title: str = "Research Docs",
    text: str | None = None,
    links: list[dict[str, str]] | None = None,
    kind: str = "html",
    canonical_url: str | None = None,
) -> FetchResult:
    body = text if text is not None else _text(url)
    return FetchResult(
        ok=True,
        url=url,
        final_url=url,
        status_code=200,
        content_type="text/html" if kind == "html" else "text/markdown",
        extracted_text=body,
        title=title,
        truncated=False,
        strategy="direct",
        word_count=len(body.split()),
        char_count=len(body),
        kind=kind,
        canonical_url=canonical_url,
        extracted_links=tuple(links or []),
    )


def _fake_fetcher(mapping: dict[str, FetchResult], calls: list[str]):
    def fetcher(url: str, **_kwargs: Any) -> FetchResult:
        calls.append(url)
        return mapping[url]

    return fetcher


def test_link_normalization_removes_fragments_and_tracking_params():
    normalized, reason = normalize_and_validate_link(
        "/docs/guide/?utm_source=x&b=2&a=1#section",
        "https://Example.com/root",
    )

    assert reason is None
    assert normalized == "https://example.com/docs/guide?a=1&b=2"


def test_unsafe_link_rejection_blocks_local_hosts():
    normalized, reason = normalize_and_validate_link(
        "http://127.0.0.1/admin",
        "https://example.com/root",
    )

    assert normalized is None
    assert reason is not None
    assert reason.startswith("unsafe_url:")


def test_noisy_link_filtering_marks_login_and_social_links():
    links = [
        LinkCandidate(
            url="https://example.com/login",
            normalized_url="https://example.com/login",
            parent_url="https://example.com/root",
            anchor_text="Sign in",
        ),
        LinkCandidate(
            url="https://twitter.com/example",
            normalized_url="https://twitter.com/example",
            parent_url="https://example.com/root",
            anchor_text="Twitter",
        ),
    ]

    prioritized = prioritize_links(links, root_url="https://example.com/root", question="docs")

    assert [item.skip_reason for item in prioritized] == ["noise_login", "noise_social_link"]


def test_prioritization_prefers_same_domain_relevant_documentation():
    candidates = [
        LinkCandidate(
            url="https://external.test/blog",
            normalized_url="https://external.test/blog",
            parent_url="https://example.com/root",
            anchor_text="general blog",
        ),
        LinkCandidate(
            url="https://example.com/docs/research-agent-reference",
            normalized_url="https://example.com/docs/research-agent-reference",
            parent_url="https://example.com/root",
            anchor_text="Research agent reference",
        ),
        LinkCandidate(
            url="https://example.com/pricing",
            normalized_url="https://example.com/pricing",
            parent_url="https://example.com/root",
            anchor_text="Pricing",
        ),
    ]

    prioritized = prioritize_links(
        candidates,
        root_url="https://example.com/root",
        question="research agent documentation",
    )

    assert prioritized[0].candidate.normalized_url.endswith("/docs/research-agent-reference")
    assert prioritized[-1].skip_reason == "noise_pricing"


def test_crawler_enforces_max_links_per_source(tmp_path: Path):
    calls: list[str] = []
    root = "https://example.com/root"
    mapping = {
        root: _fetch_result(
            root,
            links=[
                {"url": "https://example.com/docs/research", "anchor_text": "Research docs"},
                {"url": "https://example.com/reference", "anchor_text": "Reference"},
                {"url": "https://example.com/blog", "anchor_text": "Blog"},
            ],
        ),
        "https://example.com/docs/research": _fetch_result(
            "https://example.com/docs/research",
            text=_text("docs research"),
        ),
    }

    result = crawl_sources(
        question="research documentation",
        root_urls=[root],
        thread_id="t1",
        thread_dir=tmp_path,
        timeout_s=1,
        max_chars=50_000,
        follow_links=True,
        max_links_per_source=1,
        fetcher=_fake_fetcher(mapping, calls),
    )

    assert calls == [root, "https://example.com/docs/research"]
    assert result.budget.global_link_budget_used == 1
    assert any(source.skip_reason == "max_links_per_source" for source in result.sources)


def test_crawler_enforces_global_budget(tmp_path: Path):
    calls: list[str] = []
    root1 = "https://example.com/root-1"
    root2 = "https://second.example.com/root-2"
    mapping = {
        root1: _fetch_result(
            root1,
            links=[{"url": "https://example.com/docs/a", "anchor_text": "Docs A"}],
        ),
        root2: _fetch_result(
            root2,
            links=[{"url": "https://second.example.com/docs/b", "anchor_text": "Docs B"}],
        ),
        "https://example.com/docs/a": _fetch_result("https://example.com/docs/a", text=_text("a")),
    }

    result = crawl_sources(
        question="docs",
        root_urls=[root1, root2],
        thread_id="t2",
        thread_dir=tmp_path,
        timeout_s=1,
        max_chars=50_000,
        follow_links=True,
        max_links_per_source=2,
        global_link_budget=1,
        fetcher=_fake_fetcher(mapping, calls),
    )

    assert calls == [root1, root2, "https://example.com/docs/a"]
    assert result.budget.global_link_budget_used == 1
    assert any(source.skip_reason == "global_crawl_budget" for source in result.sources)


def test_dedupe_detects_normalized_canonical_content_and_title_domain_duplicates():
    index = DedupeIndex()
    url = "https://example.com/docs?a=1&utm_source=x"
    normalized = normalize_url(url)
    index.register(
        "S1",
        url=url,
        normalized_url=normalized,
        canonical_url="https://example.com/canonical",
        title="Same Title",
        text="same body",
    )

    assert index.precheck_url(url, normalized).reason == "duplicate_exact_url"
    assert index.precheck_url("https://example.com/docs?a=1", normalized).reason == (
        "duplicate_normalized_url"
    )
    assert (
        index.check_fetched(
            url="https://example.com/other",
            normalized_url="https://example.com/other",
            canonical_url="https://example.com/canonical",
            title="Other",
            text="different",
        ).reason
        == "duplicate_canonical_url"
    )
    assert (
        index.check_fetched(
            url="https://example.com/body-copy",
            normalized_url="https://example.com/body-copy",
            canonical_url=None,
            title="Other",
            text="same body",
        ).reason
        == "duplicate_content_hash"
    )
    assert (
        index.check_fetched(
            url="https://example.com/title-copy",
            normalized_url="https://example.com/title-copy",
            canonical_url=None,
            title="Same Title",
            text="different title body",
        ).reason
        == "duplicate_title_domain"
    )


def test_source_graph_artifacts_include_skipped_unsafe_links(tmp_path: Path):
    calls: list[str] = []
    root = "https://example.com/root"
    mapping = {
        root: _fetch_result(
            root,
            links=[
                {"url": "http://127.0.0.1/admin", "anchor_text": "internal"},
                {"url": "https://example.com/docs", "anchor_text": "Docs"},
            ],
        ),
        "https://example.com/docs": _fetch_result("https://example.com/docs", text=_text("docs")),
    }

    result = crawl_sources(
        question="docs",
        root_urls=[root],
        thread_id="graph",
        thread_dir=tmp_path,
        timeout_s=1,
        max_chars=50_000,
        follow_links=True,
        max_links_per_source=2,
        fetcher=_fake_fetcher(mapping, calls),
    )

    graph = json.loads((tmp_path / "source_graph.json").read_text(encoding="utf-8"))
    assert (tmp_path / "source_graph.md").exists()
    assert graph["root_urls"] == [root]
    assert graph["budget"]["fetched_count"] == 2
    assert any("unsafe_url" in item["skip_reason"] for item in graph["skipped_links"])
    assert any(
        source.parent_url == root
        for source in result.sources
        if source.source_kind == "discovered"
    )


def test_follow_links_false_fetches_only_roots(tmp_path: Path):
    calls: list[str] = []
    root = "https://example.com/root"
    mapping = {
        root: _fetch_result(
            root,
            links=[{"url": "https://example.com/docs", "anchor_text": "Docs"}],
        )
    }

    result = crawl_sources(
        question="docs",
        root_urls=[root],
        thread_id="no-follow",
        thread_dir=tmp_path,
        timeout_s=1,
        max_chars=50_000,
        follow_links=False,
        max_links_per_source=10,
        fetcher=_fake_fetcher(mapping, calls),
    )

    assert calls == [root]
    assert result.discovered_links == []
    assert [source.source_kind for source in result.sources] == ["root"]
