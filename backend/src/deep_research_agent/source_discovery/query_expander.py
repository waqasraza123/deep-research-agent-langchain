from __future__ import annotations

import re

from .contracts import SearchQuery, SearchQueryPlan, SourceType

STOP_ENTITIES = {
    "Compare",
    "What",
    "How",
    "Why",
    "When",
    "Where",
    "Which",
    "Production",
    "Backend",
    "Agent",
    "Research",
}

TECH_TERMS = {
    "api",
    "backend",
    "benchmark",
    "database",
    "deployment",
    "framework",
    "github",
    "langchain",
    "langgraph",
    "library",
    "memory",
    "orchestration",
    "python",
    "sdk",
    "server",
    "tool",
}

LEGAL_TERMS = {
    "compliance",
    "law",
    "legal",
    "liability",
    "policy",
    "regulation",
    "regulatory",
    "security rule",
    "standard",
}

ACADEMIC_TERMS = {
    "academic",
    "evaluation",
    "literature",
    "paper",
    "research",
    "study",
    "systematic review",
}

FRESHNESS_TERMS = {
    "2024",
    "2025",
    "2026",
    "current",
    "latest",
    "new",
    "news",
    "recent",
    "release",
    "today",
    "updated",
}

IMPLEMENTATION_TERMS = {
    "api",
    "architecture",
    "backend",
    "deploy",
    "implementation",
    "integrate",
    "production",
    "sdk",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_.+-]*", text.lower())


def _has_any(text: str, terms: set[str]) -> bool:
    low = text.lower()
    return any(term in low for term in terms)


def _extract_entities(question: str) -> list[str]:
    entities: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_.+-]*(?:\s+[A-Z][A-Za-z0-9_.+-]*){0,3}", question):
        entity = " ".join(match.group(0).split())
        entity = re.sub(r"^(Compare|What|How|Why|When|Where|Which)\s+", "", entity).strip()
        if entity in STOP_ENTITIES or len(entity) < 2:
            continue
        if entity not in entities:
            entities.append(entity)
    for token in re.findall(r"\b[A-Za-z]+[A-Z][A-Za-z0-9]*\b", question):
        if token not in entities and not any(token in entity.split() for entity in entities):
            entities.append(token)
    return entities[:8]


def _entity_phrase(question: str, entities: list[str]) -> str:
    if entities:
        return " ".join(entities[:4])
    words = [t for t in _tokens(question) if len(t) > 2]
    return " ".join(words[:8])


def _comparative(question: str) -> bool:
    low = question.lower()
    return bool(
        re.search(r"\b(compare|comparison|versus|vs\.?|tradeoff|which is better)\b", low)
        or re.search(r"\b\w+\s+and\s+\w+\b", low)
    )


def _freshness_required(question: str) -> bool:
    return _has_any(question, FRESHNESS_TERMS) or bool(
        re.search(
            r"\b(price|pricing|law|regulation|release|changelog|version|market)\b", question.lower()
        )
    )


def expand_queries(
    question: str, *, max_queries: int = 8, freshness_required: bool | None = None
) -> SearchQueryPlan:
    q = " ".join(question.strip().split())
    entities = _extract_entities(q)
    phrase = _entity_phrase(q, entities)
    comparative = _comparative(q)
    technical = _has_any(q, TECH_TERMS) or any(
        e.lower() in {"langgraph", "crewai", "fastapi", "langchain"} for e in entities
    )
    legal = _has_any(q, LEGAL_TERMS)
    academic = _has_any(q, ACADEMIC_TERMS)
    implementation = _has_any(q, IMPLEMENTATION_TERMS)
    fresh = _freshness_required(q) if freshness_required is None else bool(freshness_required)

    specs: list[tuple[str, str, str, list[SourceType], bool]] = []
    if comparative:
        specs.append(
            (
                "comparison",
                f"{phrase} comparison production agent orchestration",
                "Compare entities on the requested decision surface.",
                ["official_docs", "benchmark_report", "source_code_repository"],
                fresh,
            )
        )
    else:
        specs.append(
            (
                "broad_overview",
                f"{phrase} overview key concepts",
                "Establish the baseline terminology and scope.",
                ["official_docs", "tutorial_or_blog"],
                fresh,
            )
        )

    if len(entities) >= 1:
        for entity in entities[:2]:
            topic_tail = (
                "agents tools memory"
                if entity.lower() in {"crewai", "crew ai"}
                else "persistence checkpointing tools"
            )
            specs.append(
                (
                    "official_documentation",
                    f"{entity} official docs {topic_tail}",
                    "Prefer documentation controlled by the project or vendor.",
                    ["official_docs"],
                    fresh,
                )
            )

    specs.append(
        (
            "primary_source",
            f"{phrase} GitHub repository issues release notes",
            "Find primary project artifacts such as source repositories and release notes.",
            ["source_code_repository", "release_notes"],
            fresh,
        )
    )

    if fresh:
        specs.append(
            (
                "recent_current",
                f"{phrase} latest release current status 2026",
                "Freshness appears material to the question.",
                ["release_notes", "company_announcement", "official_docs"],
                True,
            )
        )

    if comparative:
        specs.append(
            (
                "comparison",
                f"{phrase} vs reliability deployment",
                "Look for direct comparison on production reliability and deployment.",
                ["benchmark_report", "tutorial_or_blog", "forum_discussion"],
                fresh,
            )
        )

    specs.append(
        (
            "risk_failure_mode",
            f"{phrase} failure modes production",
            "Surface operational risks, limitations, and failure reports.",
            ["source_code_repository", "forum_discussion", "tutorial_or_blog"],
            fresh,
        )
    )

    if technical:
        specs.append(
            (
                "benchmark_evaluation",
                f"{phrase} benchmark evaluation performance",
                "Technical questions benefit from evaluation or benchmark evidence.",
                ["benchmark_report", "academic_paper", "source_code_repository"],
                fresh,
            )
        )

    if legal:
        specs.append(
            (
                "regulatory_legal",
                f"{phrase} regulation policy compliance legal requirements",
                "Legal or policy claims need authoritative public sources.",
                ["government_or_policy", "legal_or_regulatory"],
                True,
            )
        )

    if academic:
        specs.append(
            (
                "academic_literature",
                f"{phrase} academic paper literature review",
                "Research-style questions need literature search terms.",
                ["academic_paper"],
                fresh,
            )
        )

    if implementation or technical:
        specs.append(
            (
                "implementation",
                f"{phrase} implementation architecture examples",
                "Implementation details need technical examples and source material.",
                ["official_docs", "source_code_repository", "tutorial_or_blog"],
                fresh,
            )
        )

    seen: set[str] = set()
    queries: list[SearchQuery] = []
    for intent, text, rationale, source_types, query_freshness in specs:
        normalized = re.sub(r"\s+", " ", text).strip()
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(
            SearchQuery(
                query_id=f"q{len(queries) + 1}",
                text=normalized,
                intent=intent,  # type: ignore[arg-type]
                target_source_types=source_types,
                freshness_required=query_freshness,
                rationale=rationale,
            )
        )
        if len(queries) >= max(0, max_queries):
            break

    warnings: list[str] = []
    if not queries:
        warnings.append("No search queries were generated because max_queries is zero.")

    return SearchQueryPlan(
        question=q,
        queries=queries,
        freshness_required=fresh,
        comparative=comparative,
        technical=technical,
        legal_or_policy=legal,
        academic=academic,
        entities=entities,
        warnings=warnings,
    )
