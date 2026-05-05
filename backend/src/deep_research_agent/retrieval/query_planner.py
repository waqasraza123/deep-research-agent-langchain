from __future__ import annotations

import hashlib
import re
from typing import Iterable

from .contracts import RetrievalQuery
from .indexer import extract_dates, extract_entities, extract_numbers
from .lexical import extract_quoted_phrases, tokenize

COMPARISON_TERMS = {
    "compare",
    "versus",
    "vs",
    "difference",
    "differences",
    "alternative",
    "alternatives",
    "better",
    "tradeoff",
    "tradeoffs",
}
RISK_TERMS = [
    "risk",
    "failure",
    "limitation",
    "drawback",
    "caveat",
    "pitfall",
    "concern",
    "security",
    "compliance",
]
FRESHNESS_TERMS = {
    "latest",
    "current",
    "recent",
    "today",
    "this year",
    "2024",
    "2025",
    "2026",
}


def _query_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def _dedupe(queries: Iterable[RetrievalQuery]) -> list[RetrievalQuery]:
    out: list[RetrievalQuery] = []
    seen: set[str] = set()
    for query in queries:
        key = " ".join(query.text.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(query)
    return out


def _split_subquestions(text: str) -> list[str]:
    parts = re.split(r"\n+|(?:^|\s)(?:\d+[\).]|[-*])\s+", text or "")
    out = []
    for part in parts:
        clean = " ".join(part.split()).strip(" -")
        if len(clean) >= 12:
            out.append(clean)
    return out


def plan_retrieval_queries(
    question: str,
    *,
    subquestions: list[str] | None = None,
    max_queries: int = 16,
) -> list[RetrievalQuery]:
    question = " ".join((question or "").split())
    entities = extract_entities(question)
    dates = extract_dates(question)
    numbers = extract_numbers(question)
    phrases = extract_quoted_phrases(question)
    terms = set(tokenize(question))
    freshness_required = any(term in question.lower() for term in FRESHNESS_TERMS)
    primary_preferred = any(term in terms for term in {"official", "primary", "filing", "report"})

    queries: list[RetrievalQuery] = [
        RetrievalQuery(
            query_id=_query_id("main", question),
            text=question,
            query_type="main",
            entities=entities,
            dates=dates,
            numbers=numbers,
            phrases=phrases,
            freshness_required=freshness_required,
            primary_source_preferred=primary_preferred,
        )
    ]

    for idx, subquestion in enumerate(subquestions or [], start=1):
        clean = " ".join(subquestion.split())
        if not clean:
            continue
        queries.append(
            RetrievalQuery(
                query_id=_query_id(f"sub{idx}", clean),
                text=clean,
                query_type="subquestion",
                source="subquestion",
                entities=extract_entities(clean),
                dates=extract_dates(clean),
                numbers=extract_numbers(clean),
                phrases=extract_quoted_phrases(clean),
                freshness_required=freshness_required
                or any(term in clean.lower() for term in FRESHNESS_TERMS),
            )
        )

    for idx, entity in enumerate(entities[:6], start=1):
        queries.append(
            RetrievalQuery(
                query_id=_query_id(f"entity{idx}", entity),
                text=f"{entity} {question}",
                query_type="entity",
                source="named_entity",
                entities=[entity],
                freshness_required=freshness_required,
            )
        )

    if terms & COMPARISON_TERMS or re.search(r"\bvs\.?\b| versus ", question.lower()):
        queries.append(
            RetrievalQuery(
                query_id=_query_id("cmp", question),
                text=f"{question} comparison differences tradeoffs advantages disadvantages",
                query_type="comparison",
                source="comparison_dimensions",
                entities=entities,
                freshness_required=freshness_required,
                metadata={"include_opposing_views": True},
            )
        )

    queries.append(
        RetrievalQuery(
            query_id=_query_id("risk", question),
            text=f"{question} {' '.join(RISK_TERMS)}",
            query_type="risk",
            source="risk_failure_terms",
            entities=entities[:4],
            freshness_required=freshness_required,
        )
    )

    if freshness_required or dates:
        queries.append(
            RetrievalQuery(
                query_id=_query_id("fresh", question),
                text=f"{question} date updated published latest current version",
                query_type="freshness",
                source="freshness_terms",
                entities=entities[:4],
                dates=dates,
                freshness_required=True,
            )
        )

    queries.append(
        RetrievalQuery(
            query_id=_query_id("cite", question),
            text=f"{question} source citation evidence quote official documentation report",
            query_type="citation_verification",
            source="citation_verification_terms",
            entities=entities[:4],
            dates=dates,
            numbers=numbers,
            primary_source_preferred=True,
        )
    )

    if not subquestions:
        for idx, inferred in enumerate(_split_subquestions(question), start=1):
            if inferred == question:
                continue
            queries.append(
                RetrievalQuery(
                    query_id=_query_id(f"inferred{idx}", inferred),
                    text=inferred,
                    query_type="subquestion",
                    source="inferred",
                    entities=extract_entities(inferred),
                    dates=extract_dates(inferred),
                    numbers=extract_numbers(inferred),
                )
            )

    return _dedupe(queries)[:max_queries]
