from __future__ import annotations

import re
from collections import Counter

from .contracts import ExtractedTopic
from .repository import normalize_question

STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "and",
    "are",
    "because",
    "before",
    "between",
    "both",
    "can",
    "compare",
    "could",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "its",
    "latest",
    "more",
    "most",
    "not",
    "over",
    "research",
    "should",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "through",
    "under",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


DOMAIN_TOPIC_HINTS = {
    "privacy": ("privacy", "compliance", "policy"),
    "gdpr": ("gdpr", "compliance", "policy"),
    "hipaa": ("hipaa", "compliance", "healthcare"),
    "langgraph": ("langgraph", "agent orchestration", "framework"),
    "langchain": ("langchain", "agent orchestration", "framework"),
    "fastapi": ("fastapi", "python backend", "framework"),
    "postgres": ("postgres", "database", "persistence"),
    "sqlite": ("sqlite", "database", "persistence"),
    "vector": ("vector search", "retrieval", "embeddings"),
    "retrieval": ("retrieval", "memory", "knowledge reuse"),
    "agent": ("agent systems", "orchestration", "automation"),
    "security": ("security", "risk", "controls"),
    "pricing": ("pricing", "cost", "commercial"),
}


def tokenize(text: str) -> list[str]:
    normalized = normalize_question(text)
    return [token for token in normalized.split() if len(token) >= 3 and token not in STOPWORDS]


def _ngrams(tokens: list[str], size: int) -> list[str]:
    return [" ".join(tokens[i : i + size]) for i in range(0, max(0, len(tokens) - size + 1))]


def extract_topics(
    *,
    question: str,
    title: str | None = None,
    text: str = "",
    max_topics: int = 12,
) -> list[ExtractedTopic]:
    source = " ".join([question or "", title or "", text[:6000]])
    tokens = tokenize(source)
    if not tokens:
        return []

    counts: Counter[str] = Counter()
    counts.update(tokens)
    for phrase in _ngrams(tokens, 2) + _ngrams(tokens, 3):
        if len(set(phrase.split())) > 1:
            counts[phrase] += 2 if " " in phrase else 1

    for token in set(tokens):
        for hint in DOMAIN_TOPIC_HINTS.get(token, ()):
            counts[hint] += 4

    scored: list[ExtractedTopic] = []
    max_count = max(counts.values()) if counts else 1
    for name, count in counts.most_common(max_topics * 4):
        if len(name) < 4:
            continue
        if re.fullmatch(r"\d+", name):
            continue
        keywords = [part for part in name.split() if part not in STOPWORDS]
        score = min(1.0, 0.25 + (count / max_count) * 0.75)
        scored.append(ExtractedTopic(name=name, score=round(score, 3), keywords=keywords[:6]))
        if len(scored) >= max_topics:
            break
    return scored


def topic_similarity(left: list[ExtractedTopic], right: list[ExtractedTopic]) -> float:
    left_names = {topic.name.lower() for topic in left}
    right_names = {topic.name.lower() for topic in right}
    if not left_names or not right_names:
        return 0.0
    return len(left_names & right_names) / len(left_names | right_names)
