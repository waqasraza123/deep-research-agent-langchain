from __future__ import annotations

import re
from collections import Counter, defaultdict
from urllib.parse import urlsplit

from .contracts import EntityType, ExtractedEntity, ExtractionResult
from .topic_index import STOPWORDS, extract_topics

ORG_SUFFIXES = (
    "Inc",
    "LLC",
    "Ltd",
    "Corp",
    "Corporation",
    "Company",
    "University",
    "Institute",
    "Foundation",
    "Agency",
    "Commission",
    "Department",
    "Ministry",
    "Council",
    "Authority",
    "Association",
    "Group",
)

KNOWN_FRAMEWORKS = {
    "LangChain",
    "LangGraph",
    "Deep Agents",
    "FastAPI",
    "Pydantic",
    "Django",
    "Flask",
    "React",
    "Next.js",
    "Vue",
    "Svelte",
    "Postgres",
    "PostgreSQL",
    "SQLite",
    "Redis",
    "Qdrant",
    "OpenAI",
    "Ollama",
    "LlamaIndex",
    "PyTorch",
    "TensorFlow",
    "Kubernetes",
    "Docker",
}

KNOWN_PRODUCTS = {
    "ChatGPT",
    "Claude",
    "Gemini",
    "Copilot",
    "Supabase",
    "Vercel",
    "AWS",
    "Azure",
    "Google Cloud",
    "GitHub Actions",
}

KNOWN_LOCATIONS = {
    "United States",
    "United Kingdom",
    "European Union",
    "Europe",
    "California",
    "New York",
    "Washington",
    "Pakistan",
    "India",
    "China",
    "Japan",
    "Germany",
    "France",
    "Canada",
    "Australia",
}

TECH_TERMS = {
    "API",
    "SDK",
    "LLM",
    "RAG",
    "MCP",
    "ETL",
    "SQL",
    "HTTP",
    "OAuth",
    "OIDC",
    "embeddings",
    "vector search",
    "semantic search",
    "knowledge graph",
    "agent orchestration",
    "rate limiting",
    "serverless",
    "streaming",
}

LEGAL_POLICY_TERMS = {
    "GDPR",
    "HIPAA",
    "CCPA",
    "SOC 2",
    "terms of service",
    "privacy policy",
    "data retention",
    "data processing agreement",
    "regulation",
    "compliance",
    "copyright",
    "license",
}

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _snippet(text: str, start: int, end: int, *, window: int = 60) -> str:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


def _add(
    bucket: dict[tuple[EntityType, str], list[str]],
    entity_type: EntityType,
    name: str,
    evidence: str,
) -> None:
    name = re.sub(r"\s+", " ", name.strip(" .,;:()[]{}")).strip()
    if len(name) < 2:
        return
    if name.lower() in STOPWORDS:
        return
    bucket[(entity_type, name)].append(evidence[:220])


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)


def _find_known_terms(
    text: str,
    bucket: dict[tuple[EntityType, str], list[str]],
    terms: set[str],
    entity_type: EntityType,
) -> None:
    for term in sorted(terms, key=len, reverse=True):
        for match in _phrase_pattern(term).finditer(text):
            _add(bucket, entity_type, term, _snippet(text, match.start(), match.end()))


def _extract_capitalized_entities(
    text: str, bucket: dict[tuple[EntityType, str], list[str]]
) -> None:
    cap = r"(?:[A-Z][a-zA-Z0-9&.'-]+|[A-Z]{2,})"
    phrase_re = re.compile(rf"\b{cap}(?:\s+{cap}){{0,4}}\b")
    honorific_re = re.compile(
        r"\b(?:Dr|Mr|Ms|Mrs|Prof|Professor|Senator|President)\.?\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
    )

    for match in honorific_re.finditer(text):
        _add(bucket, EntityType.PERSON, match.group(1), _snippet(text, match.start(), match.end()))

    for match in phrase_re.finditer(text):
        name = match.group(0)
        words = name.split()
        if len(words) == 1 and name not in KNOWN_FRAMEWORKS and name not in KNOWN_PRODUCTS:
            continue
        if any(word in ORG_SUFFIXES for word in words) or words[-1] in ORG_SUFFIXES:
            _add(bucket, EntityType.ORGANIZATION, name, _snippet(text, match.start(), match.end()))
        elif 2 <= len(words) <= 4 and all(word[0].isupper() for word in words):
            if not any(word.lower() in STOPWORDS for word in words):
                _add(bucket, EntityType.PERSON, name, _snippet(text, match.start(), match.end()))


def _extract_values(text: str, bucket: dict[tuple[EntityType, str], list[str]]) -> None:
    patterns = [
        (
            EntityType.DATE,
            re.compile(
                rf"\b(?:{'|'.join(MONTHS)})\s+\d{{1,2}},?\s+\d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b(?:19|20)\d{{2}}\b"
            ),
        ),
        (
            EntityType.MONEY,
            re.compile(
                r"(?<!\w)(?:[$€£]\s?\d[\d,]*(?:\.\d+)?|"
                r"\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|dollars|euros))\b",
                re.IGNORECASE,
            ),
        ),
        (
            EntityType.PERCENTAGE,
            re.compile(
                r"\b\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?percent\b",
                re.IGNORECASE,
            ),
        ),
        (
            EntityType.NUMERIC_VALUE,
            re.compile(
                r"\b\d+(?:\.\d+)?\s?"
                r"(?:million|billion|trillion|ms|s|seconds|minutes|GB|MB|TB|"
                r"tokens|users|requests)\b",
                re.IGNORECASE,
            ),
        ),
    ]
    for entity_type, pattern in patterns:
        for match in pattern.finditer(text):
            _add(bucket, entity_type, match.group(0), _snippet(text, match.start(), match.end()))


def _domain_hint(url: str | None, bucket: dict[tuple[EntityType, str], list[str]]) -> None:
    host = urlsplit(url or "").hostname or ""
    if not host:
        return
    labels = [part for part in host.split(".") if part and part not in {"www", "com", "org", "net"}]
    if labels:
        label = labels[-1].replace("-", " ").title()
        _add(bucket, EntityType.ORGANIZATION, label, f"Domain hint from {host}")


def extract_entities_and_topics(
    *,
    question: str,
    text: str,
    title: str | None = None,
    url: str | None = None,
    max_entities: int = 80,
) -> ExtractionResult:
    combined = "\n".join(part for part in (question, title or "", text or "") if part)
    bucket: dict[tuple[EntityType, str], list[str]] = defaultdict(list)

    _domain_hint(url, bucket)
    _find_known_terms(combined, bucket, KNOWN_FRAMEWORKS, EntityType.FRAMEWORK_LIBRARY)
    _find_known_terms(combined, bucket, KNOWN_PRODUCTS, EntityType.PRODUCT)
    _find_known_terms(combined, bucket, KNOWN_LOCATIONS, EntityType.LOCATION)
    _find_known_terms(combined, bucket, TECH_TERMS, EntityType.TECHNICAL_TERM)
    _find_known_terms(combined, bucket, LEGAL_POLICY_TERMS, EntityType.LEGAL_POLICY_TERM)
    _extract_capitalized_entities(combined, bucket)
    _extract_values(combined, bucket)

    counts = Counter({key: len(value) for key, value in bucket.items()})
    entities: list[ExtractedEntity] = []
    for (entity_type, name), mentions in counts.most_common(max_entities):
        evidence = list(dict.fromkeys(bucket[(entity_type, name)]))[:4]
        confidence = min(0.98, 0.45 + mentions * 0.12)
        if entity_type in {
            EntityType.DATE,
            EntityType.MONEY,
            EntityType.PERCENTAGE,
            EntityType.FRAMEWORK_LIBRARY,
            EntityType.LEGAL_POLICY_TERM,
        }:
            confidence = min(0.99, confidence + 0.15)
        entities.append(
            ExtractedEntity(
                name=name,
                entity_type=entity_type,
                mentions=mentions,
                confidence=round(confidence, 3),
                evidence=evidence,
            )
        )

    warnings: list[str] = []
    if not entities:
        warnings.append("No deterministic entities extracted.")

    return ExtractionResult(
        entities=entities,
        topics=extract_topics(question=question, title=title, text=text),
        warnings=warnings,
    )
