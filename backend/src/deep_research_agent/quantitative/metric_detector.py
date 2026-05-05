from __future__ import annotations

import re

from .contracts import MetricDefinition, NumericValue

METRIC_HINTS = {
    "uptime",
    "latency",
    "context window",
    "context",
    "requests per second",
    "request rate",
    "price",
    "pricing",
    "cost",
    "users",
    "stars",
    "issues",
    "release date",
    "benchmark",
    "score",
    "tokens",
    "memory",
    "storage",
    "limit",
    "cap",
    "throughput",
}

LOWER_IS_BETTER = {"latency", "cost", "price", "pricing", "issues", "duration", "response time"}
HIGHER_IS_BETTER = {"uptime", "stars", "score", "throughput", "requests per second"}


def normalize_metric_name(name: str | None) -> str | None:
    if not name:
        return None
    cleaned = re.sub(r"[^a-zA-Z0-9/%$ ]+", " ", name.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def detect_metric_name(context: str, raw_text: str) -> str | None:
    if not context:
        return None
    try:
        start = context.lower().find(raw_text.lower())
    except Exception:
        start = -1
    if start < 0:
        start = max(0, len(context) // 2)
        end = start
    else:
        end = start + len(raw_text)

    before = context[max(0, start - 60) : start]
    after = context[end : min(len(context), end + 60)]

    nearest: tuple[int, str] | None = None
    for hint in sorted(METRIC_HINTS, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(hint)}\b")
        for match in pattern.finditer(before.lower()):
            distance = len(before) - match.end()
            if nearest is None or distance < nearest[0]:
                nearest = (distance, hint)
        for match in pattern.finditer(after.lower()):
            distance = match.start()
            if nearest is None or distance < nearest[0]:
                nearest = (distance, hint)
    if nearest is not None:
        return nearest[1]

    before_words = re.findall(r"[A-Za-z][A-Za-z0-9/_-]*", before)[-4:]
    after_words = re.findall(r"[A-Za-z][A-Za-z0-9/_-]*", after)[:4]
    if after_words and after_words[0].lower() in {
        "uptime",
        "latency",
        "users",
        "tokens",
        "stars",
        "issues",
        "score",
    }:
        return after_words[0].lower()
    if before_words:
        phrase = " ".join(before_words[-2:]).lower()
        if phrase not in {"in the", "of the", "and the", "per user"}:
            return phrase
    return None


def metric_from_value(value: NumericValue) -> MetricDefinition | None:
    name = normalize_metric_name(value.metric_name)
    if not name or value.kind == "version":
        return None
    category = "general"
    if value.currency:
        category = "pricing"
    elif value.percentage:
        category = "percentage"
    elif value.kind == "date":
        category = "date"
    elif "benchmark" in name or "score" in name:
        category = "benchmark"
    higher_is_better = None
    if name in LOWER_IS_BETTER:
        higher_is_better = False
    if name in HIGHER_IS_BETTER:
        higher_is_better = True
    return MetricDefinition(
        name=name,
        normalized_name=name,
        unit=value.unit,
        currency=value.currency,
        category=category,
        higher_is_better=higher_is_better,
        source_ids=[value.source_id] if value.source_id else [],
        confidence_score=value.confidence_score,
    )


def build_metric_definitions(values: list[NumericValue]) -> list[MetricDefinition]:
    by_name: dict[tuple[str, str | None, str | None], MetricDefinition] = {}
    for value in values:
        metric = metric_from_value(value)
        if metric is None:
            continue
        key = (metric.normalized_name, metric.unit, metric.currency)
        existing = by_name.get(key)
        if existing is None:
            by_name[key] = metric
            continue
        for source_id in metric.source_ids:
            if source_id not in existing.source_ids:
                existing.source_ids.append(source_id)
        existing.confidence_score = max(existing.confidence_score, metric.confidence_score)
    return sorted(
        by_name.values(),
        key=lambda m: (m.normalized_name, m.unit or "", m.currency or ""),
    )
