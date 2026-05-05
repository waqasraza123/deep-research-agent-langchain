from __future__ import annotations

import hashlib
import re

from .contracts import ComparisonValue, NumericValue, QuantitativeComparison
from .metric_detector import normalize_metric_name


def _entity_from_value(value: NumericValue) -> str:
    context = value.context or ""
    raw_idx = context.lower().find(value.raw_text.lower())
    before = context[:raw_idx] if raw_idx >= 0 else context[:80]
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9.+#-]*(?:\s+[A-Z][A-Za-z0-9.+#-]*){0,3}", before)
    if candidates:
        return candidates[-1].strip()
    return value.source_id or value.source_url or "unknown"


def _comparison_id(metric: str, values: list[NumericValue]) -> str:
    digest = hashlib.sha1(
        (metric + "|" + "|".join(value.raw_text for value in values)).encode("utf-8")
    ).hexdigest()[:12]
    return f"qcmp-{digest}"


def build_comparisons(
    values: list[NumericValue],
    *,
    max_per_metric: int = 12,
) -> list[QuantitativeComparison]:
    buckets: dict[tuple[str, str | None, str | None], list[NumericValue]] = {}
    for value in values:
        if value.kind == "version" or value.normalized_value is None:
            continue
        metric = normalize_metric_name(value.metric_name)
        if not metric:
            continue
        key = (metric, value.unit, value.currency)
        buckets.setdefault(key, []).append(value)

    comparisons: list[QuantitativeComparison] = []
    for (metric, unit, currency), metric_values in buckets.items():
        entities: dict[str, NumericValue] = {}
        for value in metric_values:
            entity = _entity_from_value(value)
            if entity not in entities:
                entities[entity] = value
        if len(entities) < 2:
            continue
        selected = list(entities.items())[:max_per_metric]
        warnings: list[str] = []
        units = {value.unit for _, value in selected if value.unit}
        currencies = {value.currency for _, value in selected if value.currency}
        comparable = True
        if len(units) > 1 or len(currencies) > 1:
            comparable = False
            warnings.append(
                "Values use different units or currencies and are not directly comparable."
            )
        if any(value.kind == "benchmark" and not value.context for _, value in selected):
            warnings.append("Benchmark values are missing context.")
        winner = None
        direction = None
        if comparable:
            lower_keywords = {"latency", "cost", "price", "pricing", "issues"}
            lower_is_better = metric in lower_keywords
            sorted_values = sorted(
                selected,
                key=lambda item: item[1].normalized_value
                if item[1].normalized_value is not None
                else float("inf"),
                reverse=not lower_is_better,
            )
            winner = sorted_values[0][0]
            direction = "lower_is_better" if lower_is_better else "higher_is_better"
        comparisons.append(
            QuantitativeComparison(
                comparison_id=_comparison_id(metric, [value for _, value in selected]),
                metric_name=metric,
                unit=unit,
                currency=currency,
                values=[
                    ComparisonValue(
                        entity=entity,
                        value=value,
                        source_id=value.source_id,
                        source_url=value.source_url,
                    )
                    for entity, value in selected
                ],
                winner=winner,
                direction=direction,
                comparable=comparable,
                warnings=warnings,
            )
        )
    return sorted(comparisons, key=lambda comparison: comparison.metric_name)
