from __future__ import annotations

import hashlib
import math

from .contracts import (
    NumericClaim,
    NumericValue,
    QuantitativeConsistencyCheck,
    QuantitativeWarning,
)
from .metric_detector import normalize_metric_name


def _check_id(message: str, claim_id: str | None = None) -> str:
    digest = hashlib.sha1(f"{claim_id or ''}|{message}".encode("utf-8")).hexdigest()[:12]
    return f"qcheck-{digest}"


def _warning_id(code: str, message: str) -> str:
    digest = hashlib.sha1(f"{code}|{message}".encode("utf-8")).hexdigest()[:12]
    return f"qwarn-{digest}"


def _same_numeric(a: NumericValue, b: NumericValue, *, tolerance: float = 0.005) -> bool:
    if a.normalized_value is None or b.normalized_value is None:
        return False
    if a.currency != b.currency or a.unit != b.unit or a.percentage != b.percentage:
        return False
    scale = max(1.0, abs(a.normalized_value), abs(b.normalized_value))
    return math.isclose(a.normalized_value, b.normalized_value, abs_tol=tolerance * scale)


def check_report_claims(
    report_claims: list[NumericClaim],
    source_values: list[NumericValue],
) -> list[QuantitativeConsistencyCheck]:
    checks: list[QuantitativeConsistencyCheck] = []
    for claim in report_claims:
        for observed in claim.values:
            if observed.kind == "version":
                continue
            matches = [value for value in source_values if _same_numeric(observed, value)]
            if matches:
                checks.append(
                    QuantitativeConsistencyCheck(
                        check_id=_check_id("numeric claim supported", claim.claim_id),
                        claim_id=claim.claim_id,
                        status="pass",
                        message=(
                            f"Numeric claim value {observed.raw_text} appears in source evidence."
                        ),
                        expected_value=matches[0],
                        observed_value=observed,
                        source_ids=[m.source_id for m in matches if m.source_id],
                        confidence_score=0.86,
                    )
                )
                continue
            checks.append(
                QuantitativeConsistencyCheck(
                    check_id=_check_id("numeric claim unsupported", claim.claim_id),
                    claim_id=claim.claim_id,
                    status="warning",
                    message=(
                        f"Numeric claim value {observed.raw_text} was not found with matching "
                        "unit/currency in source evidence."
                    ),
                    observed_value=observed,
                    confidence_score=0.72,
                )
            )
    return checks


def check_conflicts(values: list[NumericValue]) -> list[QuantitativeConsistencyCheck]:
    checks: list[QuantitativeConsistencyCheck] = []
    buckets: dict[tuple[str, str | None, str | None, bool], list[NumericValue]] = {}
    for value in values:
        metric = normalize_metric_name(value.metric_name)
        if not metric or value.normalized_value is None or value.kind in {"version", "date"}:
            continue
        buckets.setdefault((metric, value.unit, value.currency, value.percentage), []).append(value)
    for (metric, _unit, _currency, _percentage), bucket in buckets.items():
        source_ids = {value.source_id for value in bucket if value.source_id}
        if len(source_ids) < 2 or len(bucket) < 2:
            continue
        normalized = [
            value.normalized_value for value in bucket if value.normalized_value is not None
        ]
        if not normalized:
            continue
        spread = max(normalized) - min(normalized)
        scale = max(1.0, max(abs(v) for v in normalized))
        if spread / scale > 0.1:
            checks.append(
                QuantitativeConsistencyCheck(
                    check_id=_check_id(f"conflicting {metric}"),
                    status="warning",
                    message=(
                        f"Conflicting values detected for {metric}: "
                        f"min={min(normalized):g}, max={max(normalized):g}."
                    ),
                    expected_value=bucket[0],
                    observed_value=bucket[-1],
                    source_ids=sorted(s for s in source_ids if s),
                    confidence_score=0.74,
                )
            )
    return checks


def check_percentage_formula(claims: list[NumericClaim]) -> list[QuantitativeConsistencyCheck]:
    checks: list[QuantitativeConsistencyCheck] = []
    for claim in claims:
        percentages = [value for value in claim.values if value.percentage]
        raw_values = [
            value
            for value in claim.values
            if not value.percentage and value.normalized_value is not None
        ]
        if not percentages or len(raw_values) < 2:
            continue
        baseline = raw_values[1].normalized_value
        current = raw_values[0].normalized_value
        if baseline in (None, 0.0) or current is None:
            continue
        expected = (current - baseline) / abs(baseline) * 100.0
        for pct in percentages:
            if pct.normalized_value is None:
                continue
            if abs(expected - pct.normalized_value) > max(1.0, abs(expected) * 0.03):
                checks.append(
                    QuantitativeConsistencyCheck(
                        check_id=_check_id("percentage formula mismatch", claim.claim_id),
                        claim_id=claim.claim_id,
                        status="warning",
                        message=(
                            f"Percentage {pct.raw_text} does not match raw values "
                            f"({expected:.2f}% calculated)."
                        ),
                        observed_value=pct,
                        confidence_score=0.68,
                    )
                )
    return checks


def warnings_from_checks(
    checks: list[QuantitativeConsistencyCheck],
    values: list[NumericValue],
) -> list[QuantitativeWarning]:
    warnings: list[QuantitativeWarning] = []
    for check in checks:
        if check.status in {"warning", "fail"}:
            warnings.append(
                QuantitativeWarning(
                    warning_id=_warning_id(check.status, check.message),
                    code=f"consistency_{check.status}",
                    message=check.message,
                    severity="warning" if check.status == "warning" else "error",
                    claim_id=check.claim_id,
                    context=check.observed_value.context if check.observed_value else "",
                )
            )
    for value in values:
        if value.kind == "benchmark" and not value.metric_name:
            warnings.append(
                QuantitativeWarning(
                    warning_id=_warning_id("benchmark_missing_context", value.raw_text),
                    code="benchmark_missing_context",
                    message=f"Benchmark-like value {value.raw_text} is missing metric context.",
                    severity="warning",
                    source_id=value.source_id,
                    context=value.context,
                )
            )
        if value.currency and value.unit is None and "per" in value.context.lower():
            warnings.append(
                QuantitativeWarning(
                    warning_id=_warning_id("pricing_unit_unclear", value.raw_text),
                    code="pricing_unit_unclear",
                    message=f"Pricing value {value.raw_text} may need a billing unit.",
                    severity="info",
                    source_id=value.source_id,
                    context=value.context,
                )
            )
    deduped: dict[str, QuantitativeWarning] = {}
    for warning in warnings:
        deduped[warning.warning_id] = warning
    return list(deduped.values())
