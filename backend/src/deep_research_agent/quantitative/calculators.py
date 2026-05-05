from __future__ import annotations

import hashlib
from statistics import mean

from .contracts import CalculationResult, NumericValue


def _calc_id(operation: str, values: list[NumericValue]) -> str:
    raw = "|".join(value.raw_text for value in values)
    digest = hashlib.sha1(f"{operation}|{raw}".encode("utf-8")).hexdigest()[:12]
    return f"calc-{digest}"


def _compatible(values: list[NumericValue]) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    units = {value.unit for value in values if value.unit}
    currencies = {value.currency for value in values if value.currency}
    if len(units) > 1:
        warnings.append("Input units differ; calculation may be invalid.")
    if len(currencies) > 1:
        warnings.append("Input currencies differ; calculation may be invalid.")
    missing = [value.raw_text for value in values if value.normalized_value is None]
    if missing:
        warnings.append("Some inputs do not have normalized numeric values.")
    return not warnings, warnings


def calculate(operation: str, values: list[NumericValue]) -> CalculationResult:
    op = operation.lower().strip()
    numeric = [value for value in values if value.normalized_value is not None]
    compatible, warnings = _compatible(values)
    unit = next((value.unit for value in values if value.unit), None)
    currency = next((value.currency for value in values if value.currency), None)
    result: float | None = None
    valid = compatible and bool(numeric)

    if op == "difference":
        if len(numeric) < 2:
            valid = False
            warnings.append("Difference requires at least two numeric inputs.")
        else:
            result = numeric[0].normalized_value - numeric[1].normalized_value  # type: ignore[operator]
    elif op == "percentage_difference":
        if len(numeric) < 2 or numeric[1].normalized_value in (None, 0.0):
            valid = False
            warnings.append("Percentage difference requires a non-zero baseline.")
        else:
            result = (
                (numeric[0].normalized_value - numeric[1].normalized_value)
                / abs(numeric[1].normalized_value)
                * 100.0
            )
            unit = "%"
            currency = None
    elif op == "sum":
        result = sum(
            value.normalized_value for value in numeric if value.normalized_value is not None
        )
    elif op == "average":
        result = mean(
            value.normalized_value for value in numeric if value.normalized_value is not None
        )
    elif op == "min":
        result = min(
            value.normalized_value for value in numeric if value.normalized_value is not None
        )
    elif op == "max":
        result = max(
            value.normalized_value for value in numeric if value.normalized_value is not None
        )
    elif op == "range_comparison":
        ranges = [
            value
            for value in values
            if value.range_start is not None and value.range_end is not None
        ]
        if len(ranges) < 2:
            valid = False
            warnings.append("Range comparison requires at least two ranges.")
        else:
            result = ranges[0].range_end - ranges[1].range_end  # type: ignore[operator]
    else:
        valid = False
        warnings.append(f"Unsupported calculation operation: {operation}")

    expression = f"{op}(" + ", ".join(value.raw_text for value in values) + ")"
    return CalculationResult(
        calculation_id=_calc_id(op, values),
        operation=op,
        inputs=values,
        result_value=result,
        unit=unit,
        currency=currency,
        expression=expression,
        valid=valid and result is not None,
        warnings=warnings,
    )


def convert_rate(value: NumericValue, target_unit: str) -> CalculationResult:
    source_unit = (value.unit or "").lower()
    target = target_unit.lower()
    result = None
    warnings: list[str] = []
    valid = value.normalized_value is not None
    if not valid:
        warnings.append("Rate conversion requires a numeric value.")
    elif source_unit in {"requests per second", "requests/sec", "request/sec"} and target in {
        "requests per minute",
        "requests/min",
    }:
        result = value.normalized_value * 60.0
    elif source_unit in {"seconds", "second", "sec", "secs"} and target == "ms":
        result = value.normalized_value * 1000.0
    elif source_unit in {"ms", "milliseconds"} and target in {"seconds", "second", "sec"}:
        result = value.normalized_value / 1000.0
    else:
        valid = False
        warnings.append(f"Unsupported rate/unit conversion from {source_unit} to {target}.")
    return CalculationResult(
        calculation_id=_calc_id(f"convert_to_{target}", [value]),
        operation="rate_conversion",
        inputs=[value],
        result_value=result,
        unit=target_unit,
        currency=value.currency,
        expression=f"convert({value.raw_text}, {target_unit})",
        valid=valid and result is not None,
        warnings=warnings,
    )
