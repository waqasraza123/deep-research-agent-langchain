from __future__ import annotations

import hashlib
import re
from statistics import mean
from typing import Any

from .contracts import ColumnProfile, TableProfile
from .number_extractor import extract_numeric_values

_DATE_CELL_RE = re.compile(
    r"^\s*(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\s*$",
    re.I,
)


def _clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _safe_number(cell: str) -> tuple[float | None, str | None]:
    values = [
        value
        for value in extract_numeric_values(cell, include_dates=False)
        if value.normalized_value is not None and value.kind != "version"
    ]
    if len(values) != 1:
        return None, None
    value = values[0]
    if value.raw_text.strip() != cell.strip() and value.unit is None and value.currency is None:
        return None, None
    return value.normalized_value, value.unit or value.currency


def _headers_from_rows(rows: list[list[str]]) -> tuple[list[str], list[list[str]], list[str]]:
    warnings: list[str] = []
    if not rows:
        return [], [], warnings
    first = [_clean_cell(cell) for cell in rows[0]]
    has_text = any(re.search(r"[A-Za-z]", cell) for cell in first)
    if has_text:
        headers = [cell or f"column_{idx + 1}" for idx, cell in enumerate(first)]
        body = rows[1:]
    else:
        headers = [f"column_{idx + 1}" for idx in range(max(len(row) for row in rows))]
        body = rows
        warnings.append("No header row detected; generated generic column names.")
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for header in headers:
        key = header
        count = seen.get(key, 0)
        seen[key] = count + 1
        deduped.append(key if count == 0 else f"{key}_{count + 1}")
    return deduped, body, warnings


def profile_rows(
    rows: list[list[str]],
    *,
    table_id: str | None = None,
    source_id: str | None = None,
    source_url: str | None = None,
    caption: str | None = None,
) -> TableProfile:
    cleaned = [
        [_clean_cell(cell) for cell in row]
        for row in rows
        if any(_clean_cell(c) for c in row)
    ]
    if not table_id:
        digest = hashlib.sha1(repr(cleaned[:5]).encode("utf-8")).hexdigest()[:12]
        table_id = f"table-{digest}"
    headers, body, warnings = _headers_from_rows(cleaned)
    column_count = max([len(headers), *(len(row) for row in body)] or [0])
    if len(headers) < column_count:
        headers.extend(f"column_{idx + 1}" for idx in range(len(headers), column_count))
    normalized_rows = [row + [""] * (column_count - len(row)) for row in body]
    empty_values = sum(1 for row in normalized_rows for cell in row if not cell)
    duplicate_rows = len(normalized_rows) - len({tuple(row) for row in normalized_rows})
    malformed_rows = sum(1 for row in body if len(row) != column_count)
    if malformed_rows:
        warnings.append(f"{malformed_rows} rows had a different column count.")

    column_profiles: list[ColumnProfile] = []
    numeric_columns: list[str] = []
    date_columns: list[str] = []
    categorical_columns: list[str] = []
    detected_units: dict[str, str] = {}
    possible_ids: list[str] = []

    for idx, name in enumerate(headers[:column_count]):
        cells = [row[idx] for row in normalized_rows]
        non_empty = [cell for cell in cells if cell]
        numbers: list[float] = []
        units: dict[str, int] = {}
        date_count = 0
        for cell in non_empty:
            if _DATE_CELL_RE.match(cell):
                date_count += 1
                continue
            number, unit = _safe_number(cell)
            if number is not None:
                numbers.append(number)
                if unit:
                    units[unit] = units.get(unit, 0) + 1
        empty_count = len(cells) - len(non_empty)
        distinct_count = len(set(non_empty))
        detected_unit = max(units.items(), key=lambda item: item[1])[0] if units else None
        numeric_ratio = len(numbers) / max(1, len(non_empty))
        date_ratio = date_count / max(1, len(non_empty))
        identifier_name = bool(re.search(r"\bid\b|identifier|uuid|slug", name, re.I))
        if identifier_name and non_empty and distinct_count / max(1, len(non_empty)) >= 0.5:
            possible_ids.append(name)
        if not non_empty:
            inferred = "empty"
        elif numeric_ratio >= 0.8:
            inferred = "numeric"
            numeric_columns.append(name)
        elif date_ratio >= 0.8:
            inferred = "date"
            date_columns.append(name)
        elif distinct_count == len(non_empty) and len(non_empty) >= 3 and (
            identifier_name or distinct_count > 0.9 * len(non_empty)
        ):
            inferred = "identifier"
            if name not in possible_ids:
                possible_ids.append(name)
        elif numeric_ratio > 0 or date_ratio > 0:
            inferred = "mixed"
        else:
            inferred = "categorical"
            categorical_columns.append(name)
        if detected_unit:
            detected_units[name] = detected_unit
        column_profiles.append(
            ColumnProfile(
                name=name,
                index=idx,
                empty_count=empty_count,
                numeric_count=len(numbers),
                date_count=date_count,
                distinct_count=distinct_count,
                detected_unit=detected_unit,
                inferred_type=inferred,  # type: ignore[arg-type]
                min_value=min(numbers) if numbers else None,
                max_value=max(numbers) if numbers else None,
                mean_value=mean(numbers) if numbers else None,
                sample_values=non_empty[:5],
            )
        )

    return TableProfile(
        table_id=table_id,
        source_id=source_id,
        source_url=source_url,
        caption=caption,
        column_names=headers[:column_count],
        row_count=len(normalized_rows),
        column_count=column_count,
        empty_values=empty_values,
        numeric_columns=numeric_columns,
        date_columns=date_columns,
        categorical_columns=categorical_columns,
        detected_units=detected_units,
        column_profiles=column_profiles,
        duplicate_rows=duplicate_rows,
        possible_identifier_columns=possible_ids,
        warnings=warnings,
    )
