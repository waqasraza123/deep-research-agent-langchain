from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

from .contracts import CSVProfile
from .table_normalizer import profile_rows


def _sniff_delimiter(text: str) -> str:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except Exception:
        counts = {delimiter: sample.count(delimiter) for delimiter in [",", "\t", ";", "|"]}
        return max(counts.items(), key=lambda item: item[1])[0] if any(counts.values()) else ","


def profile_csv_text(
    text: str,
    *,
    csv_id: str | None = None,
    source_id: str | None = None,
    source_url: str | None = None,
    path: str | None = None,
) -> CSVProfile:
    delimiter = _sniff_delimiter(text)
    rows: list[list[str]] = []
    malformed_rows = 0
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        expected: int | None = None
        for row in reader:
            rows.append(row)
            if expected is None and any(cell.strip() for cell in row):
                expected = len(row)
            elif expected is not None and len(row) != expected:
                malformed_rows += 1
    except csv.Error:
        malformed_rows += 1
        rows = [line.split(delimiter) for line in text.splitlines()]

    if csv_id is None:
        digest = hashlib.sha1((path or text[:1000]).encode("utf-8")).hexdigest()[:12]
        csv_id = f"csv-{digest}"
    table_profile = profile_rows(
        rows,
        table_id=csv_id,
        source_id=source_id,
        source_url=source_url,
    )
    warnings = list(table_profile.warnings)
    if malformed_rows:
        warnings.append(f"{malformed_rows} malformed CSV rows detected.")
    return CSVProfile(
        csv_id=csv_id,
        source_id=source_id,
        source_url=source_url,
        path=path,
        delimiter=delimiter,
        column_names=table_profile.column_names,
        row_count=table_profile.row_count,
        column_count=table_profile.column_count,
        empty_values=table_profile.empty_values,
        numeric_columns=table_profile.numeric_columns,
        date_columns=table_profile.date_columns,
        categorical_columns=table_profile.categorical_columns,
        detected_units=table_profile.detected_units,
        column_profiles=table_profile.column_profiles,
        duplicate_rows=table_profile.duplicate_rows,
        possible_identifier_columns=table_profile.possible_identifier_columns,
        malformed_rows=malformed_rows,
        warnings=warnings,
    )


def profile_csv_file(
    path: Path,
    *,
    source_id: str | None = None,
    source_url: str | None = None,
) -> CSVProfile:
    return profile_csv_text(
        path.read_text(encoding="utf-8", errors="ignore"),
        source_id=source_id,
        source_url=source_url,
        path=str(path),
    )
