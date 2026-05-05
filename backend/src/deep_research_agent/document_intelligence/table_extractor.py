from __future__ import annotations

import csv
import hashlib
import io
import re

from .contracts import DocumentTable, DocumentTableCell


def _stable_table_id(source_id: str, start: int, rows: list[list[str]]) -> str:
    preview = "|".join("|".join(row) for row in rows[:3])
    digest = hashlib.sha1(f"{source_id}|{start}|{preview}".encode("utf-8")).hexdigest()[:12]
    return f"{source_id}-tbl-{digest}"


def _line_offsets(text: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    pos = 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\n")
        out.append((line, pos, pos + len(line)))
        pos += len(raw)
    return out


def _clean_cell(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.strip()).strip()


def _cells_from_rows(rows: list[list[str]], *, header: bool) -> list[DocumentTableCell]:
    cells: list[DocumentTableCell] = []
    for r_idx, row in enumerate(rows):
        for c_idx, cell in enumerate(row):
            cells.append(
                DocumentTableCell(
                    row_index=r_idx,
                    column_index=c_idx,
                    text=cell,
                    is_header=header and r_idx == 0,
                )
            )
    return cells


def render_readable_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [0] * max(len(row) for row in rows)
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines: list[str] = []
    for row_idx, row in enumerate(rows):
        padded = [row[idx] if idx < len(row) else "" for idx in range(len(widths))]
        lines.append(
            " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(padded)).rstrip()
        )
        if row_idx == 0 and len(rows) > 1:
            lines.append("-+-".join("-" * width for width in widths).rstrip())
    return "\n".join(lines)


def _table_from_rows(
    *,
    source_id: str,
    rows: list[list[str]],
    start: int,
    end: int,
    table_kind: str,
    confidence: float,
    warnings: list[str] | None = None,
) -> DocumentTable | None:
    rows = [[_clean_cell(cell) for cell in row] for row in rows]
    rows = [row for row in rows if any(row)]
    if len(rows) < 2 and table_kind != "key_value":
        return None
    max_cols = max(len(row) for row in rows) if rows else 0
    if max_cols < 2:
        return None
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]
    return DocumentTable(
        table_id=_stable_table_id(source_id, start, normalized),
        source_id=source_id,
        start_offset=start,
        end_offset=end,
        rows=normalized,
        cells=_cells_from_rows(normalized, header=table_kind != "key_value"),
        readable_text=render_readable_table(normalized),
        table_kind=table_kind,
        confidence_score=confidence,
        warnings=warnings or [],
    )


def _extract_markdown_tables(text: str, source_id: str) -> list[DocumentTable]:
    tables: list[DocumentTable] = []
    lines = _line_offsets(text)
    i = 0
    while i < len(lines):
        line, start, _ = lines[i]
        if "|" not in line:
            i += 1
            continue
        if i + 1 >= len(lines):
            i += 1
            continue
        sep = lines[i + 1][0].strip()
        if not re.match(r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?$", sep):
            i += 1
            continue
        block: list[tuple[str, int, int]] = [lines[i], lines[i + 1]]
        j = i + 2
        while j < len(lines) and "|" in lines[j][0].strip():
            block.append(lines[j])
            j += 1
        table_rows: list[list[str]] = []
        for idx, (raw, _, _) in enumerate(block):
            if idx == 1:
                continue
            cells = [_clean_cell(part) for part in raw.strip().strip("|").split("|")]
            table_rows.append(cells)
        table = _table_from_rows(
            source_id=source_id,
            rows=table_rows,
            start=start,
            end=block[-1][2],
            table_kind="markdown",
            confidence=0.94,
        )
        if table:
            tables.append(table)
        i = j
    return tables


def _extract_delimited_tables(text: str, source_id: str) -> list[DocumentTable]:
    tables: list[DocumentTable] = []
    lines = _line_offsets(text)
    i = 0
    while i < len(lines):
        raw, start, _ = lines[i]
        delimiter = None
        if "\t" in raw:
            delimiter = "\t"
        elif raw.count("|") >= 2:
            delimiter = "|"
        elif raw.count(",") >= 2:
            delimiter = ","
        if delimiter is None:
            i += 1
            continue
        block: list[tuple[str, int, int]] = []
        j = i
        while j < len(lines):
            line = lines[j][0]
            if (delimiter == "\t" and "\t" in line) or (
                delimiter != "\t" and line.count(delimiter) >= 2
            ):
                block.append(lines[j])
                j += 1
                continue
            break
        if len(block) >= 2:
            rows: list[list[str]] = []
            for line, _, _ in block:
                if delimiter == ",":
                    try:
                        parsed = next(csv.reader(io.StringIO(line)))
                    except Exception:
                        parsed = line.split(",")
                else:
                    parsed = line.strip().strip("|").split(delimiter)
                rows.append([_clean_cell(cell) for cell in parsed])
            table = _table_from_rows(
                source_id=source_id,
                rows=rows,
                start=start,
                end=block[-1][2],
                table_kind={
                    "\t": "tab_separated",
                    "|": "pipe_separated",
                    ",": "csv",
                }[delimiter],
                confidence=0.78 if delimiter == "," else 0.82,
                warnings=["Delimited text table detected heuristically."]
                if delimiter in {",", "|"}
                else [],
            )
            if table:
                tables.append(table)
        i = max(j, i + 1)
    return tables


def _extract_key_value_tables(text: str, source_id: str) -> list[DocumentTable]:
    tables: list[DocumentTable] = []
    lines = _line_offsets(text)
    i = 0
    while i < len(lines):
        block: list[tuple[str, int, int]] = []
        j = i
        while j < len(lines):
            line = lines[j][0].strip()
            if re.match(r"^[A-Za-z][A-Za-z0-9 /_-]{1,60}\s*[:=]\s*\S", line):
                block.append(lines[j])
                j += 1
                continue
            break
        if len(block) >= 3:
            rows = [["Key", "Value"]]
            for line, _, _ in block:
                key, value = re.split(r"\s*[:=]\s*", line.strip(), maxsplit=1)
                rows.append([key.strip(), value.strip()])
            table = _table_from_rows(
                source_id=source_id,
                rows=rows,
                start=block[0][1],
                end=block[-1][2],
                table_kind="key_value",
                confidence=0.74,
                warnings=["Key-value block represented as a table."],
            )
            if table:
                tables.append(table)
        i = max(j, i + 1)
    return tables


def extract_tables(
    text: str, *, source_id: str, source_type: str = "unknown"
) -> list[DocumentTable]:
    candidates: list[DocumentTable] = []
    candidates.extend(_extract_markdown_tables(text, source_id))
    candidates.extend(_extract_delimited_tables(text, source_id))
    candidates.extend(_extract_key_value_tables(text, source_id))
    candidates.sort(
        key=lambda table: (table.start_offset, -(table.end_offset - table.start_offset))
    )
    non_overlapping: list[DocumentTable] = []
    occupied: list[tuple[int, int]] = []
    for table in candidates:
        if any(table.start_offset < end and table.end_offset > start for start, end in occupied):
            continue
        non_overlapping.append(table)
        occupied.append((table.start_offset, table.end_offset))
    return non_overlapping
