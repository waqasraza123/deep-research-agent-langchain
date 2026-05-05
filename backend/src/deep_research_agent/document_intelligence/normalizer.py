from __future__ import annotations

import csv
import hashlib
import io
import re
import unicodedata

from .contracts import DocumentExtractionWarning, DocumentNormalizationResult

UNICODE_SPACES = dict.fromkeys(
    map(
        ord,
        [
            "\u00a0",
            "\u1680",
            "\u2000",
            "\u2001",
            "\u2002",
            "\u2003",
            "\u2004",
            "\u2005",
            "\u2006",
            "\u2007",
            "\u2008",
            "\u2009",
            "\u200a",
            "\u202f",
            "\u205f",
            "\u3000",
        ],
    ),
    " ",
)

NAVIGATION_PATTERNS = (
    re.compile(r"^(?:home|menu|navigation|skip to content|privacy|terms|login|sign in)$", re.I),
    re.compile(r"^(?:previous|next|back to top|share|subscribe)$", re.I),
)


def content_hash(text: str) -> str:
    normalized = " ".join((text or "").strip().lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _line_key(line: str) -> str:
    return re.sub(r"\d+", "#", " ".join(line.lower().split()))


def _remove_repeated_headers_footers(lines: list[str]) -> tuple[list[str], int]:
    candidates: dict[str, int] = {}
    original: dict[str, str] = {}
    for line in lines:
        s = line.strip()
        if not s or len(s) > 120:
            continue
        key = _line_key(s)
        candidates[key] = candidates.get(key, 0) + 1
        original.setdefault(key, s)
    repeated = {
        key for key, count in candidates.items() if count >= 3 and len(original.get(key, "")) >= 3
    }
    if not repeated:
        return lines, 0
    out = [line for line in lines if _line_key(line.strip()) not in repeated]
    return out, len(lines) - len(out)


def _remove_repeated_boilerplate_blocks(text: str) -> tuple[str, int]:
    blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]
    seen: set[str] = set()
    removed = 0
    out: list[str] = []
    for block in blocks:
        key = re.sub(r"\s+", " ", block.lower()).strip()
        if len(key) < 40:
            out.append(block)
            continue
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(block)
    return "\n\n".join(out), removed


def _looks_like_wrapped_continuation(prev: str, current: str) -> bool:
    if not prev or not current:
        return False
    if prev.endswith((".", "!", "?", ":", ";", ")", "]")):
        return False
    if re.match(r"^(?:#{1,6}\s+|\d+(?:\.\d+)*\.?\s+|[-*]\s+)", current):
        return False
    if current[:1].isupper() and len(current.split()) <= 8:
        return False
    return True


def _repair_pdf_line_wraps(text: str) -> tuple[str, bool]:
    text2 = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    changed = text2 != text
    lines = text2.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not out or not stripped:
            out.append(stripped)
            continue
        prev = out[-1]
        if _looks_like_wrapped_continuation(prev, stripped):
            out[-1] = f"{prev} {stripped}"
            changed = True
        else:
            out.append(stripped)
    return "\n".join(out), changed


def _normalize_markdown_headings(text: str) -> tuple[str, bool]:
    changed = False

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        hashes = match.group(1)
        heading = re.sub(r"\s+", " ", match.group(2)).strip(" #")
        changed = True
        return f"{hashes} {heading}"

    return re.sub(r"^\s*(#{1,6})\s*(.*?)\s*#*\s*$", repl, text, flags=re.M), changed


def _csv_to_readable_text(text: str) -> tuple[str, bool]:
    sample = text.strip()
    if not sample or "\n" not in sample:
        return text, False
    try:
        dialect = csv.Sniffer().sniff(sample[:4096])
    except Exception:
        dialect = csv.excel
    try:
        rows = list(csv.reader(io.StringIO(sample), dialect))
    except Exception:
        return text, False
    rows = [[cell.strip() for cell in row] for row in rows if any(cell.strip() for cell in row)]
    if len(rows) < 2 or max(len(row) for row in rows) < 2:
        return text, False
    header = rows[0]
    lines = ["CSV table:"]
    for row in rows[1:]:
        pairs = []
        for idx, cell in enumerate(row):
            label = header[idx] if idx < len(header) and header[idx] else f"Column {idx + 1}"
            pairs.append(f"{label}: {cell}")
        lines.append("- " + "; ".join(pairs))
    return "\n".join(lines), True


def normalize_text(
    raw_text: str,
    *,
    source_id: str,
    source_type: str = "unknown",
) -> DocumentNormalizationResult:
    warnings: list[DocumentExtractionWarning] = []
    transformations: list[str] = []
    text = raw_text or ""
    if not text.strip():
        warnings.append(
            DocumentExtractionWarning(
                code="empty_extraction",
                message="Source extraction produced no readable text.",
                severity="error",
                source_id=source_id,
            )
        )

    normalized = unicodedata.normalize("NFKC", text).translate(UNICODE_SPACES)
    if normalized != text:
        transformations.append("unicode_spacing")

    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[\t ]+", " ", normalized)
    if source_type == "pdf":
        normalized, changed = _repair_pdf_line_wraps(normalized)
        if changed:
            transformations.append("pdf_line_wrap_cleanup")
    else:
        normalized = re.sub(r"(\w)-\n(\w)", r"\1\2", normalized)
        transformations.append("hyphenated_line_break_cleanup")

    if source_type in {"csv", "text/csv"}:
        normalized, changed = _csv_to_readable_text(normalized)
        if changed:
            transformations.append("csv_readable_text")

    if source_type in {"md", "markdown", "text/markdown"}:
        normalized, changed = _normalize_markdown_headings(normalized)
        if changed:
            transformations.append("markdown_heading_cleanup")

    lines = [line.rstrip() for line in normalized.splitlines()]
    lines, removed_headers = _remove_repeated_headers_footers(lines)
    if removed_headers:
        transformations.append("repeated_header_footer_removal")
        warnings.append(
            DocumentExtractionWarning(
                code="repeated_header_footer_removed",
                message=f"Removed {removed_headers} repeated short header/footer lines.",
                source_id=source_id,
            )
        )

    filtered_lines: list[str] = []
    nav_removed = 0
    for line in lines:
        stripped = line.strip()
        if any(pattern.match(stripped) for pattern in NAVIGATION_PATTERNS):
            nav_removed += 1
            continue
        filtered_lines.append(stripped)
    if nav_removed:
        transformations.append("navigation_text_removal")

    normalized = "\n".join(filtered_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r" {2,}", " ", normalized)
    normalized, removed_blocks = _remove_repeated_boilerplate_blocks(normalized)
    if removed_blocks:
        transformations.append("repeated_boilerplate_block_removal")

    normalized = normalized.strip()
    if len(normalized) < max(40, len(text.strip()) * 0.05) and text.strip():
        warnings.append(
            DocumentExtractionWarning(
                code="aggressive_normalization",
                message="Normalized text is much shorter than raw extraction.",
                source_id=source_id,
            )
        )

    return DocumentNormalizationResult(
        source_id=source_id,
        raw_text=text,
        normalized_text=normalized,
        source_type=source_type,
        content_hash=content_hash(normalized),
        warnings=warnings,
        transformations=transformations,
    )
