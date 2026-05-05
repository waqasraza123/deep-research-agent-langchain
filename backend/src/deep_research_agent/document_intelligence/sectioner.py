from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .contracts import DocumentSection


@dataclass(frozen=True)
class HeadingCandidate:
    heading: str
    level: int
    start: int
    end: int
    confidence: float


INTRO_REFERENCE_HEADINGS = {
    "abstract",
    "summary",
    "executive summary",
    "introduction",
    "background",
    "methodology",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "reference",
    "bibliography",
    "appendix",
}


def _stable_id(source_id: str, heading: str, start: int) -> str:
    digest = hashlib.sha1(f"{source_id}|{start}|{heading}".encode("utf-8")).hexdigest()[:12]
    return f"{source_id}-sec-{digest}"


def _line_offsets(text: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    pos = 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\n")
        out.append((line, pos, pos + len(line)))
        pos += len(raw)
    return out


def _is_all_caps_heading(line: str) -> bool:
    s = line.strip(" .:-")
    if len(s) < 4 or len(s) > 90:
        return False
    letters = [ch for ch in s if ch.isalpha()]
    if len(letters) < 3:
        return False
    return sum(1 for ch in letters if ch.isupper()) / len(letters) >= 0.85


def _clean_heading(heading: str) -> str:
    heading = re.sub(r"^\s*#{1,6}\s*", "", heading)
    heading = re.sub(r"^\s*(?:section\s+)?\d+(?:\.\d+)*[\).:-]?\s*", "", heading, flags=re.I)
    heading = re.sub(r"\s+", " ", heading).strip(" #\t:-")
    return heading or "Untitled section"


def detect_heading_candidates(
    text: str, *, html_headings: list[str] | None = None
) -> list[HeadingCandidate]:
    candidates: list[HeadingCandidate] = []
    html_heading_set = {re.sub(r"\s+", " ", h).strip().lower() for h in html_headings or [] if h}

    for line, start, end in _line_offsets(text):
        stripped = line.strip()
        if not stripped or len(stripped) > 180:
            continue

        md = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if md:
            candidates.append(
                HeadingCandidate(
                    heading=_clean_heading(md.group(2)),
                    level=len(md.group(1)),
                    start=start,
                    end=end,
                    confidence=0.95,
                )
            )
            continue

        numbered = re.match(
            r"^(?:(?:section|article|part|chapter)\s+)?(\d+(?:\.\d+){0,4})[\).:-]?\s+(.+)$",
            stripped,
            flags=re.I,
        )
        if numbered and len(numbered.group(2)) <= 120:
            depth = numbered.group(1).count(".") + 1
            candidates.append(
                HeadingCandidate(
                    heading=_clean_heading(stripped),
                    level=min(depth, 6),
                    start=start,
                    end=end,
                    confidence=0.84,
                )
            )
            continue

        legal = re.match(r"^(?:[A-Z]\.|\([a-z0-9ivx]+\))\s+(.+)$", stripped)
        if legal and len(legal.group(1)) <= 120:
            candidates.append(
                HeadingCandidate(
                    heading=_clean_heading(stripped),
                    level=2,
                    start=start,
                    end=end,
                    confidence=0.72,
                )
            )
            continue

        lowered = stripped.lower().strip(":")
        if lowered in INTRO_REFERENCE_HEADINGS:
            candidates.append(
                HeadingCandidate(
                    heading=_clean_heading(stripped),
                    level=1,
                    start=start,
                    end=end,
                    confidence=0.8,
                )
            )
            continue

        if _is_all_caps_heading(stripped):
            candidates.append(
                HeadingCandidate(
                    heading=_clean_heading(stripped.title()),
                    level=1,
                    start=start,
                    end=end,
                    confidence=0.68,
                )
            )
            continue

        if stripped.lower() in html_heading_set:
            candidates.append(
                HeadingCandidate(
                    heading=_clean_heading(stripped),
                    level=2,
                    start=start,
                    end=end,
                    confidence=0.78,
                )
            )

    return candidates


def section_document(
    text: str,
    *,
    source_id: str,
    html_headings: list[str] | None = None,
) -> list[DocumentSection]:
    candidates = detect_heading_candidates(text, html_headings=html_headings)
    candidates = sorted(
        {(c.start, c.heading): c for c in candidates}.values(), key=lambda c: c.start
    )
    toc_filtered: list[HeadingCandidate] = []
    for idx, c in enumerate(candidates):
        near_next = idx + 1 < len(candidates) and candidates[idx + 1].start - c.start < 80
        if c.heading.lower() in {"table of contents", "contents"} or (
            near_next and re.search(r"\.{3,}\s*\d+$", c.heading)
        ):
            continue
        toc_filtered.append(c)
    candidates = toc_filtered

    if not candidates:
        return [
            DocumentSection(
                section_id=_stable_id(source_id, "Document", 0),
                heading="Document",
                level=1,
                start_offset=0,
                end_offset=len(text),
                text=text,
                parent_section_id=None,
                path=["Document"],
                confidence_score=0.4,
            )
        ]

    sections: list[DocumentSection] = []
    stack: list[tuple[int, str, str]] = []
    for idx, candidate in enumerate(candidates):
        end_offset = candidates[idx + 1].start if idx + 1 < len(candidates) else len(text)
        heading = candidate.heading
        section_id = _stable_id(source_id, heading, candidate.start)
        while stack and stack[-1][0] >= candidate.level:
            stack.pop()
        parent_id = stack[-1][1] if stack else None
        path = [item[2] for item in stack] + [heading]
        section_text = text[candidate.end : end_offset].strip()
        sections.append(
            DocumentSection(
                section_id=section_id,
                heading=heading,
                level=candidate.level,
                start_offset=candidate.start,
                end_offset=end_offset,
                text=section_text,
                parent_section_id=parent_id,
                path=path,
                confidence_score=candidate.confidence,
            )
        )
        stack.append((candidate.level, section_id, heading))

    if candidates[0].start > 0 and text[: candidates[0].start].strip():
        intro_id = _stable_id(source_id, "Preamble", 0)
        sections.insert(
            0,
            DocumentSection(
                section_id=intro_id,
                heading="Preamble",
                level=1,
                start_offset=0,
                end_offset=candidates[0].start,
                text=text[: candidates[0].start].strip(),
                parent_section_id=None,
                path=["Preamble"],
                confidence_score=0.45,
            ),
        )
    return sections
