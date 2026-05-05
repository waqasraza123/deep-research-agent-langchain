from __future__ import annotations

import re

from .contracts import SourceVersionSignal, VersionSignalType

SEMVER_RE = re.compile(r"\bv?([0-9]+\.[0-9]+(?:\.[0-9]+){0,2}(?:[-+][a-z0-9.\-]+)?)\b", re.I)
MAJOR_VERSION_RE = re.compile(r"\bv([1-9][0-9]?)(?!\.\d)\b", re.I)

KEYWORD_SIGNALS: tuple[tuple[re.Pattern[str], VersionSignalType, bool, bool, float], ...] = (
    (re.compile(r"\brelease notes?\b", re.I), "release_notes", False, False, 0.82),
    (re.compile(r"\bchangelog\b", re.I), "changelog", False, False, 0.82),
    (re.compile(r"\bdeprecated|deprecation\b", re.I), "deprecation", True, False, 0.86),
    (re.compile(r"\bmigration guide\b", re.I), "migration_guide", False, False, 0.82),
    (re.compile(r"\bold docs?\b", re.I), "old_docs", True, False, 0.84),
    (re.compile(r"\barchived docs?|archive(d)?\b", re.I), "archived_docs", True, False, 0.9),
    (re.compile(r"\blegacy docs?|legacy\b", re.I), "legacy_docs", True, False, 0.88),
    (
        re.compile(r"\bstable docs?|stable version|stable release\b", re.I),
        "stable_docs",
        False,
        True,
        0.74,
    ),
    (
        re.compile(r"\bcurrent docs?|current version|current release\b", re.I),
        "current_docs",
        False,
        True,
        0.78,
    ),
    (
        re.compile(r"\blatest docs?|latest version|latest release\b", re.I),
        "latest_docs",
        False,
        True,
        0.8,
    ),
    (re.compile(r"\bbeta\b", re.I), "beta_docs", False, False, 0.72),
    (re.compile(r"\bpreview\b", re.I), "preview_docs", False, False, 0.72),
)


def detect_version_signals(
    text: str,
    *,
    source_id: str | None = None,
    source_url: str | None = None,
    title: str | None = None,
    max_chars: int = 80_000,
) -> list[SourceVersionSignal]:
    joined = "\n".join([title or "", source_url or "", text[:max_chars] if text else ""])
    signals: list[SourceVersionSignal] = []
    for match in SEMVER_RE.finditer(joined):
        signals.append(
            SourceVersionSignal(
                source_id=source_id,
                source_url=source_url,
                signal_type="semantic_version",
                raw_text=match.group(0),
                normalized_version=match.group(1),
                confidence_score=0.82,
                text_offset=match.start(),
                surrounding_context=_context(joined, match.start(), match.end()),
            )
        )
    for match in MAJOR_VERSION_RE.finditer(joined):
        signals.append(
            SourceVersionSignal(
                source_id=source_id,
                source_url=source_url,
                signal_type="major_version",
                raw_text=match.group(0),
                normalized_version=match.group(1),
                confidence_score=0.72,
                text_offset=match.start(),
                surrounding_context=_context(joined, match.start(), match.end()),
            )
        )
    for pattern, signal_type, outdated, current, confidence in KEYWORD_SIGNALS:
        for match in pattern.finditer(joined):
            signals.append(
                SourceVersionSignal(
                    source_id=source_id,
                    source_url=source_url,
                    signal_type=signal_type,
                    raw_text=match.group(0),
                    confidence_score=confidence,
                    text_offset=match.start(),
                    surrounding_context=_context(joined, match.start(), match.end()),
                    outdated_hint=outdated,
                    current_hint=current,
                )
            )
    return _dedupe(signals)[:40]


def _context(text: str, start: int, end: int, *, radius: int = 90) -> str:
    return " ".join(text[max(0, start - radius) : min(len(text), end + radius)].split())


def _dedupe(signals: list[SourceVersionSignal]) -> list[SourceVersionSignal]:
    best: dict[tuple[str, str | None, str | None], SourceVersionSignal] = {}
    for signal in signals:
        key = (signal.signal_type, signal.raw_text.lower(), signal.source_id)
        current = best.get(key)
        if current is None or signal.confidence_score > current.confidence_score:
            best[key] = signal
    return sorted(
        best.values(),
        key=lambda item: (item.outdated_hint, item.current_hint, item.confidence_score),
        reverse=True,
    )
