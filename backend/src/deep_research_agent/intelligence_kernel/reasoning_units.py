from __future__ import annotations

import json
import re
from pathlib import Path

from .contracts import (
    EvidenceUnit,
    KernelWarning,
    ResearchBlueprint,
    ResearchKernelSettings,
    SourceUnit,
    clamp_score,
    stable_id,
    tokenize,
    write_json,
)
from .source_units import load_sources_payload

DATE_RE = re.compile(
    r"\b(?:20\d{2}|19\d{2}|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
    re.I,
)
NUMBER_RE = re.compile(
    r"(?:[$€£]\s*)?\b\d+(?:[.,]\d+)*(?:%|x|ms|s|GB|MB|k|m|bn|million|billion)?\b", re.I
)
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[- ][A-Z][A-Za-z0-9]*)*\b")


def _read(path: Path, max_chars: int = 120_000) -> str:
    if not path.exists() or path.is_dir():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def _sentences(text: str) -> list[tuple[str | None, str]]:
    out: list[tuple[str | None, str]] = []
    section: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip()
            continue
        if stripped.startswith(("-", "*")):
            stripped = stripped.lstrip("-* ").strip()
        parts = re.split(r"(?<=[.!?])\s+", stripped)
        for part in parts:
            clean = part.strip()
            if len(clean) >= 28:
                out.append((section, clean))
    return out


def _classify(text: str) -> str:
    lc = text.lower()
    if "|" in text and text.count("|") >= 2:
        return "table_row"
    if text.startswith(('"', "'")) or "“" in text:
        return "quote"
    if NUMBER_RE.search(text):
        return "numeric_metric"
    if DATE_RE.search(text) or "version" in lc or "release" in lc:
        return "date_or_version"
    if any(w in lc for w in ("compared", "versus", "better", "worse", "faster", "cheaper")):
        return "comparison"
    if any(w in lc for w in ("because", "leads to", "results in", "due to")):
        return "causal_claim"
    if any(w in lc for w in ("should", "recommend", "best", "suitable", "ideal")):
        return "recommendation_support"
    if any(w in lc for w in ("risk", "failure", "limitation", "concern", "warning")):
        return "risk_signal"
    if any(w in lc for w in ("policy", "terms", "regulation", "compliance")):
        return "policy_statement"
    if any(w in lc for w in ("api", "sdk", "function", "class", "endpoint", "database")):
        return "code_or_api_reference"
    if " is " in lc or " are " in lc:
        return "definition"
    return "factual_statement"


def _source_texts(
    run_dir: Path, source_units: list[SourceUnit]
) -> list[tuple[SourceUnit | None, str, str]]:
    output: list[tuple[SourceUnit | None, str, str]] = []
    payload = load_sources_payload(run_dir)
    by_norm = {u.normalized_url: u for u in source_units}
    for item in payload:
        rel = str(item.get("local_path") or "")
        text = str(item.get("text") or item.get("content") or item.get("summary") or "")
        if rel:
            marker = "/runs/"
            if marker in rel:
                parts = rel.split(marker, 1)[-1].split("/", 1)
                rel = parts[1] if len(parts) == 2 else ""
            if rel and not rel.startswith("/") and "\\" not in rel and ".." not in rel:
                path = (run_dir / rel).resolve()
                if str(path).startswith(str(run_dir.resolve())):
                    text = _read(path)
        if not text:
            continue
        url = str(item.get("final_url") or item.get("url") or "")
        unit = next((u for u in source_units if u.url == url), None)
        if unit is None:
            from .source_units import normalize_source_url

            unit = by_norm.get(normalize_source_url(url))
        output.append((unit, "source", text))
    for artifact in ("notes.md", "report.md", "plan.md"):
        text = _read(run_dir / artifact)
        if text:
            output.append((None, artifact, text))
    return output


def build_evidence_units(
    run_dir: Path,
    blueprint: ResearchBlueprint,
    source_units: list[SourceUnit],
    settings: ResearchKernelSettings | None = None,
) -> tuple[list[EvidenceUnit], dict[str, object], list[KernelWarning]]:
    settings = settings or ResearchKernelSettings()
    q_terms = tokenize(blueprint.question)
    requirements = [r.lower() for r in blueprint.evidence_requirements]
    units: list[EvidenceUnit] = []
    seen: set[str] = set()
    for source_unit, artifact, text in _source_texts(run_dir, source_units):
        trust = 0.45
        role_boost = 0.0
        quality = 0.4
        if source_unit is not None:
            quality = source_unit.extraction_quality
            trust = {"high": 0.85, "medium": 0.65, "low": 0.3, "unknown": 0.45, "risky": 0.1}[
                source_unit.trust_level
            ]
            role_boost = (
                0.12
                if source_unit.source_role == "primary_evidence"
                else 0.06
                if source_unit.source_role == "secondary_context"
                else 0.0
            )
        for section, sentence in _sentences(text):
            normalized = " ".join(sentence.lower().split())
            if normalized in seen:
                continue
            sentence_terms = tokenize(sentence)
            overlap = len(q_terms & sentence_terms) / max(1, len(q_terms))
            entities = sorted(set(ENTITY_RE.findall(sentence)))[:12]
            numbers = NUMBER_RE.findall(sentence)[:12]
            dates = DATE_RE.findall(sentence)[:12]
            type_name = _classify(sentence)
            specificity = 0.08 if entities else 0.0
            specificity += 0.08 if numbers or dates else 0.0
            requirement_match = (
                0.08
                if any(
                    req and any(term in normalized for term in req.split()[:3])
                    for req in requirements
                )
                else 0.0
            )
            relevance = clamp_score(0.15 + overlap * 0.52 + specificity + requirement_match)
            support = clamp_score(relevance * 0.52 + trust * 0.25 + quality * 0.16 + role_boost)
            if relevance < 0.22 and not (numbers or dates or entities):
                continue
            eid = stable_id(
                "ev", artifact, source_unit.source_unit_id if source_unit else "", normalized
            )
            units.append(
                EvidenceUnit(
                    evidence_unit_id=eid,
                    source_unit_id=source_unit.source_unit_id if source_unit else None,
                    source_id=source_unit.source_id if source_unit else None,
                    url=source_unit.url if source_unit else None,
                    title=source_unit.title if source_unit else artifact,
                    text=sentence,
                    normalized_text=normalized,
                    section_hint=section,
                    evidence_type=type_name,  # type: ignore[arg-type]
                    entities=entities,
                    numbers=numbers,
                    dates=dates,
                    relevance_score=relevance,
                    support_score=support,
                    citation_ready=bool(
                        source_unit
                        and source_unit.source_role in {"primary_evidence", "secondary_context"}
                        and support >= 0.45
                    ),
                )
            )
            seen.add(normalized)
            if len(units) >= settings.max_evidence_units:
                break
        if len(units) >= settings.max_evidence_units:
            break
    by_source: dict[str, int] = {}
    for unit in units:
        key = unit.source_id or unit.title or "artifact"
        by_source[key] = by_source.get(key, 0) + 1
    missing = []
    all_text = " ".join(u.normalized_text for u in units)
    for req in blueprint.evidence_requirements:
        req_terms = tokenize(req)
        if req_terms and not (req_terms & tokenize(all_text)):
            missing.append(req)
    warnings: list[KernelWarning] = []
    if len([u for u in units if u.source_id]) <= 1 and len(units) > 0:
        warnings.append(
            KernelWarning(
                warning_id=stable_id("warn", "evidence", "diversity", blueprint.thread_id),
                subsystem="evidence_unitization",
                code="low_source_diversity",
                severity="medium",
                message="Evidence units are concentrated in one or no source-backed artifacts.",
                recommended_action="Add independent corroborating sources.",
            )
        )
    coverage = {
        "evidence_unit_count": len(units),
        "citation_ready_count": len([u for u in units if u.citation_ready]),
        "evidence_by_source": by_source,
        "missing_evidence_requirements": missing,
        "weak_evidence_areas": missing[:10],
        "source_diversity_warning": bool(warnings),
    }
    return units, coverage, warnings


def render_evidence_units_markdown(units: list[EvidenceUnit], coverage: dict[str, object]) -> str:
    lines = [
        "# Evidence Units",
        "",
        f"Total evidence units: {len(units)}",
        f"Citation-ready: {coverage.get('citation_ready_count', 0)}",
        "",
        "| Evidence | Type | Source | Relevance | Support |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for unit in units[:200]:
        source = unit.source_id or unit.title or "artifact"
        lines.append(
            f"| {unit.text[:120]} | `{unit.evidence_type}` | {source} | {unit.relevance_score:.2f} | {unit.support_score:.2f} |"
        )
    return "\n".join(lines) + "\n"


def write_evidence_artifacts(
    run_dir: Path, units: list[EvidenceUnit], coverage: dict[str, object]
) -> list[str]:
    write_json(run_dir / "evidence_units.json", {"evidence_units": units})
    write_json(run_dir / "evidence_coverage.json", coverage)
    (run_dir / "evidence_units.md").write_text(
        render_evidence_units_markdown(units, coverage), encoding="utf-8"
    )
    missing = coverage.get("missing_evidence_requirements") or []
    if isinstance(missing, list):
        missing_lines = "\n".join(f"- {item}" for item in missing) or "- None"
    else:
        missing_lines = "- None"
    (run_dir / "evidence_coverage.md").write_text(
        "# Evidence Coverage\n\n"
        f"- Evidence units: `{coverage.get('evidence_unit_count', 0)}`\n"
        f"- Citation-ready: `{coverage.get('citation_ready_count', 0)}`\n\n"
        "## Missing Requirements\n\n"
        f"{missing_lines}\n",
        encoding="utf-8",
    )
    return [
        "evidence_units.json",
        "evidence_units.md",
        "evidence_coverage.json",
        "evidence_coverage.md",
    ]


def read_evidence_units(run_dir: Path) -> list[EvidenceUnit]:
    path = run_dir / "evidence_units.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("evidence_units", data if isinstance(data, list) else [])
    return [EvidenceUnit(**row) for row in rows if isinstance(row, dict)]
