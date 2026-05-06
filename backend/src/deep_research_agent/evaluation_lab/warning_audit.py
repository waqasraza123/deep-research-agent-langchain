from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .contracts import WarningAudit
from .governance_report import render_warning_audit, write_json

SERIOUS_PATTERNS = (
    "resourcewarning",
    "unclosed",
    "coroutine was never awaited",
    "runtimewarning",
    "securitywarning",
)


def classify_warning_text(text: str) -> tuple[str, str, str]:
    lowered = text.lower()
    if "pydantic" in lowered:
        category = "pydantic"
    elif "fastapi" in lowered or "starlette" in lowered:
        category = "fastapi"
    elif "langchain" in lowered:
        category = "langchain"
    elif "deepagents" in lowered:
        category = "deepagents"
    elif "pytest" in lowered:
        category = "pytest"
    elif "resourcewarning" in lowered or "unclosed" in lowered:
        category = "resource"
    elif "deprecat" in lowered:
        category = "deprecation"
    elif "userwarning" in lowered:
        category = "user_warning"
    else:
        category = "unknown"

    if any(pattern in lowered for pattern in SERIOUS_PATTERNS):
        severity = "serious"
    elif category in {"resource"}:
        severity = "high"
    elif category in {"pydantic", "fastapi", "langchain", "deepagents", "deprecation"}:
        severity = "medium"
    elif category == "pytest":
        severity = "low"
    else:
        severity = "low"

    origin = (
        "project" if "deep_research_agent" in lowered or "backend/src" in lowered else "external"
    )
    return category, severity, origin


def _warning_lines(warnings: list[str] | str) -> list[str]:
    if isinstance(warnings, str):
        return [line.strip() for line in warnings.splitlines() if line.strip()]
    return [str(item).strip() for item in warnings if str(item).strip()]


def summarize_warnings(
    warnings: list[str] | str,
    *,
    max_warning_count: int | None = None,
    max_serious_warnings: int | None = None,
) -> WarningAudit:
    lines = _warning_lines(warnings)
    by_category: Counter[str] = Counter()
    by_origin: Counter[str] = Counter()
    by_file: Counter[str] = Counter()
    serious = 0
    for line in lines:
        category, severity, origin = classify_warning_text(line)
        by_category[category] += 1
        by_origin[origin] += 1
        if severity == "serious":
            serious += 1
        match = re.match(r"([^:\s]+\.py):\d+", line)
        if match:
            by_file[match.group(1)] += 1
    budget_exceeded = False
    if max_warning_count is not None and len(lines) > max_warning_count:
        budget_exceeded = True
    if max_serious_warnings is not None and serious > max_serious_warnings:
        budget_exceeded = True
    recommendations = suggest_warning_fixes_from_counts(by_category, serious)
    return WarningAudit(
        total_warnings=len(lines),
        warnings_by_category=dict(sorted(by_category.items())),
        warnings_by_file=dict(by_file.most_common(20)),
        warnings_by_origin=dict(sorted(by_origin.items())),
        serious_warning_count=serious,
        budget_exceeded=budget_exceeded,
        recommendations=recommendations,
    )


def suggest_warning_fixes_from_counts(by_category: Counter[str], serious: int) -> list[str]:
    recommendations = []
    if serious:
        recommendations.append("Fix serious ResourceWarning/RuntimeWarning entries before merging.")
    if by_category.get("pydantic", 0):
        recommendations.append(
            "Migrate project Pydantic validators/config usage before broad filters."
        )
    if by_category.get("langchain", 0):
        recommendations.append("Track LangChain deprecations separately from project warnings.")
    if by_category.get("deprecation", 0):
        recommendations.append(
            "Group third-party deprecations by package and pin or upgrade intentionally."
        )
    if not recommendations:
        recommendations.append("No warning remediation needed for this audit.")
    return recommendations


def compare_warning_summaries(current: WarningAudit, baseline: WarningAudit | None) -> WarningAudit:
    if baseline is None:
        return current
    new_count = max(0, current.total_warnings - baseline.total_warnings)
    repeated = min(current.total_warnings, baseline.total_warnings)
    data = current.to_json_dict()
    data.update({"new_warning_count": new_count, "repeated_warning_count": repeated})
    validate = getattr(WarningAudit, "model_validate", None)
    return validate(data) if callable(validate) else WarningAudit.parse_obj(data)


def detect_warning_growth(
    current: WarningAudit,
    baseline: WarningAudit | None,
    *,
    max_growth_ratio: float | None,
) -> bool:
    if baseline is None or max_growth_ratio is None:
        return False
    if baseline.total_warnings == 0:
        return current.total_warnings > 0
    return (current.total_warnings / baseline.total_warnings) > max_growth_ratio


def suggest_warning_fixes(summary: WarningAudit) -> list[str]:
    return list(summary.recommendations)


def write_warning_audit(output_dir: Path, audit: WarningAudit) -> list[str]:
    write_json(output_dir / "warning_audit.json", audit)
    (output_dir / "warning_audit.md").write_text(render_warning_audit(audit), encoding="utf-8")
    return ["warning_audit.json", "warning_audit.md"]
