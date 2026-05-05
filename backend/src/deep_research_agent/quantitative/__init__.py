from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifact_writer import (
    QUANTITATIVE_ARTIFACTS,
    write_quantitative_artifacts,
)
from .calculators import calculate, convert_rate
from .comparison import build_comparisons
from .consistency_checker import (
    check_conflicts,
    check_percentage_formula,
    check_report_claims,
    warnings_from_checks,
)
from .contracts import (
    CalculationResult,
    CSVProfile,
    MetricDefinition,
    NumericClaim,
    NumericValue,
    QuantitativeComparison,
    QuantitativeConsistencyCheck,
    QuantitativeEvidence,
    QuantitativeSummary,
    QuantitativeWarning,
    TableProfile,
    model_to_plain,
)
from .csv_profiler import profile_csv_file, profile_csv_text
from .metric_detector import build_metric_definitions
from .number_extractor import extract_numeric_claims, extract_numeric_values
from .table_normalizer import profile_rows

__all__ = [
    "CSVProfile",
    "CalculationResult",
    "MetricDefinition",
    "NumericClaim",
    "NumericValue",
    "QUANTITATIVE_ARTIFACTS",
    "QuantitativeComparison",
    "QuantitativeConsistencyCheck",
    "QuantitativeEvidence",
    "QuantitativeSummary",
    "QuantitativeWarning",
    "TableProfile",
    "build_quantitative_evidence",
    "calculate",
    "convert_rate",
    "extract_numeric_claims",
    "extract_numeric_values",
    "model_to_plain",
    "profile_csv_file",
    "profile_csv_text",
    "profile_rows",
    "rebuild_quantitative_artifacts",
    "write_quantitative_artifacts",
]


def _load_json(path: Path) -> Any:
    try:
        if not path.exists() or path.is_dir():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_source_path(thread_dir: Path, local_path: str | None) -> Path | None:
    if not local_path:
        return None
    rel = str(local_path)
    if "runs/" in rel:
        parts = rel.split("runs/", 1)[-1].split("/", 1)
        rel = parts[1] if len(parts) == 2 else ""
    if not rel or rel.startswith("/") or ".." in rel or "\\" in rel:
        return None
    candidate = (thread_dir / rel).resolve()
    root = thread_dir.resolve()
    if not str(candidate).startswith(str(root)):
        return None
    return candidate


def _source_records(thread_dir: Path) -> list[dict[str, Any]]:
    data = _load_json(thread_dir / "sources.json")
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _source_id(item: dict[str, Any], idx: int) -> str:
    return str(item.get("source_id") or item.get("url") or f"source-{idx + 1}")


def _read_text(path: Path, *, max_chars: int = 250_000) -> str:
    if not path.exists() or path.is_dir():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _table_profiles_from_document_artifacts(thread_dir: Path) -> list[TableProfile]:
    data = _load_json(thread_dir / "document_tables.json")
    if not isinstance(data, dict):
        return []
    profiles: list[TableProfile] = []
    for item in data.get("tables", []):
        if not isinstance(item, dict):
            continue
        rows = item.get("rows")
        if not isinstance(rows, list):
            continue
        profiles.append(
            profile_rows(
                rows,
                table_id=str(item.get("table_id") or ""),
                source_id=item.get("source_id"),
                caption=item.get("caption"),
            )
        )
    return profiles


def _csv_profiles_from_sources(thread_dir: Path, sources: list[dict[str, Any]]) -> list[CSVProfile]:
    profiles: list[CSVProfile] = []
    for idx, source in enumerate(sources):
        path = _safe_source_path(thread_dir, source.get("local_path"))
        if path is None or not path.exists() or path.is_dir():
            continue
        source_type = " ".join(
            str(source.get(key) or "")
            for key in ("content_type", "document_kind", "url", "local_path")
        ).lower()
        if ".csv" not in source_type and "csv" not in source_type:
            continue
        profiles.append(
            profile_csv_file(
                path,
                source_id=_source_id(source, idx),
                source_url=source.get("final_url") or source.get("url"),
            )
        )
    return profiles


def _source_texts(
    thread_dir: Path,
    sources: list[dict[str, Any]],
) -> list[tuple[str, str | None, str]]:
    texts: list[tuple[str, str | None, str]] = []
    for idx, source in enumerate(sources):
        path = _safe_source_path(thread_dir, source.get("local_path"))
        if path is None:
            continue
        text = _read_text(path)
        if not text:
            continue
        texts.append((_source_id(source, idx), source.get("final_url") or source.get("url"), text))
    return texts


def build_quantitative_evidence(
    run_dir: Path,
    *,
    thread_id: str | None = None,
) -> QuantitativeEvidence:
    sources = _source_records(run_dir)
    numeric_values: list[NumericValue] = []
    numeric_claims: list[NumericClaim] = []

    for source_id, source_url, text in _source_texts(run_dir, sources):
        values = extract_numeric_values(text, source_id=source_id, source_url=source_url)
        numeric_values.extend(values)
        numeric_claims.extend(
            extract_numeric_claims(
                text,
                origin="source",
                origin_ref=source_id,
                source_id=source_id,
                source_url=source_url,
            )
        )

    report_claims: list[NumericClaim] = []
    for origin, rel_path in (("notes", "notes.md"), ("report", "report.md")):
        text = _read_text(run_dir / rel_path, max_chars=80_000)
        if not text:
            continue
        claims = extract_numeric_claims(text, origin=origin, origin_ref=rel_path)
        numeric_claims.extend(claims)
        if origin == "report":
            report_claims.extend(claims)

    table_profiles = _table_profiles_from_document_artifacts(run_dir)
    csv_profiles = _csv_profiles_from_sources(run_dir, sources)
    metrics = build_metric_definitions(numeric_values)
    comparisons = build_comparisons(numeric_values)
    checks = []
    checks.extend(check_report_claims(report_claims, numeric_values))
    checks.extend(check_conflicts(numeric_values))
    checks.extend(check_percentage_formula(numeric_claims))
    warnings = warnings_from_checks(checks, numeric_values)
    for comparison in comparisons:
        for message in comparison.warnings:
            warnings.append(
                QuantitativeWarning(
                    warning_id=f"{comparison.comparison_id}-warning",
                    code="comparison_incomparable",
                    message=message,
                    severity="medium",
                    affected_artifacts=[
                        "quantitative_profile.json",
                        "quantitative_comparisons.json",
                    ],
                )
            )
    return QuantitativeEvidence(
        evidence_id=f"quantitative-{thread_id or run_dir.name}",
        thread_id=thread_id,
        numeric_values=numeric_values,
        numeric_claims=numeric_claims,
        metrics=metrics,
        table_profiles=table_profiles,
        csv_profiles=csv_profiles,
        comparisons=comparisons,
        consistency_checks=checks,
        warnings=warnings,
    )


def rebuild_quantitative_artifacts(run_dir: Path, *, thread_id: str) -> QuantitativeSummary:
    evidence = build_quantitative_evidence(run_dir, thread_id=thread_id)
    return write_quantitative_artifacts(run_dir, thread_id=thread_id, evidence=evidence)
