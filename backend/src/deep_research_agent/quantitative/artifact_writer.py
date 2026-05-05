from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import QuantitativeEvidence, QuantitativeSummary, model_to_plain

QUANTITATIVE_ARTIFACTS = (
    "quantitative_profile.json",
    "quantitative_profile.md",
    "numeric_claims.json",
    "numeric_claims.md",
    "table_profiles.json",
    "csv_profiles.json",
    "quantitative_comparisons.json",
    "quantitative_comparisons.md",
    "quantitative_warnings.md",
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_summary(thread_id: str, evidence: QuantitativeEvidence) -> QuantitativeSummary:
    return QuantitativeSummary(
        thread_id=thread_id,
        value_count=len(evidence.numeric_values),
        claim_count=len(evidence.numeric_claims),
        metric_count=len(evidence.metrics),
        table_count=len(evidence.table_profiles),
        csv_count=len(evidence.csv_profiles),
        comparison_count=len(evidence.comparisons),
        warning_count=len(evidence.warnings),
        consistency_failures=sum(
            1 for check in evidence.consistency_checks if check.status in {"warning", "fail"}
        ),
        evidence=evidence,
        warnings=evidence.warnings,
    )


def render_profile_markdown(summary: QuantitativeSummary) -> str:
    evidence = summary.evidence
    lines = [
        "# Quantitative Profile",
        "",
        f"- Thread: `{summary.thread_id}`",
        f"- Numeric values: {summary.value_count}",
        f"- Numeric claims: {summary.claim_count}",
        f"- Metrics: {summary.metric_count}",
        f"- Tables: {summary.table_count}",
        f"- CSVs: {summary.csv_count}",
        f"- Comparisons: {summary.comparison_count}",
        f"- Warnings: {summary.warning_count}",
        "",
    ]
    if evidence and evidence.metrics:
        lines.append("## Metrics")
        lines.append("")
        for metric in evidence.metrics[:30]:
            label = metric.normalized_name
            unit = metric.currency or metric.unit or "unit unknown"
            lines.append(f"- `{label}` ({unit}), sources={len(metric.source_ids)}")
        lines.append("")
    if evidence and evidence.table_profiles:
        lines.append("## Tables")
        lines.append("")
        for table in evidence.table_profiles[:20]:
            lines.append(
                f"- `{table.table_id}` rows={table.row_count}, cols={table.column_count}, "
                f"numeric={', '.join(table.numeric_columns) or 'none'}"
            )
        lines.append("")
    if evidence and evidence.csv_profiles:
        lines.append("## CSVs")
        lines.append("")
        for csv_profile in evidence.csv_profiles[:20]:
            lines.append(
                f"- `{csv_profile.csv_id}` rows={csv_profile.row_count}, "
                f"cols={csv_profile.column_count}, malformed={csv_profile.malformed_rows}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_claims_markdown(evidence: QuantitativeEvidence) -> str:
    lines = ["# Numeric Claims", ""]
    if not evidence.numeric_claims:
        lines.append("No numeric claims detected.")
        return "\n".join(lines).rstrip() + "\n"
    for claim in evidence.numeric_claims[:200]:
        values = ", ".join(value.raw_text for value in claim.values) or "none"
        lines.extend(
            [
                f"## {claim.claim_id}",
                "",
                f"- Origin: `{claim.origin}`",
                f"- Source: `{claim.source_id or claim.origin_ref or 'unknown'}`",
                f"- Metric: `{claim.metric_name or 'unknown'}`",
                f"- Values: {values}",
                f"- Text: {claim.text}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_comparisons_markdown(evidence: QuantitativeEvidence) -> str:
    lines = ["# Quantitative Comparisons", ""]
    if not evidence.comparisons:
        lines.append("No directly comparable metrics detected.")
        return "\n".join(lines).rstrip() + "\n"
    for comparison in evidence.comparisons:
        lines.extend(
            [
                f"## {comparison.metric_name}",
                "",
                f"- Comparable: {'yes' if comparison.comparable else 'no'}",
                f"- Direction: `{comparison.direction or 'unknown'}`",
                f"- Winner: `{comparison.winner or 'unknown'}`",
                "",
            ]
        )
        for item in comparison.values:
            unit = item.value.currency or item.value.unit or ""
            normalized = (
                item.value.normalized_value
                if item.value.normalized_value is not None
                else "n/a"
            )
            lines.append(
                f"- {item.entity}: `{item.value.raw_text}` "
                f"({normalized} {unit})"
            )
        if comparison.warnings:
            lines.append("")
            for warning in comparison.warnings:
                lines.append(f"- Warning: {warning}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_warnings_markdown(evidence: QuantitativeEvidence) -> str:
    lines = ["# Quantitative Warnings", ""]
    if not evidence.warnings:
        lines.append("No quantitative warnings.")
        return "\n".join(lines).rstrip() + "\n"
    for warning in evidence.warnings:
        source = f" ({warning.source_id})" if warning.source_id else ""
        claim = f" claim={warning.claim_id}" if warning.claim_id else ""
        lines.append(f"- {warning.severity}: {warning.code}{source}{claim} - {warning.message}")
    return "\n".join(lines).rstrip() + "\n"


def write_quantitative_artifacts(
    run_dir: Path,
    *,
    thread_id: str,
    evidence: QuantitativeEvidence,
) -> QuantitativeSummary:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(thread_id, evidence)
    _write_json(run_dir / "quantitative_profile.json", model_to_plain(summary))
    (run_dir / "quantitative_profile.md").write_text(
        render_profile_markdown(summary),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "numeric_claims.json",
        {
            "thread_id": thread_id,
            "claims": [model_to_plain(claim) for claim in evidence.numeric_claims],
        },
    )
    (run_dir / "numeric_claims.md").write_text(render_claims_markdown(evidence), encoding="utf-8")
    _write_json(
        run_dir / "table_profiles.json",
        {
            "thread_id": thread_id,
            "tables": [model_to_plain(table) for table in evidence.table_profiles],
        },
    )
    _write_json(
        run_dir / "csv_profiles.json",
        {
            "thread_id": thread_id,
            "csv_profiles": [model_to_plain(profile) for profile in evidence.csv_profiles],
        },
    )
    _write_json(
        run_dir / "quantitative_comparisons.json",
        {
            "thread_id": thread_id,
            "comparisons": [model_to_plain(comparison) for comparison in evidence.comparisons],
        },
    )
    (run_dir / "quantitative_comparisons.md").write_text(
        render_comparisons_markdown(evidence),
        encoding="utf-8",
    )
    (run_dir / "quantitative_warnings.md").write_text(
        render_warnings_markdown(evidence),
        encoding="utf-8",
    )
    return summary
