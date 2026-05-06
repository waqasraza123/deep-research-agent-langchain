"""Deterministic disclosure-risk reports for run artifacts and exports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.artifacts import list_artifacts, now_iso_utc, safe_thread_id

DisclosureSeverity = Literal["info", "low", "medium", "high", "critical"]
DisclosureReadiness = Literal["clear", "review_required", "blocked"]

DISCLOSURE_REPORT_JSON = "disclosure_report.json"
DISCLOSURE_REPORT_MD = "disclosure_report.md"

RAW_SOURCE_PREFIXES = ("sources/",)
SANITIZED_SOURCE_PREFIXES = ("sanitized_sources/",)
CONTROL_ARTIFACTS = {
    "operator_audit.jsonl",
    "operator_audit.md",
    "custody_certificate.json",
    "custody_certificate.md",
    "integrity_report.json",
    "integrity_report.md",
    "handoff_manifest.json",
    "handoff_manifest.md",
    DISCLOSURE_REPORT_JSON,
    DISCLOSURE_REPORT_MD,
}
TEXT_SUFFIXES = {
    ".csv",
    ".html",
    ".htm",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
SECRET_TEXT_PATTERNS = {
    "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    "github_token": re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{16,}\b"),
    "bearer_token": re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}", re.I),
    "credential_assignment": re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*[^\s,'\"]+"
    ),
}


class RunDisclosureRequest(BaseModel):
    requested_by: str = "operator"
    require_no_high_risk: bool = False
    scan_text: bool = True
    max_files: int = Field(default=1000, ge=1, le=10000)
    max_scan_bytes: int = Field(default=500_000, ge=1024, le=10_000_000)
    notes: str = ""


class DisclosureFinding(BaseModel):
    finding_id: str
    severity: DisclosureSeverity
    category: str
    path: str | None = None
    summary: str = ""
    recommended_action: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DisclosureArtifactSummary(BaseModel):
    artifact_count: int = 0
    scanned_text_count: int = 0
    skipped_large_count: int = 0
    raw_source_count: int = 0
    sanitized_source_count: int = 0
    export_present: bool = False
    export_redact_enabled: bool | None = None
    export_includes_raw_sources: bool | None = None
    exported_count: int | None = None
    skipped_export_count: int | None = None


class RunDisclosureReport(BaseModel):
    report_version: str = "1.0"
    thread_id: str
    generated_at: str
    requested_by: str = "operator"
    readiness: DisclosureReadiness = "review_required"
    risk_level: DisclosureSeverity = "low"
    findings: list[DisclosureFinding] = Field(default_factory=list)
    high_or_critical_count: int = 0
    artifact_summary: DisclosureArtifactSummary = Field(default_factory=DisclosureArtifactSummary)
    notes: str = ""


def build_run_disclosure_report(
    *,
    runs_dir: Path,
    thread_id: str,
    request: RunDisclosureRequest | None = None,
) -> RunDisclosureReport:
    request = request or RunDisclosureRequest()
    requested_by = request.requested_by.strip() if request.requested_by.strip() else "operator"
    run_dir = _safe_run_dir(runs_dir, thread_id)
    artifacts = list_artifacts(runs_dir, thread_id)
    selected = artifacts[: request.max_files]
    findings: list[DisclosureFinding] = []
    summary = DisclosureArtifactSummary(artifact_count=len(artifacts))

    if len(artifacts) > len(selected):
        findings.append(
            DisclosureFinding(
                finding_id="artifact_scan_limit",
                severity="medium",
                category="scan_limit",
                summary="Artifact scan was truncated by max_files.",
                recommended_action="Increase max_files and regenerate the report for a full scan.",
                metadata={"max_files": request.max_files, "artifact_count": len(artifacts)},
            )
        )

    for artifact in selected:
        if artifact.path in CONTROL_ARTIFACTS:
            continue
        if artifact.path.startswith(RAW_SOURCE_PREFIXES):
            summary.raw_source_count += 1
            findings.append(
                DisclosureFinding(
                    finding_id=f"raw_source:{artifact.path}",
                    severity="high",
                    category="raw_source",
                    path=artifact.path,
                    summary="Raw source artifact is present in the run directory.",
                    recommended_action=(
                        "Exclude raw sources from public exports unless the recipient is approved "
                        "to receive captured source text."
                    ),
                    metadata={"size_bytes": artifact.size_bytes},
                )
            )
        elif artifact.path.startswith(SANITIZED_SOURCE_PREFIXES):
            summary.sanitized_source_count += 1

        path = run_dir / artifact.path
        if not request.scan_text or not _is_text_artifact(path):
            continue
        if artifact.size_bytes > request.max_scan_bytes:
            summary.skipped_large_count += 1
            findings.append(
                DisclosureFinding(
                    finding_id=f"large_text:{artifact.path}",
                    severity="medium",
                    category="large_text",
                    path=artifact.path,
                    summary="Text artifact exceeded max_scan_bytes and was not scanned for secrets.",
                    recommended_action="Increase max_scan_bytes or inspect the artifact manually.",
                    metadata={
                        "size_bytes": artifact.size_bytes,
                        "max_scan_bytes": request.max_scan_bytes,
                    },
                )
            )
            continue
        summary.scanned_text_count += 1
        findings.extend(_secret_findings(path, artifact.path))

    export_findings, export_summary = _export_findings(run_dir)
    findings.extend(export_findings)
    summary.export_present = export_summary.get("export_present", False)
    summary.export_redact_enabled = export_summary.get("redact")
    summary.export_includes_raw_sources = export_summary.get("include_raw_sources")
    summary.exported_count = export_summary.get("exported_count")
    summary.skipped_export_count = export_summary.get("skipped_count")

    risk_level = _max_severity(finding.severity for finding in findings)
    high_or_critical = sum(1 for finding in findings if finding.severity in {"high", "critical"})
    readiness: DisclosureReadiness = "clear"
    if request.require_no_high_risk and high_or_critical:
        readiness = "blocked"
    elif findings:
        readiness = "review_required"
    report = RunDisclosureReport(
        thread_id=thread_id,
        generated_at=now_iso_utc(),
        requested_by=requested_by,
        readiness=readiness,
        risk_level=risk_level,
        findings=findings,
        high_or_critical_count=high_or_critical,
        artifact_summary=summary,
        notes=request.notes,
    )
    write_run_disclosure_report(run_dir, report)
    return report


def read_run_disclosure_report(runs_dir: Path, thread_id: str) -> RunDisclosureReport:
    run_dir = _safe_run_dir(runs_dir, thread_id)
    path = run_dir / DISCLOSURE_REPORT_JSON
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(RunDisclosureReport, "model_validate", None)
    if callable(validate):
        return validate(data)
    return RunDisclosureReport.parse_obj(data)


def write_run_disclosure_report(run_dir: Path, report: RunDisclosureReport) -> list[str]:
    (run_dir / DISCLOSURE_REPORT_JSON).write_text(
        json.dumps(_model_to_plain(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / DISCLOSURE_REPORT_MD).write_text(
        render_run_disclosure_report_markdown(report),
        encoding="utf-8",
    )
    return [DISCLOSURE_REPORT_JSON, DISCLOSURE_REPORT_MD]


def render_run_disclosure_report_markdown(report: RunDisclosureReport) -> str:
    lines = [
        "# Run Disclosure Report",
        "",
        f"- Thread ID: `{report.thread_id}`",
        f"- Generated at: `{report.generated_at}`",
        f"- Requested by: `{report.requested_by}`",
        f"- Readiness: `{report.readiness}`",
        f"- Risk level: `{report.risk_level}`",
        f"- Findings: {len(report.findings)}",
        f"- High or critical findings: {report.high_or_critical_count}",
        "",
        "## Artifact Summary",
        "",
        f"- Artifacts considered: {report.artifact_summary.artifact_count}",
        f"- Text artifacts scanned: {report.artifact_summary.scanned_text_count}",
        f"- Large text artifacts skipped: {report.artifact_summary.skipped_large_count}",
        f"- Raw source artifacts: {report.artifact_summary.raw_source_count}",
        f"- Sanitized source artifacts: {report.artifact_summary.sanitized_source_count}",
        f"- Export present: `{report.artifact_summary.export_present}`",
        f"- Export redaction enabled: `{report.artifact_summary.export_redact_enabled}`",
        f"- Export includes raw sources: `{report.artifact_summary.export_includes_raw_sources}`",
        "",
        "## Findings",
        "",
    ]
    if report.findings:
        for finding in report.findings:
            path = f" `{finding.path}`" if finding.path else ""
            lines.extend(
                [
                    f"### {finding.category}{path}",
                    "",
                    f"- Severity: `{finding.severity}`",
                    f"- Summary: {finding.summary or 'None'}",
                    f"- Recommended action: {finding.recommended_action or 'None'}",
                    "",
                ]
            )
    else:
        lines.extend(["- None", ""])
    if report.notes:
        lines.extend(["## Notes", "", report.notes.strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _secret_findings(path: Path, rel_path: str) -> list[DisclosureFinding]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return [
            DisclosureFinding(
                finding_id=f"unreadable_text:{rel_path}",
                severity="medium",
                category="unreadable_text",
                path=rel_path,
                summary=f"Text artifact could not be decoded: {type(e).__name__}: {e}.",
                recommended_action="Inspect the artifact manually before external disclosure.",
            )
        ]
    findings: list[DisclosureFinding] = []
    for pattern_name, pattern in SECRET_TEXT_PATTERNS.items():
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        findings.append(
            DisclosureFinding(
                finding_id=f"secret:{pattern_name}:{rel_path}",
                severity="critical",
                category="likely_secret",
                path=rel_path,
                summary=f"Likely secret pattern `{pattern_name}` appears in the artifact.",
                recommended_action=(
                    "Remove or rotate the secret source, regenerate affected artifacts, and export "
                    "with redaction enabled."
                ),
                metadata={"pattern": pattern_name, "match_count": len(matches)},
            )
        )
    return findings


def _export_findings(run_dir: Path) -> tuple[list[DisclosureFinding], dict[str, Any]]:
    path = run_dir / "exports" / "export_manifest.json"
    if not path.exists() or path.is_dir():
        return (
            [
                DisclosureFinding(
                    finding_id="export_manifest_missing",
                    severity="medium",
                    category="export",
                    summary="Export manifest is missing.",
                    recommended_action="Generate an export bundle before external handoff.",
                )
            ],
            {"export_present": False},
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return (
            [
                DisclosureFinding(
                    finding_id="export_manifest_unreadable",
                    severity="high",
                    category="export",
                    path="exports/export_manifest.json",
                    summary=f"Export manifest is unreadable: {type(e).__name__}: {e}.",
                    recommended_action="Regenerate the export bundle before handoff.",
                )
            ],
            {"export_present": True},
        )
    if not isinstance(manifest, dict):
        return (
            [
                DisclosureFinding(
                    finding_id="export_manifest_invalid",
                    severity="high",
                    category="export",
                    path="exports/export_manifest.json",
                    summary="Export manifest is not a JSON object.",
                    recommended_action="Regenerate the export bundle before handoff.",
                )
            ],
            {"export_present": True},
        )
    findings: list[DisclosureFinding] = []
    if manifest.get("redact") is not True:
        findings.append(
            DisclosureFinding(
                finding_id="export_redaction_disabled",
                severity="high",
                category="export",
                path="exports/export_manifest.json",
                summary="Export manifest indicates redaction was disabled.",
                recommended_action="Regenerate the export with redaction enabled unless exact bytes are required.",
                metadata={"redact": manifest.get("redact")},
            )
        )
    if manifest.get("include_raw_sources") is True:
        findings.append(
            DisclosureFinding(
                finding_id="export_raw_sources_included",
                severity="high",
                category="export",
                path="exports/export_manifest.json",
                summary="Export manifest indicates raw source files were included.",
                recommended_action="Confirm recipient authorization or regenerate without raw sources.",
                metadata={"include_raw_sources": manifest.get("include_raw_sources")},
            )
        )
    return findings, {
        "export_present": True,
        "redact": manifest.get("redact"),
        "include_raw_sources": manifest.get("include_raw_sources"),
        "exported_count": manifest.get("exported_count"),
        "skipped_count": manifest.get("skipped_count"),
    }


def _is_text_artifact(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def _max_severity(severities) -> DisclosureSeverity:
    order = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    max_value = 1
    for severity in severities:
        max_value = max(max_value, order.get(str(severity), 1))
    for label, value in order.items():
        if value == max_value:
            return label  # type: ignore[return-value]
    return "low"


def _safe_run_dir(runs_dir: Path, thread_id: str) -> Path:
    safe_thread_id(thread_id)
    root = runs_dir.resolve()
    run_dir = (root / thread_id).resolve()
    if root != run_dir and root not in run_dir.parents:
        raise ValueError("Invalid thread_id")
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(thread_id)
    return run_dir


def _model_to_plain(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return model.dict()
