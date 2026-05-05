from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from deep_research_agent.source_identity import source_domain, source_identity_from_dict

from .artifact_writer import SOURCE_SAFETY_ARTIFACTS, write_source_safety_artifacts
from .content_sanitizer import sanitize_source_content
from .contracts import (
    PromptInjectionFinding,
    SanitizationMode,
    SanitizedSourceContent,
    SourcePoisoningFinding,
    SourceRiskScore,
    SourceSafetyAssessment,
    SourceSafetyBatch,
    SourceSafetyWarning,
    TrustBoundaryPolicy,
    model_to_plain,
)
from .poisoning import detect_source_poisoning
from .prompt_injection import detect_prompt_injection
from .risk_scoring import score_source_risk
from .trust_boundary import (
    default_trust_boundary_policy,
    source_trust_boundary_instructions,
    wrap_untrusted_source_content,
)


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def assess_source_text(
    *,
    text: str,
    source: dict[str, Any] | None = None,
    source_id: str | None = None,
    url: str | None = None,
    title: str | None = None,
    mode: SanitizationMode = "quote_suspicious_blocks",
    raw_local_path: str | None = None,
    sanitized_local_path: str | None = None,
) -> SourceSafetyAssessment:
    source = dict(source or {})
    if source_id:
        source["source_id"] = source_id
    if url:
        source["url"] = url
    if title:
        source["title"] = title
    identity = source_identity_from_dict(source)
    resolved_source_id = identity.source_id
    resolved_url = str(source.get("final_url") or source.get("url") or identity.url or "")
    resolved_title = source.get("title") or identity.title
    metadata = dict(source)
    prompt_findings = detect_prompt_injection(
        text,
        source_id=resolved_source_id,
        url=resolved_url,
    )
    poisoning_findings = detect_source_poisoning(
        text,
        source_id=resolved_source_id,
        url=resolved_url,
        title=resolved_title,
        metadata=metadata,
    )
    risk_score = score_source_risk(
        source_id=resolved_source_id,
        url=resolved_url,
        prompt_injection_findings=prompt_findings,
        source_poisoning_findings=poisoning_findings,
        metadata=metadata,
    )
    sanitized = sanitize_source_content(
        text=text,
        source_id=resolved_source_id,
        url=resolved_url,
        title=resolved_title,
        raw_local_path=raw_local_path or source.get("local_path"),
        sanitized_local_path=sanitized_local_path,
        prompt_findings=prompt_findings,
        poisoning_findings=poisoning_findings,
        risk_score=risk_score,
        mode=mode,
    )
    warnings = _warnings_for_assessment(
        source_id=resolved_source_id,
        prompt_findings=prompt_findings,
        poisoning_findings=poisoning_findings,
        risk_score=risk_score,
    )
    return SourceSafetyAssessment(
        source_id=resolved_source_id,
        url=resolved_url,
        final_url=source.get("final_url"),
        title=resolved_title,
        domain=source.get("source_domain") or source_domain(resolved_url),
        generated_at=now_iso_utc(),
        prompt_injection_findings=prompt_findings,
        source_poisoning_findings=poisoning_findings,
        warnings=warnings,
        risk_score=risk_score,
        sanitized_content=sanitized,
        metadata={
            "raw_local_path": raw_local_path or source.get("local_path"),
            "source_kind": source.get("source_kind"),
            "document_kind": source.get("document_kind"),
            "content_type": source.get("content_type"),
        },
    )


def assess_sources(
    *,
    sources: list[dict[str, Any]],
    texts_by_source_id: dict[str, str],
    thread_id: str | None = None,
    question: str = "",
    mode: SanitizationMode = "quote_suspicious_blocks",
    policy: TrustBoundaryPolicy | None = None,
) -> SourceSafetyBatch:
    assessments: list[SourceSafetyAssessment] = []
    for source in sources:
        identity = source_identity_from_dict(source)
        text = texts_by_source_id.get(identity.source_id)
        if text is None:
            text = texts_by_source_id.get(str(source.get("url") or ""), "")
        assessments.append(
            assess_source_text(
                text=text,
                source=source,
                mode=mode,
                raw_local_path=source.get("local_path"),
                sanitized_local_path=source.get("sanitized_local_path"),
            )
        )
    batch = SourceSafetyBatch(
        thread_id=thread_id,
        question=question,
        generated_at=now_iso_utc(),
        policy=policy or default_trust_boundary_policy(),
        assessments=assessments,
    )
    batch.summary = _summary(batch)
    batch.warnings = [
        warning for assessment in assessments for warning in assessment.warnings
    ]
    return batch


def assess_sources_from_manifest(
    *,
    thread_dir: Path,
    thread_id: str,
    question: str = "",
    mode: SanitizationMode = "quote_suspicious_blocks",
) -> SourceSafetyBatch:
    manifest_path = thread_dir / "sources.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        data = []
    if not isinstance(data, list):
        data = []
    sources = [item for item in data if isinstance(item, dict)]
    sanitized_dir = thread_dir / "sanitized_sources"
    sanitized_dir.mkdir(parents=True, exist_ok=True)
    texts: dict[str, str] = {}
    prepared_sources: list[dict[str, Any]] = []
    for source in sources:
        identity = source_identity_from_dict(source)
        raw_path = _safe_source_path(thread_dir, source.get("local_path"))
        text = ""
        if raw_path and raw_path.exists() and raw_path.is_file():
            text = raw_path.read_text(encoding="utf-8", errors="ignore")
        sanitized_rel = f"sanitized_sources/{identity.source_id}.txt"
        prepared = dict(source)
        prepared["sanitized_local_path"] = f"runs/{thread_id}/{sanitized_rel}"
        prepared_sources.append(prepared)
        texts[identity.source_id] = text
    batch = assess_sources(
        sources=prepared_sources,
        texts_by_source_id=texts,
        thread_id=thread_id,
        question=question,
        mode=mode,
    )
    for assessment in batch.assessments:
        rel_path = assessment.sanitized_content.sanitized_local_path
        path = _safe_source_path(thread_dir, rel_path)
        if path is None:
            path = sanitized_dir / f"{assessment.source_id}.txt"
            assessment.sanitized_content.sanitized_local_path = (
                f"runs/{thread_id}/sanitized_sources/{assessment.source_id}.txt"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(assessment.sanitized_content.sanitized_text, encoding="utf-8")
    _annotate_manifest(manifest_path, batch)
    write_source_safety_artifacts(thread_dir, batch)
    return batch


def source_safety_metadata(assessment: SourceSafetyAssessment) -> dict[str, Any]:
    return {
        "risk_level": assessment.risk_score.risk_level,
        "numeric_score": assessment.risk_score.numeric_score,
        "recommended_action": assessment.risk_score.recommended_action,
        "agent_context_allowed": assessment.sanitized_content.agent_context_allowed,
        "report_allowed": assessment.sanitized_content.report_allowed,
        "sanitized_local_path": assessment.sanitized_content.sanitized_local_path,
        "prompt_injection_findings": len(assessment.prompt_injection_findings),
        "source_poisoning_findings": len(assessment.source_poisoning_findings),
        "warning_count": len(assessment.warnings),
        "reasons": assessment.risk_score.reasons,
        "exclusion_reason": assessment.sanitized_content.exclusion_reason,
    }


def _warnings_for_assessment(
    *,
    source_id: str,
    prompt_findings: list[PromptInjectionFinding],
    poisoning_findings: list[SourcePoisoningFinding],
    risk_score: SourceRiskScore,
) -> list[SourceSafetyWarning]:
    warnings: list[SourceSafetyWarning] = []
    for finding in prompt_findings:
        warnings.append(
            SourceSafetyWarning(
                code=f"prompt_injection.{finding.pattern}",
                risk_level=finding.risk_level,
                source_id=source_id,
                message=finding.explanation,
                evidence=finding.matched_text,
                recommended_action=finding.recommended_action,
                explanation="Prompt-injection finding is deterministic and pattern-based.",
            )
        )
    for finding in poisoning_findings:
        warnings.append(
            SourceSafetyWarning(
                code=f"source_poisoning.{finding.category}",
                risk_level=finding.risk_level,
                source_id=source_id,
                message=finding.explanation,
                evidence=finding.evidence,
                recommended_action=finding.recommended_action,
                explanation="Source-poisoning finding is deterministic and heuristic.",
            )
        )
    if risk_score.risk_level in {"high", "critical"}:
        warnings.append(
            SourceSafetyWarning(
                code="source_safety.high_risk_context_control",
                risk_level=risk_score.risk_level,
                source_id=source_id,
                message="High-risk source content must not be passed raw into model context.",
                recommended_action=risk_score.recommended_action,
                explanation="Risk score crossed the model-context safety threshold.",
            )
        )
    return warnings


def _summary(batch: SourceSafetyBatch) -> dict[str, Any]:
    counts = {"none": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}
    excluded: list[str] = []
    for assessment in batch.assessments:
        counts[assessment.risk_score.risk_level] += 1
        if not assessment.sanitized_content.agent_context_allowed:
            excluded.append(assessment.source_id)
    return {
        **counts,
        "source_count": len(batch.assessments),
        "excluded_from_agent_context": excluded,
        "prompt_injection_findings": sum(
            len(a.prompt_injection_findings) for a in batch.assessments
        ),
        "source_poisoning_findings": sum(
            len(a.source_poisoning_findings) for a in batch.assessments
        ),
    }


def _annotate_manifest(path: Path, batch: SourceSafetyBatch) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = []
    if not isinstance(data, list):
        return
    by_id = {assessment.source_id: assessment for assessment in batch.assessments}
    by_url = {assessment.url: assessment for assessment in batch.assessments if assessment.url}
    annotated: list[Any] = []
    for item in data:
        if not isinstance(item, dict):
            annotated.append(item)
            continue
        identity = source_identity_from_dict(item)
        assessment = by_id.get(identity.source_id) or by_url.get(str(item.get("url") or ""))
        if assessment:
            item = dict(item)
            item.setdefault("raw_local_path", item.get("local_path"))
            item["sanitized_local_path"] = assessment.sanitized_content.sanitized_local_path
            item["source_safety"] = source_safety_metadata(assessment)
        annotated.append(item)
    path.write_text(json.dumps(annotated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    try:
        candidate.relative_to(thread_dir.resolve())
    except ValueError:
        return None
    return candidate


__all__ = [
    "SOURCE_SAFETY_ARTIFACTS",
    "PromptInjectionFinding",
    "SanitizedSourceContent",
    "SourcePoisoningFinding",
    "SourceRiskScore",
    "SourceSafetyAssessment",
    "SourceSafetyBatch",
    "SourceSafetyWarning",
    "TrustBoundaryPolicy",
    "assess_source_text",
    "assess_sources",
    "assess_sources_from_manifest",
    "default_trust_boundary_policy",
    "detect_prompt_injection",
    "detect_source_poisoning",
    "model_to_plain",
    "sanitize_source_content",
    "score_source_risk",
    "source_safety_metadata",
    "source_trust_boundary_instructions",
    "wrap_untrusted_source_content",
    "write_source_safety_artifacts",
]
