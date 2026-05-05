from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from deep_research_agent.artifacts import now_iso_utc

from .contracts import ConfidenceCalibration, VerificationResult, VerificationTaskStatus


class ConfidenceCalibrator:
    """Conservative confidence rules for the whole report and verification tasks."""

    def calibrate(
        self,
        run_dir: Path,
        *,
        thread_id: str,
        results: list[VerificationResult],
    ) -> ConfidenceCalibration:
        evidence_ledger = _read_json_obj(run_dir / "evidence_ledger.json")
        source_audit = _read_json_obj(run_dir / "source_audit.json")
        sources = _read_sources(run_dir / "sources.json")
        evaluation = _read_json_obj(run_dir / "evaluation.json")

        score = _base_report_confidence(evidence_ledger, evaluation, sources)
        before = score
        factors: list[str] = []
        penalties: list[str] = []

        unsupported_count = _count_status(results, VerificationTaskStatus.UNSUPPORTED)
        contradiction_count = _count_status(results, VerificationTaskStatus.CONTRADICTED)
        stale_count = _stale_source_count(source_audit)
        primary_count = _primary_source_count(source_audit)
        exact_value_count = _exact_value_support_count(results)
        diversity = _source_diversity_score(sources)

        if unsupported_count:
            penalty = min(0.30, unsupported_count * 0.06)
            score -= penalty
            penalties.append(f"{unsupported_count} unsupported verification task(s).")
        if contradiction_count:
            penalty = min(0.36, contradiction_count * 0.12)
            score -= penalty
            penalties.append(f"{contradiction_count} contradiction warning(s).")
        if stale_count:
            penalty = min(0.18, stale_count * 0.04)
            score -= penalty
            penalties.append(f"{stale_count} source(s) have stale or unknown freshness risk.")
        biased_count = _biased_source_count(source_audit)
        if biased_count:
            score -= min(0.16, biased_count * 0.04)
            penalties.append(f"{biased_count} source(s) have medium/high bias risk.")

        verified_count = _count_status(results, VerificationTaskStatus.VERIFIED)
        partial_count = _count_status(results, VerificationTaskStatus.PARTIALLY_VERIFIED)
        if verified_count:
            score += min(0.18, verified_count * 0.035)
            factors.append(f"{verified_count} verification task(s) were verified.")
        if partial_count:
            score += min(0.06, partial_count * 0.01)
            factors.append(f"{partial_count} verification task(s) were partially verified.")
        if primary_count >= 2:
            score += 0.10
            factors.append("Multiple primary/high-authority source signals were present.")
        elif primary_count == 1:
            score += 0.04
            factors.append("One primary/high-authority source signal was present.")
        if exact_value_count:
            score += min(0.12, exact_value_count * 0.035)
            factors.append(f"{exact_value_count} numeric/date task(s) had exact value support.")
        if diversity >= 0.7:
            score += 0.07
            factors.append("Source diversity is moderate to high.")
        elif sources:
            score -= 0.05
            penalties.append("Source diversity is low.")

        score = round(max(0.0, min(1.0, score)), 3)
        task_confidences = {result.task_id: result.confidence_after for result in results}
        finding_confidences = {
            finding.finding_id: result.confidence_after
            for result in results
            for finding in result.findings
        }
        calibration_id = "CC-" + hashlib.sha1(
            f"{thread_id}:{before}:{score}:{len(results)}".encode("utf-8")
        ).hexdigest()[:12]
        return ConfidenceCalibration(
            calibration_id=calibration_id,
            thread_id=thread_id,
            generated_at=now_iso_utc(),
            report_confidence_before=round(before, 3),
            report_confidence_after=score,
            task_confidences=task_confidences,
            finding_confidences=finding_confidences,
            factors=factors or ["No positive calibration factors were detected."],
            penalties=penalties,
            source_diversity_score=diversity,
            unsupported_claim_count=unsupported_count,
            contradiction_count=contradiction_count,
            stale_source_count=stale_count,
            primary_source_support_count=primary_count,
            exact_value_support_count=exact_value_count,
            confidence_label=_label(score),
        )


def _base_report_confidence(
    evidence_ledger: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    sources: list[dict[str, Any]],
) -> float:
    if evidence_ledger:
        coverage = evidence_ledger.get("coverage") or {}
        if isinstance(coverage, dict) and coverage.get("average_confidence") is not None:
            return max(0.2, min(0.72, _float(coverage.get("average_confidence"), 0.45)))
    if evaluation and evaluation.get("overall_score") is not None:
        return max(0.2, min(0.65, _float(evaluation.get("overall_score"), 0.45)))
    return 0.42 if sources else 0.25


def _count_status(results: list[VerificationResult], status: VerificationTaskStatus) -> int:
    return len([result for result in results if result.status == status])


def _stale_source_count(source_audit: dict[str, Any] | None) -> int:
    return len(
        [
            item
            for item in _audit_items(source_audit)
            if str((item.get("freshness_score") or {}).get("status"))
            in {"stale", "possibly_stale", "unknown"}
        ]
    )


def _biased_source_count(source_audit: dict[str, Any] | None) -> int:
    return len(
        [
            item
            for item in _audit_items(source_audit)
            if str((item.get("bias_risk_score") or {}).get("risk_level")) in {"medium", "high"}
        ]
    )


def _primary_source_count(source_audit: dict[str, Any] | None) -> int:
    return len(
        [
            item
            for item in _audit_items(source_audit)
            if _float((item.get("primary_source_likelihood") or {}).get("likelihood"), 0.0)
            >= 0.65
            or str((item.get("authority_score") or {}).get("source_role")) == "primary"
        ]
    )


def _exact_value_support_count(results: list[VerificationResult]) -> int:
    return len(
        [
            result
            for result in results
            if result.status == VerificationTaskStatus.VERIFIED
            and any(item.matched_values for item in result.evidence)
        ]
    )


def _source_diversity_score(sources: list[dict[str, Any]]) -> float:
    domains = {
        _domain(str(source.get("final_url") or source.get("url") or ""))
        for source in sources
        if source.get("ok", True) is not False
    }
    domains.discard("")
    if not domains:
        return 0.0
    if len(domains) == 1:
        return 0.35
    if len(domains) == 2:
        return 0.7
    return 1.0


def _read_sources(path: Path) -> list[dict[str, Any]]:
    value = _read_json_any(path)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("sources"), list):
        return [item for item in value["sources"] if isinstance(item, dict)]
    return []


def _read_json_obj(path: Path) -> dict[str, Any] | None:
    value = _read_json_any(path)
    return value if isinstance(value, dict) else None


def _read_json_any(path: Path) -> Any | None:
    try:
        if not path.exists() or path.is_dir():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _audit_items(source_audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not source_audit or not isinstance(source_audit.get("audits"), list):
        return []
    return [item for item in source_audit["audits"] if isinstance(item, dict)]


def _float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def _domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _label(score: float) -> str:
    if score >= 0.76:
        return "high"
    if score >= 0.56:
        return "medium"
    if score >= 0.34:
        return "low"
    return "very_low"
