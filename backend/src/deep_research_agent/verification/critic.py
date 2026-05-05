from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deep_research_agent.evidence.claim_extractor import (
    ClaimInput,
    extract_claims,
    extract_values,
    has_date,
    has_number,
    normalize_claim_text,
)
from deep_research_agent.evidence.ledger import load_source_documents

from .contracts import CriticFinding, VerificationConfig, VerificationTaskType

_CITATION_RE = re.compile(r"\[(S\d+(?:,\s*S\d+)*)\]", re.I)
_ABSOLUTE_RE = re.compile(
    r"\b(always|never|best|must|guarantees?|proves?|undeniably|clearly|definitely|"
    r"zero downtime|no risk|only choice)\b",
    re.I,
)
_FRESHNESS_RE = re.compile(r"\b(latest|current|currently|recent|today|now|newest|as of)\b", re.I)
_ASSUMPTION_RE = re.compile(r"\b(assumes?|assuming|likely|probably|expected to|will)\b", re.I)
_COUNTER_RE = re.compile(
    r"\b(however|although|counterargument|critics?|criticism|opposing|limitation|"
    r"caveat|trade[- ]?off|depends|uncertain)\b",
    re.I,
)
_RECOMMENDATION_RE = re.compile(r"\b(should|must|recommend|recommended|best|avoid|prefer)\b", re.I)


@dataclass(frozen=True)
class CriticInput:
    thread_id: str
    report_text: str
    notes_text: str
    sources: list[dict[str, Any]]
    evidence_ledger: dict[str, Any] | None
    source_audit: dict[str, Any] | None
    context_packs: Any | None
    evaluation: dict[str, Any] | None
    synthesis: dict[str, Any] | None
    source_text_by_id: dict[str, str]
    available_artifacts: list[str]


class ResearchCritic:
    """Deterministic skeptical reviewer over local run artifacts."""

    def load_input(self, run_dir: Path, *, thread_id: str) -> CriticInput:
        source_documents = load_source_documents(run_dir)
        return CriticInput(
            thread_id=thread_id,
            report_text=_read_text(run_dir / "report.md"),
            notes_text=_read_text(run_dir / "notes.md"),
            sources=_read_sources(run_dir / "sources.json"),
            evidence_ledger=_read_json_obj(run_dir / "evidence_ledger.json"),
            source_audit=_read_json_obj(run_dir / "source_audit.json"),
            context_packs=_read_json_any(run_dir / "context_packs.json"),
            evaluation=_read_json_obj(run_dir / "evaluation.json"),
            synthesis=_read_json_obj(run_dir / "synthesis_output.json"),
            source_text_by_id={doc.source.source_id: doc.text for doc in source_documents},
            available_artifacts=sorted(
                path.relative_to(run_dir).as_posix()
                for path in run_dir.rglob("*")
                if path.is_file()
            ),
        )

    def audit(
        self,
        run_dir: Path,
        *,
        thread_id: str,
        config: VerificationConfig | None = None,
    ) -> list[CriticFinding]:
        config = config or VerificationConfig()
        data = self.load_input(run_dir, thread_id=thread_id)
        findings: list[CriticFinding] = []
        report_claims = extract_claims(
            [ClaimInput(origin="report", text=data.report_text, origin_ref="report.md")]
        )
        ledger_claims = _ledger_claims_by_normalized_text(data.evidence_ledger)
        warned_sources = _warned_sources(data.source_audit)
        stale_sources = _stale_sources(data.source_audit)

        for claim in report_claims:
            text = claim.text.strip()
            normalized = normalize_claim_text(text)
            cited = _citation_source_ids(text)
            ledger_claim = ledger_claims.get(normalized)
            support_level = str((ledger_claim or {}).get("support_level") or "")
            confidence = _float((ledger_claim or {}).get("confidence_score"), default=0.35)
            candidate_source_ids = cited or _ledger_source_ids(ledger_claim)

            if not cited:
                findings.append(
                    self._finding(
                        kind="missing_citation",
                        claim=text,
                        reason="Report claim has no explicit source marker.",
                        task_type=VerificationTaskType.VERIFY_UNSUPPORTED_CLAIM,
                        priority=2,
                        severity="medium",
                        candidates=candidate_source_ids,
                        confidence=confidence,
                    )
                )
            if _ABSOLUTE_RE.search(text) and support_level not in {"strong", "source_backed"}:
                findings.append(
                    self._finding(
                        kind="overconfident_claim",
                        claim=text,
                        reason="Strong or absolute wording is not backed by strong evidence.",
                        task_type=VerificationTaskType.VERIFY_UNSUPPORTED_CLAIM,
                        priority=1,
                        severity="high",
                        candidates=candidate_source_ids,
                        confidence=confidence,
                    )
                )
            if config.verify_numbers and has_number(text):
                values_supported = _values_supported(text, ledger_claim)
                weak_support = support_level in {"", "weak", "unsupported"}
                if not cited or weak_support or not values_supported:
                    findings.append(
                        self._finding(
                            kind="weak_numeric_support",
                            claim=text,
                            reason="Numeric claim needs exact value support in local source text.",
                            task_type=VerificationTaskType.VERIFY_NUMERIC_CLAIM,
                            priority=1,
                            severity="high",
                            candidates=candidate_source_ids,
                            confidence=confidence,
                        )
                    )
            if config.verify_dates and has_date(text):
                values_supported = _values_supported(text, ledger_claim)
                weak_support = support_level in {"", "weak", "unsupported"}
                if not cited or weak_support or not values_supported:
                    findings.append(
                        self._finding(
                            kind="weak_date_support",
                            claim=text,
                            reason=(
                                "Date-sensitive claim needs exact date support in "
                                "local sources."
                            ),
                            task_type=VerificationTaskType.VERIFY_DATE_CLAIM,
                            priority=1,
                            severity="high",
                            candidates=candidate_source_ids,
                            confidence=confidence,
                        )
                    )
            if config.verify_recommendations and (
                claim.claim_type == "recommendation" or _RECOMMENDATION_RE.search(text)
            ):
                good_support = support_level in {"strong", "source_backed", "moderate"}
                has_counter = bool(_COUNTER_RE.search(data.report_text))
                if not good_support or not has_counter:
                    findings.append(
                        self._finding(
                            kind="recommendation_without_evidence",
                            claim=text,
                            reason=(
                                "Recommendation needs explicit support and "
                                "limitations or tradeoffs."
                            ),
                            task_type=VerificationTaskType.VERIFY_RECOMMENDATION,
                            priority=2,
                            severity="medium",
                            candidates=candidate_source_ids,
                            confidence=confidence,
                        )
                    )
            if claim.claim_type == "comparative" and not _COUNTER_RE.search(data.report_text):
                findings.append(
                    self._finding(
                        kind="missing_opposing_view",
                        claim=text,
                        reason="Comparative claim lacks counterarguments, caveats, or tradeoffs.",
                        task_type=VerificationTaskType.VERIFY_MISSING_COUNTERARGUMENT,
                        priority=3,
                        severity="medium",
                        candidates=candidate_source_ids,
                        confidence=confidence,
                    )
                )
            if _ASSUMPTION_RE.search(text) and not _COUNTER_RE.search(text):
                findings.append(
                    self._finding(
                        kind="unclear_assumption",
                        claim=text,
                        reason="Assumption or prediction is not clearly scoped with caveats.",
                        task_type=VerificationTaskType.VERIFY_UNSUPPORTED_CLAIM,
                        priority=4,
                        severity="low",
                        candidates=candidate_source_ids,
                        confidence=confidence,
                    )
                )
            if candidate_source_ids and warned_sources.intersection(candidate_source_ids):
                findings.append(
                    self._finding(
                        kind="ignored_source_warning",
                        claim=text,
                        reason="Claim cites a source that source audit flagged for caution.",
                        task_type=VerificationTaskType.VERIFY_PRIMARY_SOURCE_SUPPORT,
                        priority=2,
                        severity="medium",
                        candidates=candidate_source_ids,
                        confidence=confidence,
                    )
                )
            if support_level in {"unsupported", "contradicted"}:
                findings.append(
                    self._finding(
                        kind="claim_source_mismatch",
                        claim=text,
                        reason=f"Evidence ledger marks this claim as {support_level}.",
                        task_type=VerificationTaskType.VERIFY_CONTRADICTION
                        if support_level == "contradicted"
                        else VerificationTaskType.VERIFY_UNSUPPORTED_CLAIM,
                        priority=1 if support_level == "contradicted" else 2,
                        severity="high" if support_level == "contradicted" else "medium",
                        candidates=candidate_source_ids,
                        confidence=confidence,
                    )
                )

        if config.freshness_verification_required and (
            _FRESHNESS_RE.search(data.report_text) or stale_sources
        ):
            findings.append(
                self._finding(
                    kind="stale_source_risk",
                    claim="Verify freshness of current/latest claims and dated sources.",
                    reason=(
                        "Report uses freshness-sensitive wording or source audit "
                        "found stale risk."
                    ),
                    task_type=VerificationTaskType.VERIFY_FRESHNESS,
                    priority=2,
                    severity="medium",
                    candidates=sorted(stale_sources),
                    confidence=0.35,
                )
            )

        if config.contradiction_verification_required:
            for contradiction in _ledger_contradictions(data.evidence_ledger):
                findings.append(
                    self._finding(
                        kind="contradiction_warning",
                        claim=str(contradiction.get("explanation") or "Verify contradiction."),
                        reason="Evidence ledger detected a possible contradiction group.",
                        task_type=VerificationTaskType.VERIFY_CONTRADICTION,
                        priority=1,
                        severity=str(contradiction.get("severity") or "high"),  # type: ignore[arg-type]
                        candidates=[],
                        confidence=0.25,
                        metadata={"contradiction": contradiction},
                    )
                )

        return _dedupe_findings(findings)

    def _finding(
        self,
        *,
        kind: str,
        claim: str,
        reason: str,
        task_type: VerificationTaskType,
        priority: int,
        severity: str,
        candidates: list[str],
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> CriticFinding:
        digest = hashlib.sha1(f"{kind}:{claim}:{reason}".encode("utf-8")).hexdigest()[:10]
        return CriticFinding(
            finding_id=f"CF-{digest}",
            kind=kind,
            severity=_safe_severity(severity),
            priority=max(1, min(5, priority)),
            claim_or_question=claim,
            source_artifact="report.md",
            reason=reason,
            expected_evidence_type=_expected_evidence_type(task_type),
            suggested_task_type=task_type,
            candidate_source_ids=sorted(set(candidates)),
            confidence_before=confidence,
            metadata=metadata or {},
        )


def _expected_evidence_type(task_type: VerificationTaskType) -> str:
    if task_type in {
        VerificationTaskType.VERIFY_NUMERIC_CLAIM,
        VerificationTaskType.VERIFY_DATE_CLAIM,
    }:
        return "exact_value_match"
    if task_type == VerificationTaskType.VERIFY_PRIMARY_SOURCE_SUPPORT:
        return "primary_or_high_authority_source"
    if task_type == VerificationTaskType.VERIFY_FRESHNESS:
        return "dated_current_source"
    return "source_text"


def _safe_severity(severity: str) -> str:
    if severity in {"info", "low", "medium", "high", "critical"}:
        return severity
    return "medium"


def _read_text(path: Path) -> str:
    try:
        if not path.exists() or path.is_dir():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


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


def _read_sources(path: Path) -> list[dict[str, Any]]:
    value = _read_json_any(path)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("sources"), list):
        return [item for item in value["sources"] if isinstance(item, dict)]
    return []


def _citation_source_ids(text: str) -> list[str]:
    out: list[str] = []
    for match in _CITATION_RE.finditer(text):
        out.extend(item.strip().upper() for item in match.group(1).split(","))
    return sorted(set(out))


def _ledger_claims_by_normalized_text(ledger: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not ledger:
        return {}
    out: dict[str, dict[str, Any]] = {}
    claims = ledger.get("claims")
    if not isinstance(claims, list):
        return out
    for item in claims:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        normalized = str(item.get("normalized_text") or normalize_claim_text(text))
        if normalized and item.get("origin") == "report":
            out[normalized] = item
    return out


def _ledger_source_ids(claim: dict[str, Any] | None) -> list[str]:
    if not claim:
        return []
    source_ids = claim.get("source_ids")
    if isinstance(source_ids, list):
        return sorted(str(item) for item in source_ids if item)
    citations = claim.get("citations")
    if isinstance(citations, list):
        return sorted(
            str(item.get("source_id"))
            for item in citations
            if isinstance(item, dict) and item.get("source_id")
        )
    return []


def _values_supported(claim_text: str, ledger_claim: dict[str, Any] | None) -> bool:
    values = [value.lower() for value in extract_values(claim_text)]
    if not values:
        return True
    if not ledger_claim:
        return False
    citations = ledger_claim.get("citations")
    if not isinstance(citations, list):
        return False
    matched = {
        str(value).lower()
        for citation in citations
        if isinstance(citation, dict) and _float(citation.get("score"), default=0.0) >= 0.34
        for value in citation.get("value_matches", [])
    }
    return all(value in matched for value in values)


def _warned_sources(source_audit: dict[str, Any] | None) -> set[str]:
    out: set[str] = set()
    for audit in _audit_items(source_audit):
        warnings = audit.get("warnings")
        recommended = str(audit.get("recommended_usage") or "")
        warned_usage = {
            "use_with_caution",
            "verify_with_primary_source",
            "exclude_from_report",
        }
        if warnings or recommended in warned_usage:
            sid = str(audit.get("source_id") or "")
            if sid:
                out.add(sid)
    return out


def _stale_sources(source_audit: dict[str, Any] | None) -> set[str]:
    out: set[str] = set()
    for audit in _audit_items(source_audit):
        freshness = audit.get("freshness_score")
        status = str((freshness or {}).get("status") if isinstance(freshness, dict) else "")
        if status in {"possibly_stale", "stale", "unknown"}:
            sid = str(audit.get("source_id") or "")
            if sid:
                out.add(sid)
    return out


def _audit_items(source_audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not source_audit or not isinstance(source_audit.get("audits"), list):
        return []
    return [item for item in source_audit["audits"] if isinstance(item, dict)]


def _ledger_contradictions(ledger: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not ledger or not isinstance(ledger.get("contradictions"), list):
        return []
    return [item for item in ledger["contradictions"] if isinstance(item, dict)]


def _dedupe_findings(findings: list[CriticFinding]) -> list[CriticFinding]:
    seen: set[tuple[str, str]] = set()
    out: list[CriticFinding] = []
    for finding in findings:
        key = (finding.kind, normalize_claim_text(finding.claim_or_question))
        if key in seen:
            continue
        seen.add(key)
        out.append(finding)
    out.sort(key=lambda item: (item.priority, item.finding_id))
    return out


def _float(value: Any, *, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default
