from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from deep_research_agent.evidence.citation_mapper import SourceDocument, map_claim_citations
from deep_research_agent.evidence.claim_extractor import ClaimInput, extract_claims, extract_values
from deep_research_agent.evidence.contracts import ExtractedClaim
from deep_research_agent.evidence.ledger import load_source_documents

from .contracts import (
    VerificationEvidence,
    VerificationFinding,
    VerificationPlan,
    VerificationResult,
    VerificationTask,
    VerificationTaskStatus,
    VerificationTaskType,
)

_COUNTER_RE = re.compile(
    r"\b(however|although|limitation|caveat|counterargument|critics?|opposing|"
    r"trade[- ]?off|depends|uncertain|risk|drawback)\b",
    re.I,
)
_DATED_RE = re.compile(r"\b(20\d{2}|19\d{2}|updated|published|fetched|as of)\b", re.I)
_STRONG_RE = re.compile(
    r"\b(always|never|best|must|guarantees?|proves?|zero downtime|no risk|only)\b",
    re.I,
)


class DeterministicVerifier:
    """Runs offline verification tasks against local artifacts only."""

    def run_plan(self, run_dir: Path, plan: VerificationPlan) -> list[VerificationResult]:
        source_documents = load_source_documents(run_dir)
        source_audit = _read_json_obj(run_dir / "source_audit.json")
        evidence_ledger = _read_json_obj(run_dir / "evidence_ledger.json")
        return [
            self.run_task(
                run_dir,
                task,
                source_documents=source_documents,
                source_audit=source_audit,
                evidence_ledger=evidence_ledger,
            )
            for task in plan.tasks
        ]

    def run_task(
        self,
        run_dir: Path,
        task: VerificationTask,
        *,
        source_documents: list[SourceDocument] | None = None,
        source_audit: dict[str, Any] | None = None,
        evidence_ledger: dict[str, Any] | None = None,
    ) -> VerificationResult:
        source_documents = (
            source_documents if source_documents is not None else load_source_documents(run_dir)
        )
        source_audit = (
            source_audit
            if source_audit is not None
            else _read_json_obj(run_dir / "source_audit.json")
        )
        evidence_ledger = (
            evidence_ledger
            if evidence_ledger is not None
            else _read_json_obj(run_dir / "evidence_ledger.json")
        )
        sources = _filter_sources(source_documents, task.candidate_source_ids)
        if not sources:
            return self._result(
                task,
                VerificationTaskStatus.NOT_ENOUGH_INFORMATION,
                ["No local source text was available for this task."],
                [],
            )

        if task.task_type == VerificationTaskType.VERIFY_CONTRADICTION:
            return self._verify_contradiction(task, evidence_ledger, sources)
        if task.task_type == VerificationTaskType.VERIFY_FRESHNESS:
            return self._verify_freshness(task, source_audit, sources)
        if task.task_type == VerificationTaskType.VERIFY_PRIMARY_SOURCE_SUPPORT:
            return self._verify_primary_source_support(task, source_audit, sources)
        if task.task_type == VerificationTaskType.VERIFY_MISSING_COUNTERARGUMENT:
            return self._verify_counterargument(task, sources)

        claim = _claim_from_task(task)
        citation_map, _quotes = map_claim_citations([claim], sources, threshold=0.20)
        citations = citation_map.get(claim.claim_id, [])
        evidence = [_evidence_from_citation(task.task_id, citation) for citation in citations[:4]]
        good = [citation for citation in citations if citation.score >= 0.34]
        values = extract_values(task.claim_or_question)
        exact_values_supported = _exact_values_supported(values, good)

        if task.task_type in {
            VerificationTaskType.VERIFY_NUMERIC_CLAIM,
            VerificationTaskType.VERIFY_DATE_CLAIM,
        }:
            if good and exact_values_supported:
                return self._result(
                    task,
                    VerificationTaskStatus.VERIFIED,
                    ["Exact numeric/date values were found in supporting source context."],
                    evidence,
                )
            if good:
                return self._result(
                    task,
                    VerificationTaskStatus.PARTIALLY_VERIFIED,
                    ["Related source context was found, but exact values were not fully matched."],
                    evidence,
                )
            return self._result(
                task,
                VerificationTaskStatus.UNSUPPORTED,
                ["No local source context matched this numeric/date claim above threshold."],
                evidence,
            )

        if task.task_type == VerificationTaskType.VERIFY_RECOMMENDATION:
            has_counter_context = bool(_COUNTER_RE.search(" ".join(doc.text for doc in sources)))
            if good and has_counter_context:
                return self._result(
                    task,
                    VerificationTaskStatus.VERIFIED,
                    [
                        "Recommendation has source support and nearby tradeoff or "
                        "limitation context."
                    ],
                    evidence,
                )
            if good:
                return self._result(
                    task,
                    VerificationTaskStatus.PARTIALLY_VERIFIED,
                    ["Recommendation has related support but lacks clear tradeoff context."],
                    evidence,
                )
            return self._result(
                task,
                VerificationTaskStatus.UNSUPPORTED,
                ["Recommendation was not supported by local source context."],
                evidence,
            )

        if task.task_type == VerificationTaskType.VERIFY_UNSUPPORTED_CLAIM:
            if good and not _STRONG_RE.search(task.claim_or_question):
                return self._result(
                    task,
                    VerificationTaskStatus.VERIFIED,
                    ["Claim has source support and does not use absolute language."],
                    evidence,
                )
            if good:
                return self._result(
                    task,
                    VerificationTaskStatus.PARTIALLY_VERIFIED,
                    [
                        "Related support exists, but strong wording is not fully justified "
                        "by local evidence."
                    ],
                    evidence,
                )
            return self._result(
                task,
                VerificationTaskStatus.UNSUPPORTED,
                ["No local source support was found for this challenged claim."],
                evidence,
            )

        if good:
            return self._result(
                task,
                VerificationTaskStatus.VERIFIED,
                ["Claim has local source support above verification threshold."],
                evidence,
            )
        if citations:
            return self._result(
                task,
                VerificationTaskStatus.PARTIALLY_VERIFIED,
                ["Only weak lexical support was found in local source context."],
                evidence,
            )
        return self._result(
            task,
            VerificationTaskStatus.UNSUPPORTED,
            ["No local source support was found."],
            evidence,
        )

    def _verify_contradiction(
        self,
        task: VerificationTask,
        evidence_ledger: dict[str, Any] | None,
        sources: list[SourceDocument],
    ) -> VerificationResult:
        contradictions = evidence_ledger.get("contradictions", []) if evidence_ledger else []
        if isinstance(contradictions, list) and contradictions:
            evidence = [
                VerificationEvidence(
                    evidence_id=_evidence_id(task.task_id, "evidence_ledger", str(idx)),
                    source_artifact="evidence_ledger.json",
                    excerpt=str(item.get("explanation") or item)[:700]
                    if isinstance(item, dict)
                    else str(item)[:700],
                    evidence_type="contradiction_group",
                    polarity="contradicts",
                    score=0.85,
                    reason="Evidence ledger reported a contradiction group.",
                )
                for idx, item in enumerate(contradictions[:3], start=1)
            ]
            return self._result(
                task,
                VerificationTaskStatus.CONTRADICTED,
                ["A local contradiction artifact reported conflicting claims."],
                evidence,
            )
        claim = _claim_from_task(task)
        citation_map, _quotes = map_claim_citations([claim], sources, threshold=0.20)
        evidence = [
            _evidence_from_citation(task.task_id, citation)
            for citation in citation_map.get(claim.claim_id, [])[:3]
        ]
        if evidence:
            return self._result(
                task,
                VerificationTaskStatus.PARTIALLY_VERIFIED,
                ["No contradiction artifact was present, but related source context exists."],
                evidence,
            )
        return self._result(
            task,
            VerificationTaskStatus.NOT_ENOUGH_INFORMATION,
            ["No contradiction artifact or relevant source context was available."],
            [],
        )

    def _verify_freshness(
        self,
        task: VerificationTask,
        source_audit: dict[str, Any] | None,
        sources: list[SourceDocument],
    ) -> VerificationResult:
        audit_items = _audit_items(source_audit)
        stale = [
            item
            for item in audit_items
            if str((item.get("freshness_score") or {}).get("status"))
            in {"stale", "possibly_stale", "unknown"}
        ]
        dated_sources = [
            doc
            for doc in sources
            if doc.source.fetched_at
            or _DATED_RE.search(doc.text[:3000] + " " + (doc.source.title or ""))
        ]
        evidence = [
            VerificationEvidence(
                evidence_id=_evidence_id(task.task_id, "freshness", str(idx)),
                source_id=str(item.get("source_id") or ""),
                source_artifact="source_audit.json",
                excerpt=str((item.get("freshness_score") or {}).get("reasons") or "")[:700],
                evidence_type="source_audit_freshness",
                polarity="context",
                score=0.7,
                reason="Source audit freshness assessment.",
            )
            for idx, item in enumerate(audit_items[:4], start=1)
        ]
        if stale:
            return self._result(
                task,
                VerificationTaskStatus.PARTIALLY_VERIFIED,
                ["Freshness risk remains because at least one source is stale or unknown."],
                evidence,
            )
        if dated_sources:
            return self._result(
                task,
                VerificationTaskStatus.VERIFIED,
                ["Available sources include dated or fetched context and no stale audit warning."],
                evidence,
            )
        return self._result(
            task,
            VerificationTaskStatus.NOT_ENOUGH_INFORMATION,
            ["Sources do not expose enough date metadata to verify freshness."],
            evidence,
        )

    def _verify_primary_source_support(
        self,
        task: VerificationTask,
        source_audit: dict[str, Any] | None,
        sources: list[SourceDocument],
    ) -> VerificationResult:
        audit_items = _audit_items(source_audit)
        primary_ids = {
            str(item.get("source_id"))
            for item in audit_items
            if _float((item.get("primary_source_likelihood") or {}).get("likelihood")) >= 0.65
            or str((item.get("authority_score") or {}).get("source_role")) == "primary"
        }
        evidence = [
            VerificationEvidence(
                evidence_id=_evidence_id(task.task_id, "primary", doc.source.source_id),
                source_id=doc.source.source_id,
                source_title=doc.source.title,
                source_url=doc.source.final_url or doc.source.url,
                source_artifact=doc.source.local_path,
                excerpt=doc.text[:700],
                evidence_type="primary_source_candidate",
                polarity="supports" if doc.source.source_id in primary_ids else "context",
                score=0.8 if doc.source.source_id in primary_ids else 0.35,
                reason="Source audit primary-source likelihood or local source context.",
            )
            for doc in sources[:4]
        ]
        if primary_ids.intersection({doc.source.source_id for doc in sources}):
            return self._result(
                task,
                VerificationTaskStatus.VERIFIED,
                ["At least one candidate source is classified as primary or high-authority."],
                evidence,
            )
        if audit_items:
            return self._result(
                task,
                VerificationTaskStatus.UNSUPPORTED,
                ["Source audit did not identify primary-source support for this claim."],
                evidence,
            )
        return self._result(
            task,
            VerificationTaskStatus.NOT_ENOUGH_INFORMATION,
            ["No source audit artifact was available to judge primary-source support."],
            evidence,
        )

    def _verify_counterargument(
        self,
        task: VerificationTask,
        sources: list[SourceDocument],
    ) -> VerificationResult:
        matches = [
            doc
            for doc in sources
            if _COUNTER_RE.search(doc.text) or _COUNTER_RE.search(doc.source.title or "")
        ]
        evidence = [
            VerificationEvidence(
                evidence_id=_evidence_id(task.task_id, "counter", doc.source.source_id),
                source_id=doc.source.source_id,
                source_title=doc.source.title,
                source_url=doc.source.final_url or doc.source.url,
                source_artifact=doc.source.local_path,
                excerpt=_counter_excerpt(doc.text),
                evidence_type="counterargument_context",
                polarity="supports",
                score=0.65,
                reason="Source text includes caveat, limitation, risk, or tradeoff language.",
            )
            for doc in matches[:3]
        ]
        if matches:
            return self._result(
                task,
                VerificationTaskStatus.PARTIALLY_VERIFIED,
                [
                    "Counterargument context exists in sources but should be reflected "
                    "in the report."
                ],
                evidence,
            )
        return self._result(
            task,
            VerificationTaskStatus.UNSUPPORTED,
            ["No local source context for opposing views or tradeoffs was found."],
            [],
        )

    def _result(
        self,
        task: VerificationTask,
        status: VerificationTaskStatus,
        reasons: list[str],
        evidence: list[VerificationEvidence],
    ) -> VerificationResult:
        confidence_after = _confidence_after(task.confidence_before, status, evidence)
        finding = VerificationFinding(
            finding_id="VF-"
            + hashlib.sha1(f"{task.task_id}:{status.value}".encode("utf-8")).hexdigest()[:10],
            task_id=task.task_id,
            status=status,
            severity=_severity_for_status(status, task.priority),
            claim_or_question=task.claim_or_question,
            explanation=" ".join(reasons),
            evidence=evidence,
            confidence_delta=round(confidence_after - task.confidence_before, 3),
            suggested_action=_suggested_action(status),
        )
        return VerificationResult(
            task_id=task.task_id,
            status=status,
            confidence_before=task.confidence_before,
            confidence_after=confidence_after,
            findings=[finding],
            evidence=evidence,
            reasons=reasons,
        )


def _claim_from_task(task: VerificationTask) -> ExtractedClaim:
    claims = extract_claims(
        [
            ClaimInput(
                origin="report",
                text=task.claim_or_question,
                origin_ref=task.source_artifact,
                source_ids=tuple(task.candidate_source_ids),
            )
        ]
    )
    if claims:
        return claims[0]
    digest = hashlib.sha1(task.claim_or_question.encode("utf-8")).hexdigest()[:10]
    return ExtractedClaim(
        claim_id=f"C-{digest}",
        text=task.claim_or_question,
        normalized_text=task.claim_or_question.lower(),
        claim_type="factual",
        origin="report",
        origin_ref=task.source_artifact,
        source_ids=task.candidate_source_ids,
    )


def _filter_sources(
    source_documents: list[SourceDocument],
    candidate_source_ids: list[str],
) -> list[SourceDocument]:
    if not candidate_source_ids:
        return source_documents
    candidates = {sid.upper() for sid in candidate_source_ids}
    selected = [doc for doc in source_documents if doc.source.source_id.upper() in candidates]
    return selected or source_documents


def _evidence_from_citation(task_id: str, citation: Any) -> VerificationEvidence:
    score = _float(getattr(citation, "score", 0.0))
    polarity = "supports" if score >= 0.34 else "partially_supports"
    return VerificationEvidence(
        evidence_id=_evidence_id(
            task_id,
            getattr(citation, "source_id", ""),
            getattr(citation, "quote_id", ""),
        ),
        source_id=getattr(citation, "source_id", None),
        source_title=getattr(citation, "title", None),
        source_url=getattr(citation, "url", None),
        excerpt=getattr(citation, "matched_text", "")[:700],
        evidence_type="source_text_match",
        polarity=polarity,
        score=score,
        matched_values=list(getattr(citation, "value_matches", []) or []),
        reason=getattr(citation, "reason", ""),
    )


def _exact_values_supported(values: list[str], citations: list[Any]) -> bool:
    if not values:
        return True
    matched = {
        str(value).lower()
        for citation in citations
        for value in getattr(citation, "value_matches", []) or []
    }
    return all(value.lower() in matched for value in values)


def _confidence_after(
    before: float,
    status: VerificationTaskStatus,
    evidence: list[VerificationEvidence],
) -> float:
    delta = {
        VerificationTaskStatus.VERIFIED: 0.22,
        VerificationTaskStatus.PARTIALLY_VERIFIED: 0.04,
        VerificationTaskStatus.CONTRADICTED: -0.34,
        VerificationTaskStatus.UNSUPPORTED: -0.24,
        VerificationTaskStatus.NOT_ENOUGH_INFORMATION: -0.10,
        VerificationTaskStatus.SKIPPED: 0.0,
        VerificationTaskStatus.PENDING: 0.0,
    }[status]
    if len({item.source_id for item in evidence if item.source_id}) >= 2:
        delta += 0.05
    return round(max(0.0, min(1.0, before + delta)), 3)


def _severity_for_status(status: VerificationTaskStatus, priority: int) -> str:
    if status == VerificationTaskStatus.CONTRADICTED:
        return "critical" if priority == 1 else "high"
    if status == VerificationTaskStatus.UNSUPPORTED:
        return "high" if priority <= 2 else "medium"
    if status == VerificationTaskStatus.NOT_ENOUGH_INFORMATION:
        return "medium"
    return "low"


def _suggested_action(status: VerificationTaskStatus) -> str:
    if status == VerificationTaskStatus.VERIFIED:
        return "Keep the claim, but retain citations and source traceability."
    if status == VerificationTaskStatus.PARTIALLY_VERIFIED:
        return "Qualify the claim and cite the closest supporting context."
    if status == VerificationTaskStatus.CONTRADICTED:
        return "Do not finalize without resolving the conflicting source evidence."
    if status == VerificationTaskStatus.UNSUPPORTED:
        return "Remove, rewrite, or mark the claim as unverified."
    if status == VerificationTaskStatus.NOT_ENOUGH_INFORMATION:
        return "Add targeted sources before relying on this claim."
    return "No action."


def _audit_items(source_audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not source_audit or not isinstance(source_audit.get("audits"), list):
        return []
    return [item for item in source_audit["audits"] if isinstance(item, dict)]


def _read_json_obj(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists() or path.is_dir():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except Exception:
        return None


def _counter_excerpt(text: str) -> str:
    match = _COUNTER_RE.search(text)
    if not match:
        return text[:700]
    start = max(0, match.start() - 240)
    end = min(len(text), match.end() + 420)
    return text[start:end].strip()


def _evidence_id(task_id: str, source_id: str, value: str) -> str:
    return "VE-" + hashlib.sha1(f"{task_id}:{source_id}:{value}".encode("utf-8")).hexdigest()[:10]


def _float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0
