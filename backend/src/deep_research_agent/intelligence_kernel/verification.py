from __future__ import annotations

from pathlib import Path

from .contracts import (
    CritiqueFinding,
    EvidenceUnit,
    ResearchClaim,
    ResearchKernelSettings,
    SourceUnit,
    VerificationTask,
    clamp_score,
    stable_id,
    tokenize,
    write_json,
)


def _task_type(claim: ResearchClaim, finding: CritiqueFinding | None = None) -> str:
    if finding and finding.category == "missing_primary_source":
        return "verify_primary_source_support"
    if finding and finding.category == "missing_counterargument":
        return "verify_missing_counterargument"
    if finding and finding.category == "stale_source":
        return "verify_freshness"
    if claim.claim_type == "numeric":
        return "verify_numeric_claim"
    if claim.claim_type == "temporal":
        return "verify_date_claim"
    if claim.claim_type == "comparative":
        return "verify_comparative_claim"
    if claim.claim_type == "recommendation":
        return "verify_recommendation"
    if finding and finding.category == "unsupported_claim":
        return "verify_unsupported_claim"
    return "verify_entity_claim"


def generate_verification_tasks(
    claims: list[ResearchClaim],
    findings: list[CritiqueFinding],
    evidence_units: list[EvidenceUnit],
    settings: ResearchKernelSettings | None = None,
) -> list[VerificationTask]:
    settings = settings or ResearchKernelSettings()
    by_claim: dict[str, list[CritiqueFinding]] = {}
    for finding in findings:
        for claim_id in finding.affected_claims:
            by_claim.setdefault(claim_id, []).append(finding)
    tasks: list[VerificationTask] = []
    for claim in claims:
        relevant = by_claim.get(claim.claim_id, [])
        priority = 3
        if claim.source_artifact == "report.md":
            priority -= 1
        if claim.strength in {"strong", "absolute"} or claim.claim_type in {
            "numeric",
            "temporal",
            "recommendation",
            "legal_policy",
            "medical_health",
            "financial",
        }:
            priority -= 1
        if any(f.severity in {"high", "critical"} for f in relevant):
            priority = 1
        priority = max(1, min(5, priority))
        candidates = _candidate_evidence_ids(claim, evidence_units, limit=8)
        finding = relevant[0] if relevant else None
        if (
            not relevant
            and claim.confidence_score >= 0.72
            and claim.claim_type
            not in {
                "numeric",
                "temporal",
                "recommendation",
                "legal_policy",
                "medical_health",
                "financial",
            }
        ):
            continue
        tasks.append(
            VerificationTask(
                task_id=stable_id("vt", claim.claim_id, _task_type(claim, finding)),
                claim_id=claim.claim_id,
                task_type=_task_type(claim, finding),
                priority=priority,
                reason=finding.message if finding else f"Verify {claim.claim_type} claim.",
                expected_evidence=claim.numbers + claim.dates + claim.entities[:6],
                candidate_evidence_ids=candidates,
                confidence_before=claim.confidence_score,
            )
        )
    tasks.sort(key=lambda task: (task.priority, task.task_type, task.claim_id))
    return tasks[: settings.max_claims_to_verify]


def _candidate_evidence_ids(
    claim: ResearchClaim, evidence_units: list[EvidenceUnit], limit: int
) -> list[str]:
    claim_terms = tokenize(claim.normalized_text)
    scored: list[tuple[str, float]] = []
    for evidence in evidence_units:
        overlap = len(claim_terms & tokenize(evidence.normalized_text)) / max(1, len(claim_terms))
        score = overlap + (0.3 if set(claim.numbers) & set(evidence.numbers) else 0.0)
        score += (
            0.25
            if set(d.lower() for d in claim.dates) & set(d.lower() for d in evidence.dates)
            else 0.0
        )
        score += evidence.support_score * 0.18
        if score > 0.18:
            scored.append((evidence.evidence_unit_id, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in scored[:limit]]


def run_verification_tasks(
    tasks: list[VerificationTask],
    claims: list[ResearchClaim],
    evidence_units: list[EvidenceUnit],
    source_units: list[SourceUnit],
) -> list[VerificationTask]:
    claims_by_id = {claim.claim_id: claim for claim in claims}
    evidence_by_id = {e.evidence_unit_id: e for e in evidence_units}
    source_by_id = {s.source_id: s for s in source_units}
    for task in tasks:
        claim = claims_by_id.get(task.claim_id)
        if claim is None:
            task.status = "failed"
            task.result = "Claim was not found."
            continue
        candidates = [
            evidence_by_id[eid] for eid in task.candidate_evidence_ids if eid in evidence_by_id
        ]
        if not candidates:
            task.status = "unsupported"
            task.result = "No candidate evidence was found."
            task.confidence_after = clamp_score(task.confidence_before * 0.5)
            claim.verification_status = task.status
            claim.confidence_score = task.confidence_after
            continue
        best_score = 0.0
        contradiction_score = 0.0
        claim_terms = tokenize(claim.normalized_text)
        exact_numeric_supported = not claim.numbers
        exact_date_supported = not claim.dates
        for evidence in candidates:
            overlap = len(claim_terms & tokenize(evidence.normalized_text)) / max(
                1, len(claim_terms)
            )
            numeric_match = (
                bool(set(claim.numbers) & set(evidence.numbers)) if claim.numbers else True
            )
            date_match = (
                bool(set(d.lower() for d in claim.dates) & set(d.lower() for d in evidence.dates))
                if claim.dates
                else True
            )
            if evidence.source_id:
                exact_numeric_supported = exact_numeric_supported or numeric_match
                exact_date_supported = exact_date_supported or date_match
            trust_boost = 0.0
            if evidence.source_id and evidence.source_id in source_by_id:
                source = source_by_id[evidence.source_id]
                trust_boost = {
                    "high": 0.18,
                    "medium": 0.1,
                    "unknown": 0.02,
                    "low": -0.08,
                    "risky": -0.25,
                }[source.trust_level]
                if source.source_role == "primary_evidence":
                    trust_boost += 0.08
                if source.source_role == "duplicate":
                    trust_boost -= 0.08
            elif not evidence.source_id:
                trust_boost = -0.12
            score = overlap * 0.58 + evidence.support_score * 0.24 + trust_boost
            if claim.numbers and not numeric_match:
                score -= 0.28
                contradiction_score = max(contradiction_score, 0.45)
            if claim.dates and not date_match:
                score -= 0.22
            best_score = max(best_score, score)
        if claim.numbers and not exact_numeric_supported:
            task.status = "contradicted" if best_score >= 0.25 else "unsupported"
            task.result = "No candidate evidence contains the claim's numeric value."
        elif claim.dates and not exact_date_supported:
            task.status = "not_enough_information"
            task.result = "No candidate evidence contains the claim's date or freshness value."
        elif contradiction_score >= 0.4 and best_score < 0.45:
            task.status = "contradicted"
            task.result = "Candidate evidence overlaps the claim but does not match required numeric/date values."
        elif best_score >= 0.72:
            task.status = "verified"
            task.result = "Local evidence strongly supports the claim."
        elif best_score >= 0.42:
            task.status = "partially_verified"
            task.result = "Local evidence partially supports the claim."
        elif best_score >= 0.22:
            task.status = "not_enough_information"
            task.result = "Local evidence is relevant but insufficient."
        else:
            task.status = "unsupported"
            task.result = "Local evidence does not support the claim."
        task.confidence_after = clamp_score(
            (task.confidence_before * 0.4) + (max(0.0, best_score) * 0.6)
        )
        claim.verification_status = task.status
        claim.confidence_score = task.confidence_after
        if task.status in {"verified", "partially_verified"}:
            claim.supporting_evidence_ids = task.candidate_evidence_ids[:5]
    return tasks


def render_verification_markdown(tasks: list[VerificationTask]) -> str:
    lines = [
        "# Verification Results",
        "",
        "| Task | Priority | Status | Result |",
        "| --- | ---: | --- | --- |",
    ]
    for task in tasks:
        lines.append(
            f"| `{task.task_type}` for `{task.claim_id}` | {task.priority} | `{task.status}` | {task.result[:160]} |"
        )
    return "\n".join(lines) + "\n"


def write_verification_artifacts(run_dir: Path, tasks: list[VerificationTask]) -> list[str]:
    write_json(run_dir / "verification_tasks.json", {"tasks": tasks})
    write_json(run_dir / "verification_results.json", {"tasks": tasks})
    markdown = render_verification_markdown(tasks)
    (run_dir / "verification_tasks.md").write_text(markdown, encoding="utf-8")
    (run_dir / "verification_results.md").write_text(markdown, encoding="utf-8")
    return [
        "verification_tasks.json",
        "verification_tasks.md",
        "verification_results.json",
        "verification_results.md",
    ]
