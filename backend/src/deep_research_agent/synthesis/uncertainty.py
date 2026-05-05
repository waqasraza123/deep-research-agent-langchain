from __future__ import annotations

from .contracts import OpenQuestion, ResearchFinding, SynthesisInput, UncertaintyBoundary

_FRESHNESS_TERMS = (
    "current",
    "latest",
    "today",
    "now",
    "recent",
    "202",
    "release",
    "pricing",
    "version",
)

_PRIMARY_SOURCE_TERMS = (
    "official",
    "documentation",
    "source code",
    "policy",
    "regulation",
    "paper",
    "benchmark",
    "pricing",
)


def build_uncertainty_boundaries(
    synthesis_input: SynthesisInput,
    findings: list[ResearchFinding],
) -> UncertaintyBoundary:
    known = [
        f.text
        for f in findings
        if f.confidence_label in {"source_backed", "strong"} and f.contradiction_status == "none"
    ][:12]
    likely = [
        f.text
        for f in findings
        if f.confidence_label == "moderate" and f.contradiction_status == "none"
    ][:12]
    uncertain = [
        f.text
        for f in findings
        if f.confidence_label in {"weak", "unknown"} or f.contradiction_status != "none"
    ][:12]
    not_verified = [
        f.text for f in findings if f.confidence_label in {"unsupported", "unknown"}
    ][:12]
    freshness_dependent = [
        f.text
        for f in findings
        if f.claim_type == "date_sensitive"
        or any(term in f.normalized_text for term in _FRESHNESS_TERMS)
    ][:12]
    human_review = [f.text for f in findings if f.requires_human_review][:12]
    primary_sources = [
        f.text
        for f in findings
        if f.confidence_label in {"unsupported", "weak", "unknown"}
        and any(term in f.normalized_text for term in _PRIMARY_SOURCE_TERMS)
    ][:12]
    open_questions = _open_questions(synthesis_input, findings)
    warnings: list[str] = []
    if not known:
        warnings.append("No high-confidence known findings were detected.")
    if not synthesis_input.evidence_ledger:
        warnings.append("Evidence ledger was unavailable; uncertainty is based on raw artifacts.")
    if not synthesis_input.sources:
        warnings.append("No source metadata was available.")

    return UncertaintyBoundary(
        thread_id=synthesis_input.thread_id,
        question=synthesis_input.question,
        generated_at=synthesis_input.generated_at,
        known=known,
        likely=likely,
        uncertain=uncertain,
        not_verified=not_verified,
        freshness_dependent=freshness_dependent,
        requires_human_review=human_review,
        requires_primary_sources=primary_sources,
        open_questions=open_questions,
        warnings=warnings,
    )


def _open_questions(
    synthesis_input: SynthesisInput,
    findings: list[ResearchFinding],
) -> list[OpenQuestion]:
    out: list[OpenQuestion] = []
    question_findings = [f for f in findings if f.claim_type == "question"]
    for idx, finding in enumerate(question_findings[:8], start=1):
        out.append(
            OpenQuestion(
                question_id=f"OQ-{idx}",
                text=finding.text,
                reason="Carried from research strategy or unresolved notes.",
                related_finding_ids=[finding.finding_id],
                requires_primary_source=True,
            )
        )

    covered_subquestions = {
        sq for f in findings if f.claim_type != "question" for sq in f.subquestion_ids
    }
    next_idx = len(out) + 1
    for subquestion in synthesis_input.subquestions:
        sq_id = str(subquestion.get("id") or "")
        text = str(subquestion.get("question") or "").strip()
        if not text or sq_id in covered_subquestions:
            continue
        out.append(
            OpenQuestion(
                question_id=f"OQ-{next_idx}",
                text=text,
                reason="No source-backed finding was mapped to this subquestion.",
                requires_primary_source=True,
            )
        )
        next_idx += 1
        if len(out) >= 12:
            break
    return out
