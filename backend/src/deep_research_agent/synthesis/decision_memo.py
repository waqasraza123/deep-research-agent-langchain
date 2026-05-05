from __future__ import annotations

import re

from .comparison_matrix import infer_comparison_options, is_comparative_question
from .contracts import DecisionMemo, Recommendation, ResearchFinding, SynthesisInput

_DECISION_TERMS = (
    "adopt",
    "choose",
    "decision",
    "recommend",
    "should",
    "best",
    "architecture",
    "tradeoff",
    "trade-off",
    "migrate",
    "buy",
    "build",
    "use",
)

_RISK_TERMS = ("risk", "security", "privacy", "failure", "avoid", "limitation", "compliance")
_COST_TERMS = ("cost", "price", "pricing", "complexity", "burden", "expensive", "cheap")
_REVERSIBILITY_TERMS = ("reversible", "lock-in", "migration", "switch", "rollback", "portable")


def is_decision_question(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in _DECISION_TERMS)


def build_decision_memo(
    synthesis_input: SynthesisInput,
    findings: list[ResearchFinding],
) -> DecisionMemo:
    detected = is_decision_question(synthesis_input.question) or any(
        f.claim_type == "recommendation" for f in findings
    )
    if not detected and not is_comparative_question(synthesis_input.question):
        return DecisionMemo(
            thread_id=synthesis_input.thread_id,
            question=synthesis_input.question,
            generated_at=synthesis_input.generated_at,
            detected=False,
            warnings=["Decision intent was not detected from the question or findings."],
        )

    options = infer_comparison_options(synthesis_input.question, findings)
    if not options:
        options = _options_from_findings(findings)

    recommendation = _recommendation(options, findings)
    risks = _matching_texts(findings, _RISK_TERMS, limit=6)
    cost_complexity = _joined_or_gap(_matching_texts(findings, _COST_TERMS, limit=3))
    reversibility = _joined_or_gap(_matching_texts(findings, _REVERSIBILITY_TERMS, limit=3))
    rationale = _rationale(findings, recommendation.finding_ids if recommendation else [])
    warnings: list[str] = []
    if recommendation is None or recommendation.stance == "defer":
        warnings.append("No artifact-backed winning option was strong enough to recommend.")
    if not options:
        warnings.append("No explicit decision options were inferred.")
    if not risks:
        warnings.append("No explicit risk findings were detected; risk review remains incomplete.")

    return DecisionMemo(
        thread_id=synthesis_input.thread_id,
        question=synthesis_input.question,
        generated_at=synthesis_input.generated_at,
        detected=True,
        context=_context(synthesis_input, findings),
        decision_to_make=synthesis_input.question or "Decision requested by run.",
        options=options,
        recommendation=recommendation,
        rationale=rationale,
        risks=risks,
        reversibility=reversibility,
        cost_complexity=cost_complexity,
        confidence_label=recommendation.confidence_label if recommendation else "unknown",
        next_validation_steps=_validation_steps(synthesis_input, findings),
        warnings=warnings,
    )


def _recommendation(
    options: list[str],
    findings: list[ResearchFinding],
) -> Recommendation | None:
    if not findings:
        return Recommendation(
            recommendation_id="REC-defer",
            stance="defer",
            summary="Defer the decision because no findings are available.",
            confidence_label="unknown",
            conditions=["Collect source-backed evidence before deciding."],
        )

    option_scores: dict[str, tuple[int, list[ResearchFinding]]] = {}
    for option in options:
        matched = [
            f
            for f in findings
            if option.lower() in f.normalized_text
            and f.confidence_label in {"source_backed", "strong", "moderate"}
            and f.contradiction_status == "none"
        ]
        score = sum(_score(f) for f in matched)
        option_scores[option] = (score, matched)

    if option_scores:
        ranked = sorted(option_scores.items(), key=lambda item: (-item[1][0], item[0].lower()))
        option, (score, matched) = ranked[0]
        if score > 0 and matched:
            return Recommendation(
                recommendation_id=f"REC-{_slug(option)}",
                option=option,
                stance="conditional",
                summary=f"Conditionally prefer {option} based on the strongest available findings.",
                rationale=[f.text for f in matched[:4]],
                finding_ids=[f.finding_id for f in matched[:6]],
                confidence_label=_confidence(matched),
                conditions=[
                    "Validate weak, missing, or freshness-sensitive evidence before final adoption."
                ],
            )

    recommendation_findings = [
        f
        for f in findings
        if f.claim_type == "recommendation"
        and f.confidence_label not in {"unsupported", "contradicted"}
    ]
    if recommendation_findings:
        top = sorted(recommendation_findings, key=lambda f: f.finding_id)[0]
        return Recommendation(
            recommendation_id=f"REC-{top.finding_id}",
            stance="conditional",
            summary=top.text,
            rationale=[top.text],
            finding_ids=[top.finding_id],
            confidence_label=top.confidence_label,
            conditions=["Treat as conditional because the source findings do not prove final fit."],
        )

    return Recommendation(
        recommendation_id="REC-defer",
        stance="defer",
        summary="Defer the decision pending stronger evidence.",
        confidence_label="unknown",
        conditions=["Fill evidence gaps and review primary sources."],
    )


def _context(synthesis_input: SynthesisInput, findings: list[ResearchFinding]) -> str:
    source_count = len(synthesis_input.sources)
    finding_count = len(findings)
    return (
        f"Decision context assembled from {finding_count} finding(s), "
        f"{source_count} source metadata record(s), and available run artifacts."
    )


def _rationale(findings: list[ResearchFinding], selected_ids: list[str]) -> list[str]:
    selected = [f.text for f in findings if f.finding_id in set(selected_ids)]
    if selected:
        return selected[:6]
    return [
        f.text
        for f in findings
        if f.confidence_label in {"source_backed", "strong", "moderate"}
        and f.claim_type != "question"
    ][:6]


def _validation_steps(
    synthesis_input: SynthesisInput,
    findings: list[ResearchFinding],
) -> list[str]:
    steps = [
        "Review primary or official sources for each decisive claim.",
        "Verify unsupported and weak findings before acting.",
    ]
    if any(f.claim_type == "date_sensitive" for f in findings):
        steps.append("Refresh date-sensitive claims before final use.")
    if synthesis_input.subquestions:
        steps.append("Resolve open strategy subquestions that lack source-backed findings.")
    if any(f.contradiction_status != "none" for f in findings):
        steps.append("Resolve contradictory findings before treating the recommendation as final.")
    return steps


def _matching_texts(
    findings: list[ResearchFinding],
    terms: tuple[str, ...],
    *,
    limit: int,
) -> list[str]:
    out: list[str] = []
    for finding in findings:
        if any(term in finding.normalized_text for term in terms):
            out.append(finding.text)
        if len(out) >= limit:
            break
    return out


def _joined_or_gap(items: list[str]) -> str:
    return " ".join(items) if items else "Not assessed from available artifacts."


def _options_from_findings(findings: list[ResearchFinding]) -> list[str]:
    counts: dict[str, int] = {}
    for finding in findings:
        for entity in finding.entities:
            counts[entity] = counts.get(entity, 0) + 1
    return sorted(counts, key=lambda entity: (-counts[entity], entity.lower()))[:4]


def _score(finding: ResearchFinding) -> int:
    return {
        "source_backed": 5,
        "strong": 4,
        "moderate": 3,
        "weak": 1,
        "unknown": 0,
        "unsupported": -1,
        "contradicted": -3,
    }.get(finding.confidence_label, 0)


def _confidence(findings: list[ResearchFinding]) -> str:
    if not findings:
        return "unknown"
    labels = {f.confidence_label for f in findings}
    if "contradicted" in labels:
        return "contradicted"
    if labels <= {"source_backed", "strong"}:
        return "strong"
    if labels & {"source_backed", "strong", "moderate"}:
        return "moderate"
    if "weak" in labels:
        return "weak"
    return "unknown"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "option"
