from __future__ import annotations

import json
import re
from pathlib import Path

from .contracts import (
    CritiqueFinding,
    EvidenceUnit,
    KernelWarning,
    ResearchBlueprint,
    ResearchClaim,
    ResearchKernelSettings,
    SourceUnit,
    clamp_score,
    stable_id,
    tokenize,
    write_json,
)

NUMBER_RE = re.compile(
    r"(?:[$€£]\s*)?\b\d+(?:[.,]\d+)*(?:%|x|ms|s|GB|MB|k|m|bn|million|billion)?\b", re.I
)
DATE_RE = re.compile(
    r"\b(?:20\d{2}|19\d{2}|latest|current|today|recent|now|version\s+\d+(?:\.\d+)*)\b", re.I
)
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[- ][A-Z][A-Za-z0-9]*)*\b")
ABSOLUTE = {
    "always",
    "never",
    "guaranteed",
    "proven",
    "definitely",
    "best",
    "only",
    "must",
    "no risk",
}


def _read(path: Path) -> str:
    if not path.exists() or path.is_dir():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _sentences(markdown: str) -> list[str]:
    rows: list[str] = []
    for line in markdown.splitlines():
        clean = line.strip().lstrip("-*0123456789. ").strip()
        if not clean or clean.startswith("#") or clean.startswith("| ---"):
            continue
        for part in re.split(r"(?<=[.!?])\s+", clean):
            part = part.strip()
            if len(part) >= 30:
                rows.append(part)
    return rows


def _claim_type(text: str, blueprint: ResearchBlueprint) -> str:
    lc = text.lower()
    if blueprint.intent.label == "legal_policy_review" or any(
        w in lc for w in ("law", "regulation", "policy", "liability", "compliance")
    ):
        return "legal_policy"
    if blueprint.intent.label == "medical_health_review" or any(
        w in lc for w in ("diagnosis", "treatment", "symptom", "medication", "doctor", "health")
    ):
        return "medical_health"
    if blueprint.intent.label == "financial_risk_review" or any(
        w in lc for w in ("investment", "stock", "revenue", "valuation", "pricing", "cost")
    ):
        return "financial"
    if DATE_RE.search(text):
        return "temporal"
    if NUMBER_RE.search(text):
        return "numeric"
    if any(w in lc for w in ("better", "worse", "faster", "cheaper", "compared", "versus", " vs ")):
        return "comparative"
    if any(w in lc for w in ("because", "leads to", "results in", "due to")):
        return "causal"
    if any(w in lc for w in ("should", "best", "recommend", "suitable", "ideal")):
        return "recommendation"
    if any(w in lc for w in ("risk", "failure", "limitation", "concern", "warning")):
        return "risk"
    if any(w in lc for w in ("api", "sdk", "database", "backend", "framework", "architecture")):
        return "technical"
    return "factual"


def _strength(text: str) -> str:
    lc = text.lower()
    if any(term in lc for term in ABSOLUTE):
        return "absolute"
    if any(w in lc for w in ("clearly", "significant", "material", "substantially")):
        return "strong"
    if any(w in lc for w in ("may", "might", "could", "appears", "suggests")):
        return "weak"
    return "normal"


def extract_claims(
    run_dir: Path,
    blueprint: ResearchBlueprint,
    settings: ResearchKernelSettings | None = None,
) -> tuple[list[ResearchClaim], list[KernelWarning]]:
    settings = settings or ResearchKernelSettings()
    claims: list[ResearchClaim] = []
    warnings: list[KernelWarning] = []
    for artifact in ("report.md", "notes.md"):
        for sentence in _sentences(_read(run_dir / artifact)):
            normalized = " ".join(sentence.lower().split())
            ctype = _claim_type(sentence, blueprint)
            strength = _strength(sentence)
            claim_warnings: list[KernelWarning] = []
            if strength == "absolute":
                claim_warnings.append(
                    KernelWarning(
                        warning_id=stable_id("warn", artifact, normalized, "absolute"),
                        subsystem="claim_extraction",
                        code="absolute_language",
                        severity="medium",
                        message="Claim uses strong or absolute language.",
                        recommended_action="Verify support or soften the wording.",
                    )
                )
            claim = ResearchClaim(
                claim_id=stable_id("claim", artifact, normalized),
                text=sentence,
                normalized_text=normalized,
                claim_type=ctype,  # type: ignore[arg-type]
                source_artifact=artifact,
                strength=strength,  # type: ignore[arg-type]
                entities=sorted(set(ENTITY_RE.findall(sentence)))[:12],
                numbers=NUMBER_RE.findall(sentence)[:12],
                dates=DATE_RE.findall(sentence)[:12],
                warnings=claim_warnings,
            )
            claims.append(claim)
            warnings.extend(claim_warnings)
            if len(claims) >= settings.max_claims:
                return claims, warnings
    return claims, warnings


def _evidence_support(
    claim: ResearchClaim, evidence_units: list[EvidenceUnit]
) -> tuple[list[str], float]:
    claim_terms = tokenize(claim.normalized_text)
    best: list[tuple[str, float]] = []
    for evidence in evidence_units:
        evidence_terms = tokenize(evidence.normalized_text)
        overlap = len(claim_terms & evidence_terms) / max(1, len(claim_terms))
        number_match = bool(set(claim.numbers) & set(evidence.numbers)) if claim.numbers else False
        date_match = (
            bool(set(d.lower() for d in claim.dates) & set(d.lower() for d in evidence.dates))
            if claim.dates
            else False
        )
        score = (
            overlap
            + (0.25 if number_match else 0.0)
            + (0.2 if date_match else 0.0)
            + evidence.support_score * 0.22
        )
        if score >= 0.32:
            best.append((evidence.evidence_unit_id, score))
    best.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in best[:5]], clamp_score(best[0][1] if best else 0.0)


def critique_report(
    run_dir: Path,
    blueprint: ResearchBlueprint,
    source_units: list[SourceUnit],
    evidence_units: list[EvidenceUnit],
    claims: list[ResearchClaim],
) -> tuple[list[CritiqueFinding], list[KernelWarning]]:
    findings: list[CritiqueFinding] = []
    source_text = " ".join(e.normalized_text for e in evidence_units)
    for claim in claims:
        support_ids, support_score = _evidence_support(claim, evidence_units)
        claim.supporting_evidence_ids = support_ids
        claim.confidence_score = clamp_score(support_score)
        if not support_ids or support_score < 0.34:
            findings.append(
                CritiqueFinding(
                    finding_id=stable_id("finding", "unsupported", claim.claim_id),
                    severity="high" if claim.source_artifact == "report.md" else "medium",
                    category="unsupported_claim",
                    message="Claim has weak or no overlapping local evidence.",
                    affected_artifacts=[claim.source_artifact],
                    affected_claims=[claim.claim_id],
                    recommended_action="Add citation-ready evidence or remove/soften the claim.",
                )
            )
        if claim.numbers and not any(num in source_text for num in claim.numbers):
            findings.append(
                CritiqueFinding(
                    finding_id=stable_id("finding", "numeric", claim.claim_id),
                    severity="high",
                    category="numeric_mismatch",
                    message="Claim contains numbers not found in evidence units.",
                    affected_artifacts=[claim.source_artifact],
                    affected_claims=[claim.claim_id],
                    recommended_action="Verify the numeric value against source text.",
                )
            )
        if (
            claim.dates
            and blueprint.freshness_policy.get("freshness_sensitive")
            and not any(d.lower() in source_text for d in claim.dates)
        ):
            findings.append(
                CritiqueFinding(
                    finding_id=stable_id("finding", "temporal", claim.claim_id),
                    severity="medium",
                    category="temporal_mismatch",
                    message="Currentness or date-sensitive claim lacks dated evidence.",
                    affected_artifacts=[claim.source_artifact],
                    affected_claims=[claim.claim_id],
                    recommended_action="Use dated source evidence or avoid latest/current language.",
                )
            )
        if claim.strength == "absolute":
            findings.append(
                CritiqueFinding(
                    finding_id=stable_id("finding", "overclaim", claim.claim_id),
                    severity="medium",
                    category="overclaiming",
                    message="Strong or absolute language requires explicit high-quality support.",
                    affected_artifacts=[claim.source_artifact],
                    affected_claims=[claim.claim_id],
                    recommended_action="Use qualified language unless primary evidence supports the claim.",
                )
            )
    if blueprint.intent.label in {
        "comparative_analysis",
        "technical_due_diligence",
        "vendor_evaluation",
    }:
        report_lc = _read(run_dir / "report.md").lower()
        if not any(
            w in report_lc
            for w in ("tradeoff", "limitation", "counter", "on the other hand", "however")
        ):
            findings.append(
                CritiqueFinding(
                    finding_id=stable_id("finding", "counterargument", blueprint.thread_id),
                    severity="medium",
                    category="missing_counterargument",
                    message="Comparative report does not clearly discuss tradeoffs or counterarguments.",
                    affected_artifacts=["report.md"],
                    recommended_action="Add tradeoffs, limitations, and conditions for the recommendation.",
                )
            )
    primary_count = len([u for u in source_units if u.source_role == "primary_evidence"])
    if blueprint.citation_policy.get("strict") and primary_count == 0:
        findings.append(
            CritiqueFinding(
                finding_id=stable_id("finding", "primary_missing", blueprint.thread_id),
                severity="high",
                category="missing_primary_source",
                message="Strict citation policy requires primary or reputable sources, but none were detected.",
                affected_artifacts=["sources.json"],
                recommended_action="Add primary, official, regulator, clinical, or financial filing sources as appropriate.",
            )
        )
    if blueprint.freshness_policy.get("freshness_sensitive") and not any(
        u.fetched_at for u in source_units
    ):
        findings.append(
            CritiqueFinding(
                finding_id=stable_id("finding", "freshness", blueprint.thread_id),
                severity="medium",
                category="stale_source",
                message="Freshness-sensitive request has sources with unknown fetched dates.",
                affected_artifacts=["sources.json"],
                recommended_action="Use dated source metadata and dated source text.",
            )
        )
    if blueprint.intent.label in {
        "legal_policy_review",
        "medical_health_review",
        "financial_risk_review",
    }:
        findings.append(
            CritiqueFinding(
                finding_id=stable_id("finding", "sensitive", blueprint.thread_id),
                severity="high",
                category="unclear_uncertainty",
                message="Sensitive-domain output requires human review and conservative uncertainty.",
                affected_artifacts=["report.md"],
                recommended_action="Ensure the report is not presented as professional advice.",
            )
        )
    for artifact, min_len in (("report.md", 80), ("notes.md", 20), ("plan.md", 20)):
        text = _read(run_dir / artifact).strip()
        if len(text) < min_len:
            findings.append(
                CritiqueFinding(
                    finding_id=stable_id("finding", "artifact", artifact, blueprint.thread_id),
                    severity="medium" if artifact != "report.md" else "high",
                    category="artifact_missing",
                    message=f"{artifact} is missing or too short for serious auditability.",
                    affected_artifacts=[artifact],
                    recommended_action=f"Regenerate or backfill {artifact} with meaningful content.",
                )
            )
    warnings = [
        KernelWarning(
            warning_id=stable_id("warn", finding.finding_id),
            subsystem="critique",
            code=finding.category,
            severity=finding.severity,
            message=finding.message,
            affected_artifacts=finding.affected_artifacts,
            affected_sources=finding.affected_sources,
            affected_claims=finding.affected_claims,
            recommended_action=finding.recommended_action,
        )
        for finding in findings
    ]
    return findings, warnings


def render_claims_markdown(claims: list[ResearchClaim]) -> str:
    lines = [
        "# Claims",
        "",
        "| Claim | Type | Strength | Confidence | Warnings |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for claim in claims:
        warnings = "; ".join(w.message for w in claim.warnings) or "None"
        lines.append(
            f"| {claim.text[:140]} | `{claim.claim_type}` | `{claim.strength}` | {claim.confidence_score:.2f} | {warnings[:120]} |"
        )
    return "\n".join(lines) + "\n"


def render_findings_markdown(findings: list[CritiqueFinding]) -> str:
    lines = ["# Critique Findings", ""]
    for severity in ("critical", "high", "medium", "low", "info"):
        rows = [f for f in findings if f.severity == severity]
        if not rows:
            continue
        lines.extend([f"## {severity.title()}", ""])
        for finding in rows:
            lines.append(
                f"- `{finding.category}`: {finding.message} Action: {finding.recommended_action}"
            )
        lines.append("")
    if not findings:
        lines.append("No critique findings.")
    return "\n".join(lines) + "\n"


def write_critique_artifacts(
    run_dir: Path,
    claims: list[ResearchClaim],
    findings: list[CritiqueFinding],
    warnings: list[KernelWarning],
) -> list[str]:
    write_json(run_dir / "claims.json", {"claims": claims})
    write_json(run_dir / "critique_findings.json", {"findings": findings})
    write_json(run_dir / "operator_warnings.json", {"warnings": warnings})
    (run_dir / "claims.md").write_text(render_claims_markdown(claims), encoding="utf-8")
    (run_dir / "critique_findings.md").write_text(
        render_findings_markdown(findings), encoding="utf-8"
    )
    (run_dir / "operator_warnings.md").write_text(
        render_findings_markdown(
            [
                CritiqueFinding(
                    finding_id=w.warning_id,
                    severity=w.severity,
                    category="unsupported_claim"
                    if w.code == "unsupported_claim"
                    else "source_quality",
                    message=w.message,
                    affected_artifacts=w.affected_artifacts,
                    affected_claims=w.affected_claims,
                    affected_sources=w.affected_sources,
                    recommended_action=w.recommended_action,
                )
                for w in warnings
            ]
        ),
        encoding="utf-8",
    )
    return [
        "claims.json",
        "claims.md",
        "critique_findings.json",
        "critique_findings.md",
        "operator_warnings.json",
        "operator_warnings.md",
    ]


def read_claims(run_dir: Path) -> list[ResearchClaim]:
    path = run_dir / "claims.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("claims", data if isinstance(data, list) else [])
    return [ResearchClaim(**row) for row in rows if isinstance(row, dict)]
