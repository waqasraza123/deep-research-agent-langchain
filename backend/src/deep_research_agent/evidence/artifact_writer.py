from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import ClaimCitation, EvidenceLedger
from .ledger import build_evidence_ledger

EVIDENCE_ARTIFACTS = (
    "evidence_ledger.json",
    "evidence_ledger.md",
    "unsupported_claims.md",
    "contradictions.md",
    "citation_map.json",
    "evidence_coverage.json",
)


def rebuild_evidence_artifacts(run_dir: Path, *, thread_id: str) -> EvidenceLedger:
    ledger = build_evidence_ledger(run_dir, thread_id=thread_id)
    write_evidence_artifacts(run_dir, ledger)
    return ledger


def write_evidence_artifacts(run_dir: Path, ledger: EvidenceLedger) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evidence_ledger.json").write_text(
        json.dumps(_dump_model(ledger), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "evidence_ledger.md").write_text(render_evidence_ledger_md(ledger), encoding="utf-8")
    (run_dir / "unsupported_claims.md").write_text(
        render_unsupported_claims_md(ledger), encoding="utf-8"
    )
    (run_dir / "contradictions.md").write_text(render_contradictions_md(ledger), encoding="utf-8")
    (run_dir / "citation_map.json").write_text(
        json.dumps(_citation_map_json(ledger), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "evidence_coverage.json").write_text(
        json.dumps(_dump_model(ledger.coverage), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def render_evidence_ledger_md(ledger: EvidenceLedger) -> str:
    lines = [
        "# Evidence Ledger",
        "",
        f"- Ledger: `{ledger.ledger_id}`",
        f"- Thread: `{ledger.thread_id}`",
        f"- Generated: `{ledger.generated_at}`",
        f"- Sources: {len(ledger.sources)}",
        f"- Claims: {len(ledger.claims)}",
        f"- Unsupported claims: {len(ledger.unsupported_claims)}",
        f"- Contradiction groups: {len(ledger.contradictions)}",
        "",
        "## Coverage",
        "",
        f"- Average confidence: {ledger.coverage.average_confidence:.3f}",
        f"- Supported: {ledger.coverage.supported_claims}",
        f"- Partial: {ledger.coverage.partially_supported_claims}",
        f"- Unsupported: {ledger.coverage.unsupported_claims}",
        f"- Contradicted: {ledger.coverage.contradicted_claims}",
        "",
        "## Claims",
        "",
    ]
    for claim in ledger.claims:
        origin = f"- Origin: `{claim.origin}`"
        if claim.origin_ref:
            origin += f" (`{claim.origin_ref}`)"
        lines.extend(
            [
                f"### {claim.claim_id}",
                "",
                origin,
                f"- Type: `{claim.claim_type}`",
                f"- Support: `{claim.support_level}`",
                f"- Confidence: {claim.confidence_score:.3f}",
                f"- Review: {'yes' if claim.needs_human_review else 'no'}",
                f"- Text: {claim.text}",
            ]
        )
        if claim.citations:
            lines.append("- Citations:")
            for citation in claim.citations[:3]:
                lines.append(
                    f"  - `{citation.source_id}` score {citation.score:.3f}: {citation.reason}"
                )
        if claim.contradiction_ids:
            contradiction_ids = ", ".join(f"`{cid}`" for cid in claim.contradiction_ids)
            lines.append("- Contradictions: " + contradiction_ids)
        if claim.notes:
            lines.append("- Notes:")
            for note in claim.notes[:6]:
                lines.append(f"  - {note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_unsupported_claims_md(ledger: EvidenceLedger) -> str:
    lines = ["# Unsupported Claims", ""]
    if not ledger.unsupported_claims:
        lines.append("No unsupported claims detected by deterministic heuristics.")
        return "\n".join(lines) + "\n"
    for item in ledger.unsupported_claims:
        lines.extend(
            [
                f"## {item.claim_id}",
                "",
                f"- Origin: `{item.origin}`",
                f"- Support: `{item.support_level}`",
                f"- Reason: {item.reason}",
                f"- Text: {item.text}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_contradictions_md(ledger: EvidenceLedger) -> str:
    lines = ["# Contradictions", ""]
    if not ledger.contradictions:
        lines.append("No contradiction groups detected by deterministic heuristics.")
        return "\n".join(lines) + "\n"

    claims_by_id = {claim.claim_id: claim for claim in ledger.claims}
    for group in ledger.contradictions:
        lines.extend(
            [
                f"## {group.contradiction_id}",
                "",
                f"- Severity: `{group.severity}`",
                f"- Type: `{group.contradiction_type}`",
                f"- Explanation: {group.explanation}",
            ]
        )
        if group.values:
            lines.append("- Values: " + ", ".join(f"`{value}`" for value in group.values))
        lines.append("- Claims:")
        for claim_id in group.claim_ids:
            claim = claims_by_id.get(claim_id)
            if claim is None:
                lines.append(f"  - `{claim_id}`")
            else:
                lines.append(f"  - `{claim.claim_id}` ({claim.origin}): {claim.text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _citation_map_json(ledger: EvidenceLedger) -> dict[str, list[dict[str, Any]]]:
    return {
        claim_id: [_dump_model(citation) for citation in citations]
        for claim_id, citations in ledger.citation_map.items()
    }


def _dump_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, ClaimCitation):
        return model.dict()
    raise TypeError(f"Unsupported model type: {type(model)!r}")
