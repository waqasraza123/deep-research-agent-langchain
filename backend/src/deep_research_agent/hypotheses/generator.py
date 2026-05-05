from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import now_iso_utc
from deep_research_agent.evidence.claim_extractor import (
    ClaimInput,
    extract_claims,
    normalize_claim_text,
    tokenize,
)

from .contracts import (
    HypothesisSet,
    HypothesisStatus,
    HypothesisSummary,
    HypothesisType,
    ResearchHypothesis,
)


@dataclass
class HypothesisBuildInput:
    thread_id: str
    question: str
    generated_at: str
    plan_text: str = ""
    notes_text: str = ""
    report_text: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    subquestions: list[dict[str, Any]] = field(default_factory=list)
    evidence_ledger: dict[str, Any] | None = None
    source_audit: dict[str, Any] | None = None
    retrieval_context: dict[str, Any] | None = None
    synthesis_output: dict[str, Any] | None = None
    available_artifacts: list[str] = field(default_factory=list)


_COMPARISON_RE = re.compile(
    r"\b(?:compare|versus|vs\.?|better than|prefer|choose between|alternative to)\b",
    re.IGNORECASE,
)
_TECHNICAL_RE = re.compile(
    r"\b(api|backend|frontend|database|architecture|framework|library|langgraph|crewai|"
    r"deployment|production|orchestration|checkpoint|resumability|observability|"
    r"latency|scalability|integration|runtime|agent)\b",
    re.IGNORECASE,
)
_RISK_RE = re.compile(r"\b(risk|security|failure|trade[- ]?off|limitation|unsafe)\b", re.I)
_MARKET_RE = re.compile(r"\b(market|vendor|pricing|adoption|customer|enterprise)\b", re.I)
_LEGAL_RE = re.compile(r"\b(legal|policy|regulation|compliance|law|license)\b", re.I)
_TEMPORAL_RE = re.compile(r"\b(current|latest|recent|today|now|timeline|roadmap|202\d)\b", re.I)


def load_hypothesis_input(run_dir: Path, *, thread_id: str) -> HypothesisBuildInput:
    available = sorted(
        str(path.relative_to(run_dir)).replace("\\", "/")
        for path in run_dir.rglob("*")
        if path.is_file()
    )
    run_meta = _load_json(run_dir / "run.json") or _load_json(run_dir / ".run.json") or {}
    question = str(
        run_meta.get("question")
        or (run_meta.get("input_snapshot") or {}).get("question")
        or _question_from_metadata(run_dir)
        or ""
    )
    strategy = _load_json(run_dir / "strategy.json")
    subquestions = _load_json(run_dir / "subquestions.json")
    if not isinstance(subquestions, list) and isinstance(strategy, dict):
        subquestions = strategy.get("subquestions")
    if not isinstance(subquestions, list):
        subquestions = []

    return HypothesisBuildInput(
        thread_id=thread_id,
        question=question,
        generated_at=now_iso_utc(),
        plan_text=_read_text(run_dir / "plan.md"),
        notes_text=_read_text(run_dir / "notes.md"),
        report_text=_read_text(run_dir / "report.md"),
        sources=_load_sources(run_dir / "sources.json"),
        subquestions=[item for item in subquestions if isinstance(item, dict)],
        evidence_ledger=_load_json_object(run_dir / "evidence_ledger.json"),
        source_audit=_load_json_object(run_dir / "source_audit.json"),
        retrieval_context=_load_json_object(run_dir / "context_packs.json"),
        synthesis_output=_load_json_object(run_dir / "synthesis_output.json"),
        available_artifacts=available,
    )


def generate_hypotheses(build_input: HypothesisBuildInput) -> HypothesisSet:
    hypotheses: list[ResearchHypothesis] = []
    seen: set[str] = set()

    def add(
        text: str,
        h_type: HypothesisType,
        *,
        origin: str,
        refs: list[str] | None = None,
        subquestion_ids: list[str] | None = None,
        assumptions: list[str] | None = None,
    ) -> None:
        clean = _clean_hypothesis_text(text)
        if not clean:
            return
        normalized = normalize_claim_text(clean)
        if len(tokenize(normalized)) < 4:
            return
        key = _dedupe_key(normalized)
        if key in seen:
            return
        seen.add(key)
        ordinal = len(hypotheses) + 1
        hypotheses.append(
            ResearchHypothesis(
                hypothesis_id=_hypothesis_id(build_input.thread_id, normalized, ordinal),
                text=clean,
                normalized_text=normalized,
                hypothesis_type=h_type,
                status=HypothesisStatus.PROPOSED,
                origin=origin,
                origin_refs=refs or [],
                subquestion_ids=subquestion_ids or [],
                assumptions=assumptions or [],
            )
        )

    question_type = classify_hypothesis_type(build_input.question)
    entities = _question_entities(build_input.question)
    if _COMPARISON_RE.search(build_input.question):
        left, right = _comparison_subjects(build_input.question)
        if left and right:
            add(
                (
                    f"{left} is stronger than {right} for the core requirement in "
                    "the research question."
                ),
                HypothesisType.COMPARATIVE,
                origin="question",
                refs=["question"],
                assumptions=[
                    "The comparison criteria in the question are sufficiently represented "
                    "in the sources."
                ],
            )
            add(
                f"{right} may be easier or faster to adopt than {left} for some workflows.",
                HypothesisType.COMPARATIVE,
                origin="question",
                refs=["question"],
            )
            add(
                (
                    f"The best choice between {left} and {right} depends on deployment, "
                    "observability, and maintenance requirements."
                ),
                HypothesisType.RECOMMENDATION,
                origin="question",
                refs=["question"],
            )
            if _TECHNICAL_RE.search(build_input.question):
                add(
                    (
                        f"{left} may offer stronger durable orchestration, checkpointing, "
                        f"or state management than {right}."
                    ),
                    HypothesisType.TECHNICAL,
                    origin="question",
                    refs=["question"],
                )
                add(
                    (
                        f"{right} may be better suited than {left} for role-based "
                        "multi-agent workflow prototyping."
                    ),
                    HypothesisType.TECHNICAL,
                    origin="question",
                    refs=["question"],
                )
    elif question_type != HypothesisType.UNKNOWN:
        subject = _subject_phrase(build_input.question, entities)
        add(
            f"{subject} has enough available evidence to support a research conclusion.",
            question_type,
            origin="question",
            refs=["question"],
        )

    if _TECHNICAL_RE.search(build_input.question):
        subject = _subject_phrase(build_input.question, entities)
        add(
            (
                f"{subject} is technically viable for the production use case described "
                "in the question."
            ),
            HypothesisType.TECHNICAL,
            origin="question",
            refs=["question"],
        )
        add(
            (
                f"{subject} has unresolved operational risks around deployment, monitoring, "
                "or failure recovery."
            ),
            HypothesisType.RISK,
            origin="question",
            refs=["question"],
        )

    for item in build_input.subquestions[:12]:
        q = str(item.get("question") or "").strip()
        if not q:
            continue
        sq_id = str(item.get("id") or item.get("subquestion_id") or "")
        add(
            _subquestion_to_hypothesis(q),
            classify_hypothesis_type(q),
            origin="subquestion",
            refs=["subquestions.json"],
            subquestion_ids=[sq_id] if sq_id else [],
        )

    _add_claim_hypotheses(add, build_input)
    _add_source_audit_hypotheses(add, build_input)
    _add_retrieval_hypotheses(add, build_input)

    if not hypotheses:
        add(
            (
                "The available artifacts are insufficient to form a well-supported "
                "research conclusion."
            ),
            HypothesisType.UNKNOWN,
            origin="fallback",
            refs=build_input.available_artifacts[:6],
        )

    summary = HypothesisSummary(
        thread_id=build_input.thread_id,
        generated_at=build_input.generated_at,
        total_hypotheses=len(hypotheses),
        warnings=_generation_warnings(build_input, hypotheses),
    )
    digest = hashlib.sha1(
        f"{build_input.thread_id}:{build_input.generated_at}:{len(hypotheses)}".encode("utf-8")
    ).hexdigest()[:12]
    return HypothesisSet(
        hypothesis_set_id=f"HS-{digest}",
        thread_id=build_input.thread_id,
        question=build_input.question,
        generated_at=build_input.generated_at,
        hypotheses=hypotheses,
        summary=summary,
        metadata={
            "method": "deterministic_offline_heuristics",
            "available_artifacts": build_input.available_artifacts,
            "source_count": len(build_input.sources),
            "subquestion_count": len(build_input.subquestions),
        },
    )


def classify_hypothesis_type(text: str) -> HypothesisType:
    normalized = text.lower()
    if _LEGAL_RE.search(normalized):
        return HypothesisType.LEGAL_POLICY
    if _MARKET_RE.search(normalized):
        return HypothesisType.MARKET
    if _RISK_RE.search(normalized):
        return HypothesisType.RISK
    if _TEMPORAL_RE.search(normalized):
        return HypothesisType.TEMPORAL
    if re.search(r"\b(cause|caused|because|leads to|drives|due to)\b", normalized):
        return HypothesisType.CAUSAL
    if _COMPARISON_RE.search(normalized) or re.search(
        r"\b(more|less|better|worse|than)\b", normalized
    ):
        return HypothesisType.COMPARATIVE
    if re.search(r"\b(should|recommend|best|choose|prefer|avoid)\b", normalized):
        return HypothesisType.RECOMMENDATION
    if _TECHNICAL_RE.search(normalized):
        return HypothesisType.TECHNICAL
    if len(tokenize(normalized)) >= 4:
        return HypothesisType.FACTUAL
    return HypothesisType.UNKNOWN


def _add_claim_hypotheses(add, build_input: HypothesisBuildInput) -> None:
    ledger = build_input.evidence_ledger or {}
    claims = ledger.get("claims") if isinstance(ledger, dict) else None
    if isinstance(claims, list) and claims:
        for claim in claims[:20]:
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("text") or "").strip()
            support = str(claim.get("support_level") or "")
            if not text or support == "source_backed":
                continue
            add(
                text,
                classify_hypothesis_type(text),
                origin="evidence_ledger",
                refs=["evidence_ledger.json", str(claim.get("claim_id") or "")],
            )
        return

    inputs = [
        ClaimInput(origin="notes", text=build_input.notes_text, origin_ref="notes.md"),
        ClaimInput(origin="report", text=build_input.report_text, origin_ref="report.md"),
    ]
    for claim in extract_claims(inputs)[:18]:
        add(
            claim.text,
            classify_hypothesis_type(claim.text),
            origin=claim.origin,
            refs=[claim.origin_ref or claim.origin],
        )


def _add_source_audit_hypotheses(add, build_input: HypothesisBuildInput) -> None:
    audit = build_input.source_audit or {}
    summary = audit.get("summary") if isinstance(audit, dict) else {}
    if not isinstance(summary, dict):
        return
    if summary.get("coverage_gaps"):
        add(
            "The current source set may not cover all evidence required for the research question.",
            HypothesisType.RISK,
            origin="source_audit",
            refs=["source_audit.json"],
        )
    if summary.get("recommended_primary_sources"):
        add(
            "At least one primary or high-authority source is available for testing key claims.",
            HypothesisType.FACTUAL,
            origin="source_audit",
            refs=["source_audit.json"],
        )
    if summary.get("freshness_gaps"):
        add(
            "Freshness gaps may limit confidence in time-sensitive conclusions.",
            HypothesisType.TEMPORAL,
            origin="source_audit",
            refs=["source_audit.json"],
        )


def _add_retrieval_hypotheses(add, build_input: HypothesisBuildInput) -> None:
    context = build_input.retrieval_context or {}
    packs = context.get("packs") if isinstance(context, dict) else None
    if not isinstance(packs, dict):
        return
    evidence_pack = packs.get("evidence_context_pack") or packs.get("agent_context_pack")
    items = evidence_pack.get("items") if isinstance(evidence_pack, dict) else None
    if isinstance(items, list) and items:
        add(
            (
                "Retrieved context contains enough focused excerpts to test at least part "
                "of the research question."
            ),
            HypothesisType.FACTUAL,
            origin="retrieval_context",
            refs=["context_packs.json"],
        )


def _subquestion_to_hypothesis(question: str) -> str:
    q = question.strip().rstrip("?")
    if re.match(r"^(is|are|does|do|can|should|will|would|has|have)\b", q, re.I):
        return q[0].upper() + q[1:] + "."
    return f"The answer to whether {q[0].lower() + q[1:]} materially affects the conclusion."


def _comparison_subjects(question: str) -> tuple[str | None, str | None]:
    clean = re.sub(r"[?]", "", question).strip()
    patterns = [
        r"\b(?:is|are)\s+(.+?)\s+better than\s+(.+?)\s+(?:for|as|in|$)",
        r"\bcompare\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+?)\s+(?:for|as|in|$)",
        r"\b(.+?)\s+(?:vs\.?|versus)\s+(.+?)\s+(?:for|as|in|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, re.I)
        if match:
            return _trim_subject(match.group(1)), _trim_subject(match.group(2))
    entities = _question_entities(question)
    if len(entities) >= 2:
        return entities[0], entities[1]
    return None, None


def _question_entities(question: str) -> list[str]:
    entities = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]*)+\b", question)
    titled = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", question)
    out: list[str] = []
    for entity in [*entities, *titled]:
        entity = _trim_subject(entity)
        if entity and entity.lower() not in {"is", "the", "for"} and entity not in out:
            out.append(entity)
    return out


def _subject_phrase(question: str, entities: list[str]) -> str:
    if entities:
        return " and ".join(entities[:2])
    tokens = [
        t
        for t in tokenize(question)
        if t not in {"what", "when", "where", "which", "should"}
    ]
    return " ".join(tokens[:6]) or "The researched subject"


def _trim_subject(value: str) -> str:
    value = re.sub(r"^(the|a|an|whether|if)\s+", "", value.strip(), flags=re.I)
    value = re.sub(r"\s+(for|as|in|with|when|if|because).*$", "", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" .,:;")
    words = value.split()
    if len(words) > 6:
        value = " ".join(words[-6:])
    return value


def _clean_hypothesis_text(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip(" -\t\r\n")
    if not clean:
        return ""
    if not clean.endswith((".", "?", "!")):
        clean += "."
    return clean


def _dedupe_key(normalized: str) -> str:
    terms = [t for t in tokenize(normalized) if len(t) >= 4]
    return " ".join(sorted(dict.fromkeys(terms))[:14])


def _hypothesis_id(thread_id: str, normalized: str, ordinal: int) -> str:
    digest = hashlib.sha1(f"{thread_id}:{ordinal}:{normalized}".encode("utf-8")).hexdigest()[:10]
    return f"H-{digest}"


def _generation_warnings(
    build_input: HypothesisBuildInput, hypotheses: list[ResearchHypothesis]
) -> list[str]:
    warnings: list[str] = []
    if not build_input.sources:
        warnings.append("No sources were available; generated hypotheses require evidence.")
    if build_input.evidence_ledger is None:
        warnings.append(
            "Evidence ledger was not available; generation used raw notes/report artifacts."
        )
    if len(hypotheses) <= 2:
        warnings.append("Few hypotheses were generated; conclusions should remain tentative.")
    return warnings


def _load_json(path: Path) -> Any:
    if not path.exists() or path.is_dir():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_json_object(path: Path) -> dict[str, Any] | None:
    data = _load_json(path)
    return data if isinstance(data, dict) else None


def _load_sources(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("sources"), list):
        return [item for item in data["sources"] if isinstance(item, dict)]
    return []


def _read_text(path: Path, *, max_chars: int = 120_000) -> str:
    if not path.exists() or path.is_dir():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def _question_from_metadata(run_dir: Path) -> str:
    metadata = _load_json(run_dir / "metadata.json")
    if isinstance(metadata, dict):
        return str(metadata.get("question") or "")
    return ""
