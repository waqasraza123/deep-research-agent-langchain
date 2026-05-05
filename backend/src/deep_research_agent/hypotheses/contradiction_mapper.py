from __future__ import annotations

import hashlib
import re
from typing import Any

from deep_research_agent.evidence.citation_mapper import content_terms
from deep_research_agent.evidence.claim_extractor import extract_values

from .contracts import HypothesisContradiction, HypothesisSet, ResearchHypothesis
from .generator import HypothesisBuildInput

_NEGATION_RE = re.compile(
    (
        r"\b(no|not|never|without|cannot|can't|does not|do not|is not|are not|"
        r"unsupported|incompatible)\b"
    ),
    re.IGNORECASE,
)
_OPPOSITES = (
    ("better", "worse"),
    ("stronger", "weaker"),
    ("more", "less"),
    ("higher", "lower"),
    ("faster", "slower"),
    ("easier", "harder"),
    ("supports", "does not support"),
    ("supported", "unsupported"),
    ("compatible", "incompatible"),
    ("recommended", "avoid"),
)


def map_contradictions(
    hypothesis_set: HypothesisSet,
    build_input: HypothesisBuildInput,
) -> HypothesisSet:
    contradictions: list[HypothesisContradiction] = []
    hypotheses = hypothesis_set.hypotheses

    for idx, left in enumerate(hypotheses):
        for right in hypotheses[idx + 1 :]:
            contradiction = _pair_contradiction(left, right)
            if contradiction:
                contradictions.append(contradiction)

    contradictions.extend(_ledger_contradictions(hypothesis_set, build_input))
    contradictions = _dedupe(contradictions)
    by_hypothesis: dict[str, list[str]] = {}
    competing: dict[str, list[str]] = {}
    for contradiction in contradictions:
        for hid in contradiction.hypothesis_ids:
            by_hypothesis.setdefault(hid, []).append(contradiction.contradiction_id)
            competing.setdefault(hid, [])
            for other in contradiction.hypothesis_ids:
                if other != hid and other not in competing[hid]:
                    competing[hid].append(other)

    for hypothesis in hypotheses:
        hypothesis.contradiction_ids = sorted(by_hypothesis.get(hypothesis.hypothesis_id, []))
        hypothesis.competing_hypothesis_ids = sorted(competing.get(hypothesis.hypothesis_id, []))

    hypothesis_set.contradictions = contradictions
    for result in hypothesis_set.test_results:
        linked = by_hypothesis.get(result.hypothesis_id, [])
        result.contradiction_ids = sorted(dict.fromkeys([*result.contradiction_ids, *linked]))
    return hypothesis_set


def _pair_contradiction(
    left: ResearchHypothesis, right: ResearchHypothesis
) -> HypothesisContradiction | None:
    if not _same_topic(left, right):
        return None
    left_text = left.normalized_text
    right_text = right.normalized_text
    left_values = {value.lower() for value in extract_values(left.text)}
    right_values = {value.lower() for value in extract_values(right.text)}
    if left_values and right_values and left_values != right_values:
        return _contradiction(
            [left.hypothesis_id, right.hypothesis_id],
            "conflicting_values",
            "high",
            "Hypotheses discuss the same topic but contain different numeric or date values.",
            values=sorted(left_values | right_values),
        )
    if _NEGATION_RE.search(left_text) != _NEGATION_RE.search(right_text):
        return _contradiction(
            [left.hypothesis_id, right.hypothesis_id],
            "negation_conflict",
            "medium",
            "One hypothesis contains explicit negation while the other affirms a similar topic.",
        )
    for positive, negative in _OPPOSITES:
        if _contains(left_text, positive) and _contains(right_text, negative):
            return _contradiction(
                [left.hypothesis_id, right.hypothesis_id],
                "competing_hypothesis",
                "medium",
                f"Hypotheses use competing language: {positive!r} vs {negative!r}.",
            )
        if _contains(left_text, negative) and _contains(right_text, positive):
            return _contradiction(
                [left.hypothesis_id, right.hypothesis_id],
                "competing_hypothesis",
                "medium",
                f"Hypotheses use competing language: {negative!r} vs {positive!r}.",
            )
    return None


def _ledger_contradictions(
    hypothesis_set: HypothesisSet,
    build_input: HypothesisBuildInput,
) -> list[HypothesisContradiction]:
    ledger = build_input.evidence_ledger or {}
    groups = ledger.get("contradictions") if isinstance(ledger, dict) else None
    if not isinstance(groups, list):
        return []
    claim_to_hypotheses: dict[str, list[str]] = {}
    for result in hypothesis_set.test_results:
        for evidence in [*result.supporting_evidence, *result.opposing_evidence]:
            if evidence.claim_id:
                claim_to_hypotheses.setdefault(evidence.claim_id, []).append(result.hypothesis_id)
    out: list[HypothesisContradiction] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        claim_ids = [str(cid) for cid in group.get("claim_ids") or []]
        hypothesis_ids = sorted(
            {hid for claim_id in claim_ids for hid in claim_to_hypotheses.get(claim_id, [])}
        )
        if not hypothesis_ids:
            continue
        out.append(
            HypothesisContradiction(
                contradiction_id=str(group.get("contradiction_id") or _digest(claim_ids)),
                hypothesis_ids=hypothesis_ids,
                claim_ids=claim_ids,
                contradiction_type=str(group.get("contradiction_type") or "claim_conflict"),
                severity=str(group.get("severity") or "medium"),
                explanation=str(group.get("explanation") or "Linked evidence claims conflict."),
                values=[str(value) for value in group.get("values") or []],
            )
        )
    return out


def _same_topic(left: ResearchHypothesis, right: ResearchHypothesis) -> bool:
    left_terms = content_terms(left.normalized_text)
    right_terms = content_terms(right.normalized_text)
    if not left_terms or not right_terms:
        return False
    overlap = len(left_terms & right_terms) / max(min(len(left_terms), len(right_terms)), 1)
    return overlap >= 0.38 or bool(_entities(left.text) & _entities(right.text) and overlap >= 0.20)


def _entities(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", text))


def _contains(text: str, phrase: str) -> bool:
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def _contradiction(
    hypothesis_ids: list[str],
    contradiction_type: str,
    severity: str,
    explanation: str,
    *,
    values: list[str] | None = None,
) -> HypothesisContradiction:
    return HypothesisContradiction(
        contradiction_id="HK-" + _digest([contradiction_type, *hypothesis_ids]),
        hypothesis_ids=hypothesis_ids,
        contradiction_type=contradiction_type,
        severity=severity,
        explanation=explanation,
        values=values or [],
    )


def _digest(values: list[Any]) -> str:
    key = "|".join(str(value) for value in values)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _dedupe(items: list[HypothesisContradiction]) -> list[HypothesisContradiction]:
    seen: set[str] = set()
    out: list[HypothesisContradiction] = []
    for item in items:
        key = "|".join(
            [
                item.contradiction_type,
                ",".join(sorted(item.hypothesis_ids)),
                ",".join(sorted(item.claim_ids)),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
