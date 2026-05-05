from __future__ import annotations

import hashlib
import re
from typing import Any

from .contracts import VerificationBatch, VerificationResult, VerificationTaskStatus

_ABSOLUTE_RE = re.compile(
    r"\b(always|never|best|must|guarantees?|proves?|undeniably|clearly|definitely|only)\b",
    re.I,
)
_RECOMMEND_RE = re.compile(r"\b(should|must|recommend|recommended|best|prefer|avoid)\b", re.I)
_CAUSAL_RE = re.compile(r"\b(because|caused|causes|leads to|led to|drives|therefore)\b", re.I)


class ClaimChallenger:
    """Suggests conservative rewrites. It never mutates the report text."""

    def suggestions_for_results(
        self,
        results: list[VerificationResult],
    ) -> list[dict[str, Any]]:
        suggestions: list[dict[str, Any]] = []
        for result in results:
            if result.status == VerificationTaskStatus.VERIFIED:
                continue
            for finding in result.findings:
                original = finding.claim_or_question.strip()
                if not original or len(original) < 20:
                    continue
                calibrated = self.calibrate_claim(
                    original,
                    status=result.status,
                    reasons=result.reasons,
                )
                if calibrated == original:
                    continue
                suggestion_id = "CR-" + hashlib.sha1(
                    f"{result.task_id}:{original}:{calibrated}".encode("utf-8")
                ).hexdigest()[:10]
                suggestions.append(
                    {
                        "suggestion_id": suggestion_id,
                        "task_id": result.task_id,
                        "status": result.status.value,
                        "original": original,
                        "calibrated": calibrated,
                        "reason": finding.explanation,
                    }
                )
        return _dedupe(suggestions)

    def suggestions_for_batch(self, batch: VerificationBatch) -> list[dict[str, Any]]:
        return self.suggestions_for_results(batch.results)

    def calibrate_claim(
        self,
        claim: str,
        *,
        status: VerificationTaskStatus,
        reasons: list[str] | None = None,
    ) -> str:
        clean = " ".join(claim.split()).strip()
        if not clean:
            return clean
        softened = _soften_absolute_language(clean)
        if _RECOMMEND_RE.search(softened):
            softened = _rewrite_recommendation(softened)
        elif _CAUSAL_RE.search(softened):
            softened = _rewrite_causal(softened)
        elif status in {
            VerificationTaskStatus.UNSUPPORTED,
            VerificationTaskStatus.NOT_ENOUGH_INFORMATION,
        }:
            softened = (
                "Available local artifacts do not fully verify that "
                f"{softened[0].lower()}{softened[1:]}"
            )
        elif status == VerificationTaskStatus.CONTRADICTED:
            softened = (
                "Local artifacts contain conflicting evidence about this point; do not rely on "
                f"the claim that {softened[0].lower()}{softened[1:]} until it is resolved."
            )
        elif status == VerificationTaskStatus.PARTIALLY_VERIFIED:
            softened = (
                "Available evidence partially supports that "
                f"{softened[0].lower()}{softened[1:]}"
            )

        if reasons and status != VerificationTaskStatus.VERIFIED:
            caveat = _caveat_for_status(status)
            if caveat and caveat.lower() not in softened.lower():
                softened = softened.rstrip(".") + f", {caveat}."
        return softened


def _soften_absolute_language(text: str) -> str:
    replacements = {
        "always": "may",
        "never": "does not consistently",
        "best": "potentially better suited",
        "must": "should generally",
        "guarantees": "may support",
        "guarantee": "may support",
        "proves": "suggests",
        "clearly": "appears to",
        "definitely": "appears to",
        "undeniably": "appears to",
        "only": "primary",
    }

    def repl(match: re.Match[str]) -> str:
        value = match.group(0)
        replacement = replacements.get(value.lower(), value)
        return replacement.capitalize() if value[:1].isupper() else replacement

    return _ABSOLUTE_RE.sub(repl, text)


def _rewrite_recommendation(text: str) -> str:
    lowered = text[0].lower() + text[1:] if text else text
    return (
        "A defensible recommendation would require weighing the cited evidence, constraints, "
        f"and tradeoffs; based on current local artifacts, {lowered}"
    )


def _rewrite_causal(text: str) -> str:
    lowered = text[0].lower() + text[1:] if text else text
    return (
        "The available evidence may be consistent with this relationship, but causality should "
        f"be treated cautiously: {lowered}"
    )


def _caveat_for_status(status: VerificationTaskStatus) -> str:
    if status == VerificationTaskStatus.UNSUPPORTED:
        return "but local source support was not found"
    if status == VerificationTaskStatus.NOT_ENOUGH_INFORMATION:
        return "but more targeted source evidence is needed"
    if status == VerificationTaskStatus.CONTRADICTED:
        return "and conflicting local evidence must be resolved"
    if status == VerificationTaskStatus.PARTIALLY_VERIFIED:
        return "but the wording should be narrowed to the evidence"
    return ""


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = item["original"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
