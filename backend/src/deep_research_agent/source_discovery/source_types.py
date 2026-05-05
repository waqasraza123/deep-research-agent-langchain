from __future__ import annotations

import re

from .contracts import SearchQueryPlan, SourceType


def _add_once(items: list[SourceType], value: SourceType) -> None:
    if value not in items:
        items.append(value)


def plan_source_types(
    question: str, query_plan: SearchQueryPlan
) -> tuple[list[SourceType], list[SourceType], list[SourceType], list[str]]:
    low = question.lower()
    required: list[SourceType] = []
    preferred: list[SourceType] = []
    optional: list[SourceType] = []
    rationale: list[str] = []

    _add_once(required, "official_docs")
    rationale.append("Official or primary documentation is required as a baseline source type.")

    if query_plan.technical:
        _add_once(required, "source_code_repository")
        _add_once(preferred, "release_notes")
        _add_once(preferred, "benchmark_report")
        rationale.append(
            "Technical backend questions need repository, release, or benchmark evidence."
        )

    if query_plan.comparative:
        _add_once(preferred, "benchmark_report")
        _add_once(preferred, "tutorial_or_blog")
        _add_once(optional, "forum_discussion")
        rationale.append(
            "Comparative questions benefit from evaluation reports plus secondary analysis."
        )

    if query_plan.freshness_required:
        _add_once(required, "release_notes")
        _add_once(preferred, "company_announcement")
        rationale.append(
            "Freshness appears material, so release notes or announcements "
            "are required when available."
        )

    if query_plan.legal_or_policy or re.search(r"\b(law|regulation|policy|compliance)\b", low):
        _add_once(required, "government_or_policy")
        _add_once(required, "legal_or_regulatory")
        rationale.append(
            "Legal or policy claims need government, regulatory, or legal primary sources."
        )

    if query_plan.academic or re.search(r"\b(paper|study|literature|academic)\b", low):
        _add_once(required, "academic_paper")
        _add_once(optional, "dataset")
        rationale.append(
            "Research or literature questions need academic papers and sometimes datasets."
        )

    if re.search(r"\b(dataset|data set|corpus|benchmark data)\b", low):
        _add_once(required, "dataset")
        rationale.append("Dataset terminology indicates dataset sources should be acquired.")

    for source_type in ("tutorial_or_blog", "forum_discussion", "dataset", "unknown"):
        _add_once(optional, source_type)  # type: ignore[arg-type]

    preferred = [s for s in preferred if s not in required]
    optional = [s for s in optional if s not in required and s not in preferred]
    return required, preferred, optional, rationale
