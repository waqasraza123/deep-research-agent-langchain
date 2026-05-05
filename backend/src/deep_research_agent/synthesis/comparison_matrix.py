from __future__ import annotations

import hashlib
import re

from deep_research_agent.evidence.claim_extractor import tokenize

from .contracts import (
    ComparisonCell,
    ComparisonDimension,
    ComparisonMatrix,
    ResearchFinding,
    SynthesisInput,
)

_COMPARATIVE_PATTERNS = (
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\btrade[- ]?off",
    r"\bwhich\b.+\bbetter\b",
    r"\bbetween\b.+\band\b",
)

_DEFAULT_DIMENSIONS = [
    ("capability", "Capabilities", "Functional capability or feature coverage."),
    ("integration", "Integration", "Integration effort and compatibility."),
    ("performance", "Performance", "Performance, latency, scale, or reliability evidence."),
    ("cost", "Cost", "Cost, operational burden, or complexity evidence."),
    ("maturity", "Maturity", "Adoption, maintenance, stability, and ecosystem evidence."),
    ("risk", "Risk", "Security, operational, legal, or delivery risk evidence."),
]

_DIMENSION_TERMS = {
    "performance": {"performance", "latency", "speed", "throughput", "scale", "scaling"},
    "cost": {"cost", "pricing", "price", "cheap", "expensive", "complexity", "burden"},
    "integration": {"integration", "api", "compatible", "migration", "deploy", "architecture"},
    "maturity": {"maturity", "adoption", "community", "stable", "maintenance", "release"},
    "risk": {"risk", "security", "failure", "limitation", "compliance", "privacy"},
    "capability": {"feature", "capability", "supports", "workflow", "tooling", "quality"},
}


def is_comparative_question(question: str) -> bool:
    normalized = question.lower()
    return any(re.search(pattern, normalized) for pattern in _COMPARATIVE_PATTERNS)


def infer_comparison_options(question: str, findings: list[ResearchFinding]) -> list[str]:
    options = _options_from_question(question)
    if len(options) >= 2:
        return options[:6]

    entity_counts: dict[str, int] = {}
    for finding in findings:
        for entity in finding.entities:
            if len(entity) < 2:
                continue
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
    ranked = sorted(entity_counts, key=lambda item: (-entity_counts[item], item.lower()))
    for entity in ranked:
        if entity.lower() not in {opt.lower() for opt in options}:
            options.append(entity)
        if len(options) >= 2:
            break
    return options[:6]


def build_comparison_matrix(
    synthesis_input: SynthesisInput,
    findings: list[ResearchFinding],
) -> ComparisonMatrix:
    detected = is_comparative_question(synthesis_input.question) or any(
        f.claim_type == "comparative" for f in findings
    )
    options = infer_comparison_options(synthesis_input.question, findings) if detected else []
    warnings: list[str] = []
    if detected and len(options) < 2:
        warnings.append("Comparative intent detected, but fewer than two options were found.")

    dimensions = _infer_dimensions(synthesis_input, findings) if detected else []
    cells: list[ComparisonCell] = []
    for option in options:
        for dimension in dimensions:
            matched = _match_cell_findings(option, dimension.dimension_id, findings)
            cells.append(
                ComparisonCell(
                    option=option,
                    dimension_id=dimension.dimension_id,
                    finding_ids=[f.finding_id for f in matched],
                    summary=_cell_summary(option, dimension.label, matched),
                    confidence_label=_cell_confidence(matched),
                    source_ids=sorted({sid for f in matched for sid in f.source_ids}),
                    gaps=[]
                    if matched
                    else [f"No direct finding for {option} on {dimension.label}."],
                )
            )

    return ComparisonMatrix(
        thread_id=synthesis_input.thread_id,
        question=synthesis_input.question,
        generated_at=synthesis_input.generated_at,
        detected=detected,
        options=options,
        dimensions=dimensions,
        cells=cells,
        warnings=warnings,
    )


def _infer_dimensions(
    synthesis_input: SynthesisInput,
    findings: list[ResearchFinding],
) -> list[ComparisonDimension]:
    selected: dict[str, ComparisonDimension] = {}
    question_tokens = set(tokenize(synthesis_input.question))
    for dim_id, label, rationale in _DEFAULT_DIMENSIONS:
        if question_tokens & _DIMENSION_TERMS[dim_id]:
            selected[dim_id] = ComparisonDimension(
                dimension_id=dim_id,
                label=label,
                rationale=rationale,
            )

    for finding in findings:
        tokens = set(tokenize(finding.text))
        for dim_id, terms in _DIMENSION_TERMS.items():
            if tokens & terms:
                if dim_id not in selected:
                    label = next(label for did, label, _r in _DEFAULT_DIMENSIONS if did == dim_id)
                    rationale = next(r for did, _l, r in _DEFAULT_DIMENSIONS if did == dim_id)
                    selected[dim_id] = ComparisonDimension(
                        dimension_id=dim_id,
                        label=label,
                        rationale=rationale,
                    )
                selected[dim_id].finding_ids.append(finding.finding_id)

    for subquestion in synthesis_input.subquestions:
        sq_text = str(subquestion.get("question") or "")
        sq_tokens = set(tokenize(sq_text))
        for dim_id, terms in _DIMENSION_TERMS.items():
            if sq_tokens & terms and dim_id not in selected:
                label = next(label for did, label, _r in _DEFAULT_DIMENSIONS if did == dim_id)
                rationale = next(r for did, _l, r in _DEFAULT_DIMENSIONS if did == dim_id)
                selected[dim_id] = ComparisonDimension(
                    dimension_id=dim_id,
                    label=label,
                    rationale=f"{rationale} Inferred from strategy subquestions.",
                )

    if not selected:
        for dim_id, label, rationale in _DEFAULT_DIMENSIONS[:4]:
            selected[dim_id] = ComparisonDimension(
                dimension_id=dim_id,
                label=label,
                rationale=f"{rationale} Default dimension for comparative synthesis.",
            )

    return list(selected.values())[:8]


def _match_cell_findings(
    option: str,
    dimension_id: str,
    findings: list[ResearchFinding],
) -> list[ResearchFinding]:
    option_key = option.lower()
    terms = _DIMENSION_TERMS.get(dimension_id, set())
    matched: list[ResearchFinding] = []
    for finding in findings:
        text = finding.normalized_text
        entity_match = option_key in text or any(
            entity.lower() == option_key for entity in finding.entities
        )
        if not entity_match:
            continue
        token_match = bool(set(tokenize(finding.text)) & terms) if terms else True
        if token_match or finding.claim_type == "comparative":
            matched.append(finding)
    return sorted(matched, key=lambda f: f.finding_id)[:4]


def _cell_summary(option: str, dimension_label: str, findings: list[ResearchFinding]) -> str:
    if not findings:
        return f"No artifact-backed assessment for {option} on {dimension_label}."
    return " ".join(f.text for f in findings[:2])


def _cell_confidence(findings: list[ResearchFinding]) -> str:
    if not findings:
        return "unknown"
    labels = {f.confidence_label for f in findings}
    if "contradicted" in labels:
        return "contradicted"
    if labels & {"source_backed", "strong"}:
        return "moderate" if labels & {"weak", "unsupported", "unknown"} else "strong"
    if "moderate" in labels:
        return "moderate"
    if "weak" in labels:
        return "weak"
    if "unsupported" in labels:
        return "unsupported"
    return "unknown"


def _options_from_question(question: str) -> list[str]:
    text = question.strip()
    patterns = [
        r"(.+?)\s+vs\.?\s+(.+)",
        r"(.+?)\s+versus\s+(.+)",
        r"between\s+(.+?)\s+and\s+(.+)",
        r"compare\s+(.+?)\s+and\s+(.+)",
        r"compare\s+(.+?)\s+with\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        left = _clean_option(match.group(1))
        right = _clean_option(match.group(2))
        options = [item for item in (left, right) if item]
        if len(options) >= 2:
            return _dedupe_options(options)
    return []


def _clean_option(value: str) -> str:
    value = re.sub(
        r"^(?:should\s+we\s+|should\s+i\s+|should\s+)?"
        r"(?:choose|compare|recommend|adopt|use)\s+",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"\b(?:for|in|on|as|when|which|is|are|better|best).*$", "", value, flags=re.I)
    value = value.strip(" ?.,:;\"'")
    if len(value.split()) > 5:
        tokens = re.findall(r"\b[A-Z][A-Za-z0-9_.+#-]*\b|[a-zA-Z0-9_.+-]*AI\b|llama\.cpp", value)
        if tokens:
            value = " ".join(tokens[-2:])
    return value[:80].strip()


def _dedupe_options(options: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for option in options:
        key = option.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(option)
    return out


def dimension_id_for_label(label: str) -> str:
    return hashlib.sha1(label.lower().encode("utf-8")).hexdigest()[:10]
