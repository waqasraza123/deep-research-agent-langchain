from __future__ import annotations

import hashlib

from .contracts import ArgumentMap, ArgumentNode, ArgumentRelation, ResearchFinding

_WEAK_LABELS = {"weak", "unsupported", "unknown"}
_STRONG_LABELS = {"source_backed", "strong", "moderate"}


def build_argument_map(
    *,
    thread_id: str,
    question: str,
    generated_at: str,
    findings: list[ResearchFinding],
) -> ArgumentMap:
    warnings: list[str] = []
    main_answer = _main_answer(findings)
    nodes: list[ArgumentNode] = [
        ArgumentNode(
            node_id="A-main",
            kind="main_answer",
            text=main_answer,
            finding_ids=[f.finding_id for f in _top_findings(findings, limit=3)],
            source_ids=sorted({sid for f in findings for sid in f.source_ids}),
            confidence_label=_overall_confidence(findings),
            requires_human_review=any(f.requires_human_review for f in findings),
        )
    ]
    relations: list[ArgumentRelation] = []

    strong_findings = [
        f for f in findings if f.confidence_label in _STRONG_LABELS and f.claim_type != "question"
    ]
    for finding in _top_findings(strong_findings, limit=8):
        node = _node("supporting_claim", finding)
        nodes.append(node)
        relations.append(
            _relation(node.node_id, "A-main", "supports", "Finding supports the answer.")
        )

    for finding in _top_findings(
        [
            f
            for f in findings
            if f.contradiction_status != "none" or f.confidence_label == "contradicted"
        ],
        limit=6,
    ):
        node = _node("counterclaim", finding)
        nodes.append(node)
        relations.append(
            _relation(node.node_id, "A-main", "challenges", "Finding may contradict the answer.")
        )

    for finding in _top_findings(
        [
            f
            for f in findings
            if f.confidence_label in _WEAK_LABELS
            and f.claim_type not in {"question", "recommendation"}
        ],
        limit=8,
    ):
        node = _node("weak_evidence", finding)
        nodes.append(node)
        relations.append(
            _relation(node.node_id, "A-main", "qualifies", "Finding has limited support.")
        )

    for finding in _top_findings([f for f in findings if f.claim_type == "question"], limit=6):
        node = _node("unresolved_question", finding)
        nodes.append(node)
        relations.append(
            _relation(node.node_id, "A-main", "qualifies", "Open question limits certainty.")
        )

    for finding in _top_findings(
        [f for f in findings if f.claim_type == "recommendation"], limit=5
    ):
        node = _node("implication", finding)
        nodes.append(node)
        relations.append(
            _relation(node.node_id, "A-main", "implies", "Recommendation affects action.")
        )

    risk_findings = [
        f
        for f in findings
        if f.claim_type == "risk"
        or any(term in f.normalized_text for term in ("risk", "failure", "limitation", "avoid"))
    ]
    for finding in _top_findings(risk_findings, limit=6):
        node = _node("risk", finding)
        nodes.append(node)
        relations.append(_relation(node.node_id, "A-main", "qualifies", "Risk bounds the answer."))

    if not findings:
        warnings.append("No findings were available; argument map contains only a gap statement.")
    if len(nodes) == 1:
        warnings.append("No supporting, weak, counter, risk, or question nodes were detected.")

    return ArgumentMap(
        thread_id=thread_id,
        question=question,
        generated_at=generated_at,
        main_answer=main_answer,
        nodes=_dedupe_nodes(nodes),
        relations=_dedupe_relations(relations),
        warnings=warnings,
    )


def _main_answer(findings: list[ResearchFinding]) -> str:
    top = _top_findings([f for f in findings if f.claim_type != "question"], limit=1)
    if top:
        return top[0].text
    return "No answer can be assembled from the available findings."


def _top_findings(findings: list[ResearchFinding], *, limit: int) -> list[ResearchFinding]:
    return sorted(findings, key=_finding_sort_key)[:limit]


def _finding_sort_key(finding: ResearchFinding) -> tuple[int, int, str]:
    confidence_rank = {
        "source_backed": 0,
        "strong": 1,
        "moderate": 2,
        "weak": 3,
        "unknown": 4,
        "unsupported": 5,
        "contradicted": 6,
    }.get(finding.confidence_label, 4)
    contradiction_rank = 1 if finding.contradiction_status != "none" else 0
    return (contradiction_rank, confidence_rank, finding.finding_id)


def _node(kind: str, finding: ResearchFinding) -> ArgumentNode:
    return ArgumentNode(
        node_id=f"A-{hashlib.sha1(f'{kind}:{finding.finding_id}'.encode('utf-8')).hexdigest()[:10]}",
        kind=kind,  # type: ignore[arg-type]
        text=finding.text,
        finding_ids=[finding.finding_id],
        source_ids=finding.source_ids,
        confidence_label=finding.confidence_label,
        requires_human_review=finding.requires_human_review,
    )


def _relation(source: str, target: str, relation_type: str, rationale: str) -> ArgumentRelation:
    digest = hashlib.sha1(f"{source}:{target}:{relation_type}".encode("utf-8")).hexdigest()[:10]
    return ArgumentRelation(
        relation_id=f"R-{digest}",
        source_node_id=source,
        target_node_id=target,
        relation_type=relation_type,  # type: ignore[arg-type]
        rationale=rationale,
    )


def _overall_confidence(findings: list[ResearchFinding]) -> str:
    if not findings:
        return "unknown"
    labels = [f.confidence_label for f in findings]
    if "contradicted" in labels:
        return "contradicted"
    if any(label in {"source_backed", "strong"} for label in labels):
        return "moderate" if any(label in _WEAK_LABELS for label in labels) else "strong"
    if any(label == "moderate" for label in labels):
        return "moderate"
    if any(label == "weak" for label in labels):
        return "weak"
    if any(label == "unsupported" for label in labels):
        return "unsupported"
    return "unknown"


def _dedupe_nodes(nodes: list[ArgumentNode]) -> list[ArgumentNode]:
    seen: set[tuple[str, str]] = set()
    out: list[ArgumentNode] = []
    for node in nodes:
        key = (node.kind, node.text.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(node)
    return out


def _dedupe_relations(relations: list[ArgumentRelation]) -> list[ArgumentRelation]:
    seen: set[tuple[str, str, str]] = set()
    out: list[ArgumentRelation] = []
    for relation in relations:
        key = (relation.source_node_id, relation.target_node_id, relation.relation_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(relation)
    return out
