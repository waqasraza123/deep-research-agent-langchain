from __future__ import annotations

import hashlib

from .contracts import (
    HypothesisGraph,
    HypothesisGraphEdge,
    HypothesisGraphNode,
    HypothesisSet,
)
from .evidence_tester import evidence_by_id


def build_hypothesis_graph(hypothesis_set: HypothesisSet) -> HypothesisGraph:
    nodes: dict[str, HypothesisGraphNode] = {}
    edges: list[HypothesisGraphEdge] = []
    evidence = evidence_by_id(hypothesis_set)
    updates = {item.hypothesis_id: item for item in hypothesis_set.confidence_updates}

    def node(node_id: str, node_type: str, label: str, **metadata) -> None:
        nodes.setdefault(
            node_id,
            HypothesisGraphNode(
                node_id=node_id,
                node_type=node_type,
                label=label[:220],
                metadata={key: value for key, value in metadata.items() if value is not None},
            ),
        )

    def edge(source: str, target: str, relationship: str, weight: float = 1.0, **metadata) -> None:
        edges.append(
            HypothesisGraphEdge(
                source=source,
                target=target,
                relationship=relationship,
                weight=max(0.0, min(weight, 1.0)),
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )

    node("question", "question", hypothesis_set.question)
    for hypothesis in hypothesis_set.hypotheses:
        node(
            hypothesis.hypothesis_id,
            "hypothesis",
            hypothesis.text,
            status=hypothesis.status.value,
            hypothesis_type=hypothesis.hypothesis_type.value,
            origin=hypothesis.origin,
        )
        edge("question", hypothesis.hypothesis_id, "proposes")
        for subquestion_id in hypothesis.subquestion_ids:
            if not subquestion_id:
                continue
            sid = f"subquestion:{subquestion_id}"
            node(sid, "subquestion", subquestion_id)
            edge(sid, hypothesis.hypothesis_id, "frames")
        for evidence_id in hypothesis.evidence_ids:
            ev = evidence.get(evidence_id)
            if ev is None:
                continue
            node(
                ev.evidence_id,
                "evidence",
                ev.matched_text,
                stance=ev.stance,
                score=ev.score,
                source_id=ev.source_id,
                claim_id=ev.claim_id,
            )
            edge(hypothesis.hypothesis_id, ev.evidence_id, ev.stance, weight=ev.score)
            if ev.source_id:
                source_node = f"source:{ev.source_id}"
                node(source_node, "source", ev.title or ev.url or ev.source_id, url=ev.url)
                edge(ev.evidence_id, source_node, "from_source", weight=ev.source_quality or 0.5)
            if ev.claim_id:
                claim_node = f"claim:{ev.claim_id}"
                node(claim_node, "claim", ev.claim_id)
                edge(ev.evidence_id, claim_node, "from_claim")
        update = updates.get(hypothesis.hypothesis_id)
        if update is not None:
            node(
                update.update_id,
                "confidence_update",
                update.confidence_level.value,
                posterior_score=update.posterior_score,
                needs_human_review=update.needs_human_review,
            )
            edge(
                hypothesis.hypothesis_id,
                update.update_id,
                "confidence_updated",
                weight=update.posterior_score,
            )
        for competing_id in hypothesis.competing_hypothesis_ids:
            edge(hypothesis.hypothesis_id, competing_id, "competes_with", weight=0.75)

    for contradiction in hypothesis_set.contradictions:
        node(
            contradiction.contradiction_id,
            "contradiction",
            contradiction.explanation,
            severity=contradiction.severity,
            contradiction_type=contradiction.contradiction_type,
        )
        for hypothesis_id in contradiction.hypothesis_ids:
            edge(hypothesis_id, contradiction.contradiction_id, "has_contradiction")
        for claim_id in contradiction.claim_ids:
            claim_node = f"claim:{claim_id}"
            node(claim_node, "claim", claim_id)
            edge(contradiction.contradiction_id, claim_node, "involves_claim")

    graph_id = "HG-" + hashlib.sha1(
        f"{hypothesis_set.thread_id}:{len(nodes)}:{len(edges)}".encode("utf-8")
    ).hexdigest()[:12]
    return HypothesisGraph(
        graph_id=graph_id,
        thread_id=hypothesis_set.thread_id,
        generated_at=hypothesis_set.generated_at,
        nodes=sorted(nodes.values(), key=lambda item: item.node_id),
        edges=_dedupe_edges(edges),
        warnings=list(hypothesis_set.summary.warnings),
    )


def _dedupe_edges(edges: list[HypothesisGraphEdge]) -> list[HypothesisGraphEdge]:
    seen: set[tuple[str, str, str]] = set()
    out: list[HypothesisGraphEdge] = []
    for item in edges:
        key = (item.source, item.target, item.relationship)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return sorted(out, key=lambda item: (item.source, item.relationship, item.target))
