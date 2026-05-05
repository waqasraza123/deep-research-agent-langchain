from __future__ import annotations

import hashlib

from .contracts import MemoryGraph, MemoryGraphEdge, MemoryGraphNode, MemoryRecord
from .repository import now_iso_utc


def _node_id(kind: str, value: str) -> str:
    digest = hashlib.sha1(value.lower().encode("utf-8")).hexdigest()[:14]
    return f"{kind}:{digest}"


def build_memory_graph(records: list[MemoryRecord]) -> MemoryGraph:
    nodes: dict[str, MemoryGraphNode] = {}
    edges: dict[tuple[str, str, str], MemoryGraphEdge] = {}
    source_to_records: dict[str, list[MemoryRecord]] = {}

    def add_node(node: MemoryGraphNode) -> None:
        nodes.setdefault(node.id, node)

    def add_edge(edge: MemoryGraphEdge) -> None:
        key = (edge.source, edge.target, edge.relation)
        existing = edges.get(key)
        if existing is None:
            edges[key] = edge
        else:
            existing.weight = round(existing.weight + edge.weight, 3)

    for record in records:
        run_id = _node_id("run", record.thread_id)
        question_id = _node_id("question", record.normalized_question)
        source_key = record.canonical_url or record.normalized_url or record.source_url
        source_id = _node_id("source", source_key)

        add_node(MemoryGraphNode(id=run_id, type="run", label=record.thread_id))
        add_node(
            MemoryGraphNode(
                id=question_id,
                type="question",
                label=record.question[:160],
                metadata={"normalized_question": record.normalized_question},
            )
        )
        add_node(
            MemoryGraphNode(
                id=source_id,
                type="source",
                label=record.source_title or record.source_url,
                metadata={
                    "url": record.source_url,
                    "canonical_url": record.canonical_url,
                    "domain": record.source_domain,
                    "quality_score": record.quality_score,
                    "content_hash": record.content_hash,
                },
            )
        )
        add_edge(MemoryGraphEdge(source=run_id, target=question_id, relation="asked"))
        add_edge(MemoryGraphEdge(source=question_id, target=source_id, relation="used_source"))

        source_to_records.setdefault(source_key, []).append(record)

        for entity in record.entities[:25]:
            entity_id = _node_id(f"entity:{entity.entity_type.value}", entity.name)
            add_node(
                MemoryGraphNode(
                    id=entity_id,
                    type=f"entity:{entity.entity_type.value}",
                    label=entity.name,
                    metadata={
                        "confidence": entity.confidence,
                        "mentions": entity.mentions,
                    },
                )
            )
            add_edge(
                MemoryGraphEdge(
                    source=source_id,
                    target=entity_id,
                    relation="mentions",
                    weight=max(0.1, entity.confidence),
                )
            )

            for topic in record.topics[:10]:
                topic_id = _node_id("topic", topic.name)
                add_node(
                    MemoryGraphNode(
                        id=topic_id,
                        type="topic",
                        label=topic.name,
                        metadata={"score": topic.score, "keywords": topic.keywords},
                    )
                )
                add_edge(
                    MemoryGraphEdge(
                        source=entity_id,
                        target=topic_id,
                        relation="associated_topic",
                        weight=max(0.1, topic.score),
                    )
                )
                add_edge(
                    MemoryGraphEdge(
                        source=topic_id,
                        target=run_id,
                        relation="appears_in_run",
                        weight=max(0.1, topic.score),
                    )
                )

    warnings: list[str] = []
    for source_key, source_records in source_to_records.items():
        threads = sorted({record.thread_id for record in source_records})
        if len(threads) <= 1:
            continue
        source_id = _node_id("source", source_key)
        for thread_id in threads:
            run_id = _node_id("run", thread_id)
            add_edge(
                MemoryGraphEdge(
                    source=source_id,
                    target=run_id,
                    relation="repeated_across_runs",
                    weight=float(len(threads)),
                    metadata={"thread_ids": threads},
                )
            )

    stale = [record for record in records if any("stale" in w.lower() for w in record.warnings)]
    for record in stale:
        warnings.append(f"Stale source warning for {record.source_url} in {record.thread_id}.")

    return MemoryGraph(
        generated_at=now_iso_utc(),
        nodes=sorted(nodes.values(), key=lambda node: (node.type, node.label)),
        edges=sorted(edges.values(), key=lambda edge: (edge.source, edge.relation, edge.target)),
        warnings=warnings,
    )
