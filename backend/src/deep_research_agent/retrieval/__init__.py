from __future__ import annotations

from pathlib import Path

from .artifact_writer import RETRIEVAL_ARTIFACTS, write_retrieval_artifacts
from .context_pack import build_all_context_packs, render_agent_context_block
from .contracts import (
    ContextPack,
    ContextPackBuildResult,
    ContextPackItem,
    HybridRankingConfig,
    RetrievalChunk,
    RetrievalCoverageSummary,
    RetrievalDocument,
    RetrievalIndex,
    RetrievalQuery,
    RetrievalResult,
    RetrievalScore,
    model_to_plain,
)
from .embeddings import DisabledEmbeddingProvider, EmbeddingProvider, MockEmbeddingProvider
from .hybrid_ranker import rank_all_queries, rank_retrieval_results
from .indexer import build_retrieval_index
from .query_planner import plan_retrieval_queries


def rebuild_retrieval_artifacts(
    run_dir: Path,
    *,
    thread_id: str | None,
    question: str,
    config: HybridRankingConfig | None = None,
    chunk_chars: int = 1600,
    overlap_chars: int = 180,
    context_pack_max_chars: int | None = None,
    write_artifacts: bool = True,
) -> ContextPackBuildResult:
    index = build_retrieval_index(
        run_dir,
        thread_id=thread_id,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )
    queries = plan_retrieval_queries(question)
    results = rank_all_queries(index, queries, config=config or HybridRankingConfig())
    build_result = build_all_context_packs(
        thread_id=thread_id,
        question=question,
        index=index,
        queries=queries,
        results=results,
        context_pack_max_chars=context_pack_max_chars,
    )
    if write_artifacts:
        write_retrieval_artifacts(run_dir, build_result)
    return build_result


__all__ = [
    "ContextPack",
    "ContextPackBuildResult",
    "ContextPackItem",
    "DisabledEmbeddingProvider",
    "EmbeddingProvider",
    "HybridRankingConfig",
    "MockEmbeddingProvider",
    "RETRIEVAL_ARTIFACTS",
    "RetrievalChunk",
    "RetrievalCoverageSummary",
    "RetrievalDocument",
    "RetrievalIndex",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalScore",
    "build_retrieval_index",
    "model_to_plain",
    "plan_retrieval_queries",
    "rank_all_queries",
    "rank_retrieval_results",
    "rebuild_retrieval_artifacts",
    "render_agent_context_block",
    "write_retrieval_artifacts",
]
