from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent_factory import AgentService
from .artifacts import (
    INTERNAL_FILES,
    artifact_abs_path,
    ensure_required_artifacts,
    ensure_thread_dir,
    list_artifacts,
    write_strategy_artifacts,
)
from .evaluation import EVALUATION_ARTIFACTS, rebuild_evaluation_artifacts
from .evaluation.benchmark import list_benchmark_cases
from .evaluation.contracts import model_to_plain as evaluation_model_to_plain
from .evaluation.regression_runner import run_regression_suite
from .evidence import rebuild_evidence_artifacts
from .intelligence import ResearchStrategy, create_research_strategy
from .intelligence_summary import rebuild_intelligence_summary_artifacts
from .logging_config import configure_logging
from .memory import (
    ArtifactReference,
    MemoryRecord,
    MemoryRepository,
    MemoryRetriever,
    SourceCache,
    build_memory_graph,
    extract_entities_and_topics,
    write_memory_context_artifacts,
    write_memory_graph_artifacts,
)
from .memory.repository import normalize_question, now_iso_utc, source_domain
from .model import create_chat_model
from .orchestration import OrchestrationExecutor
from .orchestration.artifact_writer import read_orchestration_json
from .runs.cleanup import apply_cleanup_plan, build_cleanup_plan
from .runs.contracts import ReviewState, RunCancellationRequest, RunStatus, model_to_dict
from .runs.lifecycle import RunLifecycle
from .runs.repository import (
    RunCancelledError,
    RunListFilters,
    RunNotFoundError,
    RunRepository,
    settings_snapshot_from_object,
)
from .runs.review import approve_review, get_review, reject_review, request_changes
from .runs.serializers import run_detail, run_summary
from .runs.state_machine import InvalidRunTransitionError
from .runtime.budgets import BudgetExceeded, read_budget_file
from .runtime.contracts import RetryPolicy, RunBudget
from .runtime.diagnostics import build_runtime_diagnostics
from .runtime.events import read_event_file
from .runtime.mock_model import write_mock_research_artifacts
from .runtime.model_registry import build_model_registry
from .runtime.retries import retry_sync
from .runtime.run_context import RunContext
from .settings import Settings
from .source_audit import (
    audit_sources,
    audit_sources_from_manifest,
    write_source_audit_artifacts,
)
from .source_audit.contracts import model_to_plain
from .source_identity import source_identity_from_dict
from .source_intelligence import CrawlBudgetStats, CrawlResult, SourceRecord, crawl_sources
from .source_intelligence.dedupe import content_hash, normalize_url
from .source_intelligence.source_graph import write_source_graph_artifacts, write_sources_manifest
from .synthesis import SYNTHESIS_ARTIFACTS, rebuild_synthesis_artifacts

log = logging.getLogger("deep_research_agent.api")


class RunRequest(BaseModel):
    question: str = Field(..., min_length=5)
    urls: list[str] = []
    thread_id: str | None = None
    generate_strategy: bool = True
    require_review: bool | None = None

    max_sources: int = Field(default=1, ge=0, le=3)
    max_links_per_source: int | None = Field(default=None, ge=0, le=10)
    follow_links: bool | None = None
    mock_mode: bool = False
    allow_mock_fallback: bool = False
    budget: RunBudget | None = None


class PlanRequest(BaseModel):
    question: str = Field(..., min_length=5)
    urls: list[str] = []
    thread_id: str | None = None
    persist: bool = True


class OrchestrationPreviewRequest(BaseModel):
    question: str = Field(..., min_length=5)
    urls: list[str] = []
    thread_id: str | None = None
    generate_strategy: bool = True
    follow_links: bool | None = None
    max_links_per_source: int | None = Field(default=None, ge=0, le=10)
    available_source_count: int = Field(default=0, ge=0, le=50)


class ReviewActionRequest(BaseModel):
    reviewer: str = Field(..., min_length=1)
    notes: str = ""
    requested_changes: list[str] = Field(default_factory=list)


class CleanupApplyRequest(BaseModel):
    thread_ids: list[str]
    confirm_delete: bool = False


class BenchmarkRunRequest(BaseModel):
    case_ids: list[str] = Field(default_factory=list)


class SourceAuditRequest(BaseModel):
    question: str = Field(..., min_length=5)
    thread_id: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    persist: bool = True


def _safe_local_rel(local_path: str) -> str | None:
    if not local_path:
        return None
    if local_path.startswith("/") or ".." in local_path or "\\" in local_path:
        return None
    if "runs/" not in local_path:
        return None
    rel = local_path.split("runs/", 1)[-1]
    if not rel or rel.startswith("/") or ".." in rel or "\\" in rel:
        return None
    return rel


def _read_text(path, *, max_chars: int) -> str:
    try:
        if not path.exists() or path.is_dir():
            return ""
        s = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not s:
            return ""
        if len(s) > max_chars:
            s = s[:max_chars] + "\n\n[TRUNCATED]\n"
        return s
    except Exception:
        return ""


def _read_json_artifact(td, rel_path: str) -> dict[str, Any]:
    path = td / rel_path
    if not path.exists() or path.is_dir():
        raise HTTPException(status_code=404, detail=f"{rel_path} not found")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid {rel_path}: {e}") from e
    if not isinstance(loaded, dict):
        raise HTTPException(status_code=500, detail=f"{rel_path} did not contain a JSON object")
    return loaded


def _load_json_file(path) -> Any:
    try:
        if not path.exists() or path.is_dir():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _memory_data_dir(settings: Settings):
    return settings.memory_data_dir or settings.runs_dir / "_memory"


def _memory_repository(settings: Settings) -> MemoryRepository:
    return MemoryRepository(_memory_data_dir(settings))


def _model_dump_jsonable(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def _relative_run_artifact(thread_id: str, local_path: str | None) -> str | None:
    if not local_path:
        return None
    marker = f"runs/{thread_id}/"
    if marker not in local_path:
        return None
    rel = local_path.split(marker, 1)[-1]
    if rel.startswith("/") or ".." in rel or "\\" in rel:
        return None
    return rel


def _summarize_text(text: str, *, max_chars: int = 600) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    cut = clean[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(".,;:") + "..."


def _memory_record_from_source(
    *,
    settings: Settings,
    thread_id: str,
    question: str,
    source: SourceRecord,
) -> MemoryRecord | None:
    rel_path = _relative_run_artifact(thread_id, source.local_path)
    if not rel_path:
        return None
    source_path = artifact_abs_path(settings.runs_dir, thread_id, rel_path)
    text = _read_text(source_path, max_chars=60_000)
    if not text or not source.ok or source.skipped:
        return None

    source_url = source.final_url or source.url
    identity = source_identity_from_dict(source.to_dict())
    normalized = source.normalized_url or normalize_url(source_url)
    canonical = normalize_url(source.canonical_url) if source.canonical_url else None
    text_hash = content_hash(text)
    memory_id = hashlib.sha1(
        f"{thread_id}|{normalized}|{text_hash}".encode("utf-8")
    ).hexdigest()
    extraction = extract_entities_and_topics(
        question=question,
        text=text,
        title=source.title,
        url=source_url,
    )
    quality = (
        source.quality_score.final_quality_score
        if source.quality_score is not None
        else source.priority_score
    )
    if quality is not None:
        quality = max(0.0, min(float(quality), 1.0))
    now = now_iso_utc()
    artifacts = [
        ArtifactReference(thread_id=thread_id, path=rel_path, artifact_type="source_text"),
        ArtifactReference(
            thread_id=thread_id,
            path="sources.json",
            artifact_type="source_manifest",
        ),
    ]
    if (settings.runs_dir / thread_id / "report.md").exists():
        artifacts.append(
            ArtifactReference(thread_id=thread_id, path="report.md", artifact_type="report")
        )
    warnings = list(extraction.warnings)
    if source.truncated:
        warnings.append("Source text was truncated during fetch.")
    return MemoryRecord(
        memory_id=memory_id,
        thread_id=thread_id,
        question=question,
        normalized_question=normalize_question(question),
        source_url=source_url,
        source_id=identity.source_id,
        source_identity=identity,
        normalized_url=normalized,
        canonical_url=canonical,
        source_title=source.title,
        source_domain=source_domain(source_url),
        content_hash=text_hash,
        extracted_text_hash=text_hash,
        source_type=source.document_kind or source.content_type,
        first_seen_at=now,
        last_seen_at=now,
        run_count=1,
        quality_score=quality,
        entities=extraction.entities,
        topics=extraction.topics,
        summary=_summarize_text(text),
        warnings=warnings,
        artifacts=artifacts,
    )


def _source_record_from_manifest_item(item: dict[str, Any]) -> SourceRecord:
    return SourceRecord(
        url=str(item.get("url") or ""),
        normalized_url=str(item.get("normalized_url") or item.get("url") or ""),
        source_kind=str(item.get("source_kind") or "root"),
        parent_url=item.get("parent_url"),
        crawl_depth=int(item.get("crawl_depth") or 0),
        ok=bool(item.get("ok")),
        skipped=bool(item.get("skipped")),
        skip_reason=item.get("skip_reason"),
        duplicate_of=item.get("duplicate_of"),
        final_url=item.get("final_url"),
        canonical_url=item.get("canonical_url"),
        title=item.get("title"),
        content_type=item.get("content_type"),
        status_code=item.get("status_code"),
        truncated=bool(item.get("truncated")),
        fetched_at=item.get("fetched_at"),
        local_path=item.get("local_path"),
        strategy=item.get("strategy"),
        word_count=int(item.get("word_count") or 0),
        char_count=int(item.get("char_count") or 0),
        document_kind=item.get("document_kind"),
        priority_score=item.get("priority_score"),
        priority_reasons=tuple(item.get("priority_reasons") or ()),
        discovered_anchor_text=str(item.get("discovered_anchor_text") or ""),
        source_id=item.get("source_id"),
    )


def _update_memory_store(
    *,
    settings: Settings,
    repository: MemoryRepository,
    thread_id: str,
    question: str,
    sources: list[SourceRecord],
) -> list[MemoryRecord]:
    stored: list[MemoryRecord] = []
    for source in sources:
        record = _memory_record_from_source(
            settings=settings,
            thread_id=thread_id,
            question=question,
            source=source,
        )
        if record is None:
            continue
        stored.append(repository.upsert(record))
    graph_records = repository.list_records(limit=1000)
    td = ensure_thread_dir(settings.runs_dir, thread_id)
    write_memory_graph_artifacts(td, build_memory_graph(graph_records))
    return stored


def _rebuild_memory_for_thread(
    *,
    settings: Settings,
    repository: MemoryRepository,
    thread_id: str,
    question: str,
) -> list[MemoryRecord]:
    td = ensure_thread_dir(settings.runs_dir, thread_id)
    manifest = td / "sources.json"
    if not manifest.exists():
        return []
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        data = []
    sources = [
        _source_record_from_manifest_item(item)
        for item in data
        if isinstance(item, dict) and item.get("local_path")
    ]
    return _update_memory_store(
        settings=settings,
        repository=repository,
        thread_id=thread_id,
        question=question,
        sources=sources,
    )


def _prior_context_prompt(context) -> str:
    if not context.confidence_score:
        return ""
    lines = [
        "Prior local memory context (not new evidence; verify against fresh fetched sources):",
    ]
    for record in context.similar_previous_questions[:3]:
        lines.append(f"- Previous question `{record.thread_id}`: {record.question}")
    for record in context.previously_useful_sources[:3]:
        label = record.source_title or record.source_url
        lines.append(f"- Previously useful source: {label} ({record.source_url})")
    for warning in context.stale_warnings[:3]:
        lines.append(f"- Memory warning: {warning}")
    if context.known_topics:
        topics = ", ".join(topic.name for topic in context.known_topics[:8])
        lines.append(f"- Known topics: {topics}")
    if context.known_entities:
        entities = ", ".join(entity.name for entity in context.known_entities[:8])
        lines.append(f"- Known entities: {entities}")
    return "\n".join(lines)


def _prefetch_sources(
    settings: Settings,
    td,
    thread_id: str,
    urls: list[str],
    *,
    question: str,
    follow_links: bool,
    max_links_per_source: int,
    run_context: RunContext | None = None,
) -> CrawlResult:
    def on_fetch_start(url: str, source_kind: str) -> None:
        if not run_context:
            return
        if source_kind == "discovered":
            run_context.budget.increment_crawl_expansion()
            run_context.persist_budget()
        run_context.source_fetch_started(url)

    def on_fetch_complete(source: SourceRecord) -> None:
        if not run_context:
            return
        meta = source.to_dict()
        if source.ok and not source.skipped:
            run_context.source_fetch_completed(source.url, meta)
        elif source.skipped:
            run_context.log("source_fetch_failed", message=source.url, metadata=meta)
        else:
            run_context.source_fetch_failed(source.url, source.skip_reason or "fetch failed")
        if source.local_path:
            rel_path = source.local_path.split(f"runs/{thread_id}/", 1)[-1]
            run_context.artifact_written(rel_path, None)

    budget_max = run_context.budget.budget.max_crawl_expansion if run_context else 20
    result = crawl_sources(
        question=question,
        root_urls=urls,
        thread_id=thread_id,
        thread_dir=td,
        timeout_s=settings.http_timeout_s,
        max_chars=settings.max_page_chars,
        follow_links=follow_links,
        max_links_per_source=max_links_per_source,
        global_link_budget=min(max(0, budget_max), max(0, len(urls) * max_links_per_source), 20),
        on_fetch_start=on_fetch_start,
        on_fetch_complete=on_fetch_complete,
    )
    if run_context:
        run_context.artifact_written("source_graph.json", (td / "source_graph.json").stat().st_size)
        run_context.artifact_written("source_graph.md", (td / "source_graph.md").stat().st_size)
    return result


def _build_deterministic_report(td) -> str:
    notes = _read_text(td / "notes.md", max_chars=4000)
    sources_json = _read_text(td / "sources.json", max_chars=2000)
    body = "# Report\n\n"
    if notes:
        body += notes.strip() + "\n\n"
    else:
        body += "Notes were not produced. This report was generated from available artifacts.\n\n"
    body += "## Sources\n\n"
    if sources_json:
        body += "```json\n" + sources_json.strip() + "\n```\n\n"
    body += "## Conclusion\n\n"
    body += "This report is grounded only in captured sources and artifacts. [S1]\n"
    return body


def _strategy_response(
    settings: Settings,
    *,
    question: str,
    urls: list[str],
    thread_id: str | None,
    persist: bool,
) -> dict[str, Any]:
    tid = thread_id or str(uuid.uuid4())
    strategy = create_research_strategy(question, urls)
    artifacts: list[dict[str, Any]] = []
    if persist:
        ensure_thread_dir(settings.runs_dir, tid)
        artifacts = [a.__dict__ for a in write_strategy_artifacts(settings.runs_dir, tid, strategy)]
    return {
        "thread_id": tid,
        "strategy": strategy.to_json_dict(),
        "artifacts": artifacts,
    }


def _ensure_report_with_model(
    settings: Settings,
    td,
    question: str,
    sources_meta: list[dict[str, Any]],
    run_context: RunContext | None = None,
) -> None:
    report_path = td / "report.md"
    if report_path.exists():
        return

    usable = [
        m
        for m in sources_meta
        if isinstance(m, dict) and m.get("ok") is True and isinstance(m.get("local_path"), str)
    ]
    if not usable:
        report_path.write_text(_build_deterministic_report(td), encoding="utf-8")
        if run_context:
            run_context.artifact_written("report.md", report_path.stat().st_size)
        return

    m = usable[0]
    rel = _safe_local_rel(m["local_path"])
    if not rel:
        report_path.write_text(_build_deterministic_report(td), encoding="utf-8")
        if run_context:
            run_context.artifact_written("report.md", report_path.stat().st_size)
        return

    src_fs = (settings.runs_dir / rel).resolve()
    src_text = _read_text(src_fs, max_chars=8000)
    if not src_text:
        report_path.write_text(_build_deterministic_report(td), encoding="utf-8")
        if run_context:
            run_context.artifact_written("report.md", report_path.stat().st_size)
        return

    notes = _read_text(td / "notes.md", max_chars=2500)
    src_url = m.get("final_url") or m.get("url") or ""
    src_title = m.get("title") or ""

    prompt = (
        "Write report.md as Markdown using only the provided source text and notes.\n"
        "Requirements:\n"
        "- Title\n"
        "- Exactly 6 bullet points\n"
        "- 1-line conclusion\n"
        "- Cite claims using [S1]\n"
        "- No outside knowledge\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Source S1 URL: {src_url}\nTitle: {src_title}\n\n"
        f"Notes:\n{notes}\n\n"
        f"Source text:\n{src_text}\n"
    )

    try:
        if run_context:
            model_name = (
                settings.ollama_model
                if settings.model_provider == "ollama"
                else settings.openai_model
            )
            run_context.model_call_started(
                provider=settings.model_provider,
                model_name=model_name,
                purpose="ensure report artifact",
            )
        model = create_chat_model(settings)
        policy = RetryPolicy(max_attempts=max(1, settings.openai_max_retries + 1))
        msg, retry_meta = retry_sync(
            lambda: model.invoke([{"role": "user", "content": prompt}]), policy
        )
        content = (getattr(msg, "content", "") or "").strip()
        if run_context:
            run_context.model_call_completed(
                generated_chars=len(content),
                metadata={"purpose": "ensure report artifact", "retry": retry_meta},
            )
        if len(content) >= 200:
            report_path.write_text(content + "\n", encoding="utf-8")
            if run_context:
                run_context.artifact_written("report.md", report_path.stat().st_size)
            return
    except BudgetExceeded:
        raise
    except Exception as e:
        if run_context:
            run_context.log(
                "model_call_failed",
                message="ensure report artifact",
                metadata={"error": f"{type(e).__name__}: {e}"},
            )
        log.exception("ensure_report_with_model failed")

    report_path.write_text(_build_deterministic_report(td), encoding="utf-8")
    if run_context:
        run_context.artifact_written("report.md", report_path.stat().st_size)


def _normalize_dt(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _advance_to_building_evidence(
    repository: RunRepository,
    lifecycle: RunLifecycle,
    thread_id: str,
) -> None:
    order = [
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.FETCHING_SOURCES,
        RunStatus.ANALYZING,
        RunStatus.WRITING_REPORT,
        RunStatus.BUILDING_EVIDENCE,
    ]
    current = repository.get(thread_id).status
    if current == RunStatus.BUILDING_EVIDENCE:
        return
    if current not in order:
        raise InvalidRunTransitionError(current, RunStatus.BUILDING_EVIDENCE)
    for status in order[order.index(current) + 1 :]:
        lifecycle.transition(status)


def _write_mock_source_artifacts(
    *,
    thread_dir,
    thread_id: str,
    urls: list[str],
) -> CrawlResult:
    result = CrawlResult(root_urls=urls)
    result.budget = CrawlBudgetStats(
        root_count=len(urls),
        global_link_budget=0,
        max_links_per_source=0,
        max_depth=0,
    )
    result.sources = [
        SourceRecord(
            url=url,
            normalized_url=url,
            source_kind="root",
            ok=False,
            skipped=True,
            skip_reason="mock_mode_not_fetched",
            title="Mock source placeholder",
            source_id=f"S{idx}",
        )
        for idx, url in enumerate(urls, start=1)
    ]
    if not result.sources:
        result.sources = [
            SourceRecord(
                url="mock://no-source-provided",
                normalized_url="mock://no-source-provided",
                source_kind="root",
                ok=False,
                skipped=True,
                skip_reason="no_source_provided",
                title="No source provided",
                source_id="S1",
            )
        ]
    result.budget.skipped_count = len(result.sources)
    write_source_graph_artifacts(thread_dir, result)
    write_sources_manifest(thread_dir / "sources.json", result.sources)
    return result


def _try_rebuild_evaluation(
    td,
    thread_id: str,
    warnings: list[str],
    run_context: RunContext | None = None,
) -> None:
    try:
        evaluation = rebuild_evaluation_artifacts(td, thread_id=thread_id)
    except Exception as e:
        log.exception("evaluation rebuild failed")
        warnings.append(f"Evaluation artifacts were not generated: {type(e).__name__}: {e}")
        return
    if run_context:
        for rel_path in EVALUATION_ARTIFACTS:
            path = td / rel_path
            run_context.artifact_written(rel_path, path.stat().st_size if path.exists() else None)
        run_context.log(
            "artifact_written",
            message="evaluation",
            metadata={
                "overall_score": evaluation.overall_score,
                "confidence": evaluation.confidence,
            },
        )


def _try_rebuild_intelligence_summary(
    *,
    settings: Settings,
    td,
    thread_id: str,
    warnings: list[str],
    run_context: RunContext | None = None,
) -> None:
    try:
        rebuild_intelligence_summary_artifacts(td, runs_dir=settings.runs_dir, thread_id=thread_id)
    except Exception as e:
        log.exception("intelligence summary rebuild failed")
        warnings.append(
            f"Intelligence summary artifacts were not generated: {type(e).__name__}: {e}"
        )
        return
    if run_context:
        for rel_path in ("intelligence_summary.json", "intelligence_summary.md"):
            path = td / rel_path
            run_context.artifact_written(rel_path, path.stat().st_size if path.exists() else None)


def _write_audit_for_sources(
    *,
    thread_dir,
    thread_id: str,
    question: str,
    sources: list[SourceRecord] | list[dict[str, Any]],
    run_context: RunContext | None = None,
):
    source_dicts = [s.to_dict() if hasattr(s, "to_dict") else s for s in sources]
    batch = audit_sources(
        source_dicts,
        question=question,
        thread_id=thread_id,
        thread_dir=thread_dir,
    )
    for rel_path in write_source_audit_artifacts(thread_dir, batch):
        if run_context:
            path = thread_dir / rel_path
            run_context.artifact_written(rel_path, path.stat().st_size if path.exists() else None)
    return batch


def create_app(*, settings: Settings | None = None, service: AgentService | None = None) -> FastAPI:
    configure_logging()
    settings = settings or Settings.load()
    service = service or AgentService(settings)
    run_repository = RunRepository(settings.runs_dir)
    memory_repository = _memory_repository(settings)
    memory_source_cache = SourceCache(
        memory_repository,
        stale_after_days=settings.memory_stale_after_days,
    )
    memory_retriever = MemoryRetriever(memory_repository, memory_source_cache)
    orchestration_executor = OrchestrationExecutor(settings.runs_dir)

    app = FastAPI(title="Deep Research Agent")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/models")
    def models() -> list[dict[str, Any]]:
        return [m.dict() for m in build_model_registry(settings)]

    @app.get("/runtime/diagnostics")
    def diagnostics() -> dict[str, Any]:
        return build_runtime_diagnostics(settings).dict()

    @app.post("/run")
    def run(req: RunRequest) -> dict[str, Any]:
        thread_id = req.thread_id or str(uuid.uuid4())
        td = ensure_thread_dir(settings.runs_dir, thread_id)
        effective_settings = settings
        if req.mock_mode:
            effective_settings = replace(settings, model_provider="mock")
        if req.allow_mock_fallback and not effective_settings.allow_mock_fallback:
            effective_settings = replace(effective_settings, allow_mock_fallback=True)
        run_context = RunContext(
            thread_id=thread_id,
            thread_dir=td,
            budget=req.budget or effective_settings.default_budget(),
        )
        run_context.log(
            "run_started",
            message=req.question.strip(),
            metadata={
                "provider": effective_settings.model_provider,
                "mock_mode": effective_settings.model_provider == "mock",
            },
        )

        urls = [u.strip() for u in req.urls if u and u.strip()]
        urls = urls[: max(0, min(req.max_sources, 3))] if urls else []
        follow_links = (
            effective_settings.default_follow_links
            if req.follow_links is None
            else bool(req.follow_links)
        )
        max_links_per_source = (
            effective_settings.default_max_links_per_source
            if req.max_links_per_source is None
            else int(req.max_links_per_source)
        )
        max_links_per_source = max(0, min(max_links_per_source, 10))
        require_review = (
            effective_settings.review_gate_default
            if req.require_review is None
            else bool(req.require_review)
        )
        run_repository.create(
            thread_id=thread_id,
            question=req.question.strip(),
            urls=urls,
            settings_snapshot={
                **settings_snapshot_from_object(effective_settings),
                "max_sources": max(0, min(req.max_sources, 3)),
                "max_links_per_source": max_links_per_source,
                "follow_links": follow_links,
                "generate_strategy": bool(req.generate_strategy),
                "require_review": require_review,
                "mock_mode": effective_settings.model_provider == "mock",
            },
            require_review=require_review,
        )
        memory_context = memory_retriever.retrieve(question=req.question.strip(), urls=urls)
        write_memory_context_artifacts(td, memory_context)
        for rel_path in ("memory_context.json", "memory_context.md"):
            path = td / rel_path
            run_context.artifact_written(rel_path, path.stat().st_size if path.exists() else None)
        if memory_context.confidence_score > 0:
            run_context.log(
                "memory_context_loaded",
                message="prior memory context available",
                metadata={
                    "confidence_score": memory_context.confidence_score,
                    "similar_questions": len(memory_context.similar_previous_questions),
                    "source_reuse_candidates": len(
                        memory_context.suggested_source_reuse_candidates
                    ),
                },
            )
        lifecycle = RunLifecycle(run_repository, thread_id)

        strategy: ResearchStrategy | None = None
        try:
            lifecycle.transition(RunStatus.PLANNING)
            if req.generate_strategy:
                strategy = create_research_strategy(req.question.strip(), urls)
                strategy_artifacts = write_strategy_artifacts(
                    effective_settings.runs_dir, thread_id, strategy
                )
                run_context.log(
                    "strategy_created",
                    message=strategy.research_id,
                    metadata={"research_id": strategy.research_id},
                )
                for artifact in strategy_artifacts:
                    run_context.artifact_written(artifact.path, artifact.size_bytes)
                lifecycle.checkpoint()

            lifecycle.transition(RunStatus.FETCHING_SOURCES)
            if effective_settings.model_provider == "mock":
                sources_meta = [
                    {"ok": False, "url": u, "title": "Mock source placeholder", "mock": True}
                    for u in urls
                ]
                metadata = write_mock_research_artifacts(
                    thread_dir=td,
                    thread_id=thread_id,
                    question=req.question.strip(),
                    sources_meta=sources_meta,
                )
                mock_crawl_result = _write_mock_source_artifacts(
                    thread_dir=td, thread_id=thread_id, urls=urls
                )
                _write_audit_for_sources(
                    thread_dir=td,
                    thread_id=thread_id,
                    question=req.question.strip(),
                    sources=mock_crawl_result.sources,
                    run_context=run_context,
                )
                _update_memory_store(
                    settings=effective_settings,
                    repository=memory_repository,
                    thread_id=thread_id,
                    question=req.question.strip(),
                    sources=mock_crawl_result.sources,
                )
                graph = orchestration_executor.build_graph(
                    thread_id=thread_id,
                    question=req.question.strip(),
                    urls=urls,
                    strategy=strategy,
                    follow_links=follow_links,
                    max_links_per_source=max_links_per_source,
                    available_source_count=len(mock_crawl_result.usable_sources()),
                )
                _, _, orchestration_summary = orchestration_executor.execute(
                    graph,
                    strategy=strategy.to_json_dict() if strategy else None,
                    persist=True,
                )
                for rel_path in orchestration_summary.artifact_paths:
                    path = td / rel_path
                    run_context.artifact_written(
                        rel_path,
                        path.stat().st_size if path.exists() else None,
                    )
                for rel_path in (
                    "plan.md",
                    "notes.md",
                    "sources.json",
                    "source_graph.json",
                    "source_graph.md",
                    "source_audit.json",
                    "source_audit.md",
                    "source_rankings.json",
                    "source_warnings.md",
                    "citation_readiness.json",
                    "memory_context.json",
                    "memory_context.md",
                    "memory_graph.json",
                    "memory_graph.md",
                    "report.md",
                    "metadata.json",
                ):
                    path = td / rel_path
                    run_context.artifact_written(
                        rel_path,
                        path.stat().st_size if path.exists() else None,
                    )
                lifecycle.transition(RunStatus.ANALYZING)
                lifecycle.transition(RunStatus.WRITING_REPORT)
                lifecycle.transition(RunStatus.BUILDING_EVIDENCE)
                ledger = rebuild_evidence_artifacts(td, thread_id=thread_id)
                for rel_path in (
                    "evidence_ledger.json",
                    "evidence_ledger.md",
                    "unsupported_claims.md",
                    "contradictions.md",
                    "citation_map.json",
                    "evidence_coverage.json",
                ):
                    path = td / rel_path
                    run_context.artifact_written(
                        rel_path,
                        path.stat().st_size if path.exists() else None,
                    )
                run_context.log(
                    "artifact_written",
                    message="evidence_coverage",
                    metadata={"total_claims": ledger.coverage.total_claims},
                )
                warnings: list[str] = []
                try:
                    synthesis = rebuild_synthesis_artifacts(td, thread_id=thread_id)
                    for rel_path in (*SYNTHESIS_ARTIFACTS, "report.raw.md"):
                        path = td / rel_path
                        if path.exists():
                            run_context.artifact_written(rel_path, path.stat().st_size)
                    run_context.log(
                        "artifact_written",
                        message="synthesis_output",
                        metadata={
                            "findings": len(synthesis.findings),
                            "profile": synthesis.report_assembly_plan.profile,
                        },
                    )
                except Exception as e:
                    log.exception("synthesis rebuild failed")
                    warnings.append(
                        f"Synthesis artifacts were not generated: {type(e).__name__}: {e}"
                    )
                _try_rebuild_evaluation(td, thread_id, warnings, run_context)
                _try_rebuild_intelligence_summary(
                    settings=effective_settings,
                    td=td,
                    thread_id=thread_id,
                    warnings=warnings,
                    run_context=run_context,
                )
                run_context.budget_warning()
                run_context.log("run_completed", message="mock run completed", metadata=metadata)
                run_repository.set_warnings(thread_id, warnings)
                run_repository.refresh_artifacts(thread_id)
                run_repository.set_output_summary(
                    thread_id,
                    budget_summary=read_budget_file(td / "budget.json"),
                )
                lifecycle.complete(
                    require_review=require_review,
                    summary="[MOCK OUTPUT] Deterministic offline run completed.",
                )
                _try_rebuild_intelligence_summary(
                    settings=effective_settings,
                    td=td,
                    thread_id=thread_id,
                    warnings=warnings,
                    run_context=run_context,
                )
                run_repository.refresh_artifacts(thread_id)
                return {
                    "thread_id": thread_id,
                    "summary": "[MOCK OUTPUT] Deterministic offline run completed.",
                    "strategy": strategy.to_json_dict() if strategy else None,
                    "warnings": warnings,
                    "artifacts": [
                        a.__dict__ for a in list_artifacts(effective_settings.runs_dir, thread_id)
                    ],
                    "hint": f"Report should be at runs/{thread_id}/report.md",
                    "mock": True,
                    "budget": read_budget_file(td / "budget.json"),
                    "run": run_summary(run_repository.get(thread_id)),
                }

            crawl_result = _prefetch_sources(
                effective_settings,
                td,
                thread_id,
                urls,
                question=req.question.strip(),
                follow_links=follow_links,
                max_links_per_source=max_links_per_source,
                run_context=run_context,
            )
            sources_meta = crawl_result.to_sources_json()
            source_audit_batch = _write_audit_for_sources(
                thread_dir=td,
                thread_id=thread_id,
                question=req.question.strip(),
                sources=crawl_result.sources,
                run_context=run_context,
            )
            agent_source_urls = [source.url for source in crawl_result.usable_sources()]
            usable_source_count = len(crawl_result.usable_sources())
            graph = orchestration_executor.build_graph(
                thread_id=thread_id,
                question=req.question.strip(),
                urls=urls,
                strategy=strategy,
                follow_links=follow_links,
                max_links_per_source=max_links_per_source,
                available_source_count=usable_source_count,
            )
            _, _, orchestration_summary = orchestration_executor.execute(
                graph,
                strategy=strategy.to_json_dict() if strategy else None,
                persist=True,
            )
            for rel_path in orchestration_summary.artifact_paths:
                path = td / rel_path
                run_context.artifact_written(
                    rel_path,
                    path.stat().st_size if path.exists() else None,
                )
            lifecycle.checkpoint()

            user_msg = req.question.strip()
            if agent_source_urls:
                user_msg += "\n\nSources (call fetch_and_store on these):\n" + "\n".join(
                    f"- {u}" for u in agent_source_urls
                )
            if strategy is not None:
                user_msg += (
                    "\n\nResearch strategy to follow:\n" + strategy.agent_instructions.strip()
                )
            if orchestration_summary.agent_instruction_block:
                user_msg += (
                    "\n\nAdaptive orchestration context:\n"
                    + orchestration_summary.agent_instruction_block
                )
            if source_audit_batch.summary.instruction_block:
                user_msg += (
                    "\n\nSource audit context:\n"
                    + source_audit_batch.summary.instruction_block
                )
            prior_context = _prior_context_prompt(memory_context)
            if prior_context:
                user_msg += "\n\n" + prior_context
            user_msg += (
                "\n\nRules:\n"
                "- Use only fetched sources.\n"
                "- Do not use outside knowledge.\n"
                "- Treat memory context as prior background only, not as fresh evidence.\n"
                "- Write all required files."
            )

            lifecycle.transition(RunStatus.ANALYZING)
            active_service = (
                service if effective_settings is settings else AgentService(effective_settings)
            )
            agent = active_service.build_agent(
                thread_id,
                max_sources=max(0, min(req.max_sources, 3)),
                max_links_per_source=max_links_per_source,
                follow_links=follow_links,
                run_context=run_context,
            )
            model_name = (
                effective_settings.ollama_model
                if effective_settings.model_provider == "ollama"
                else effective_settings.openai_model
            )
            run_context.model_call_started(
                provider=effective_settings.model_provider,
                model_name=model_name,
                purpose="agent orchestration",
            )
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_msg}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            run_context.model_call_completed(
                generated_chars=len(str(result)) if result is not None else 0,
                metadata={"purpose": "agent orchestration"},
            )
            lifecycle.transition(RunStatus.WRITING_REPORT)
            write_sources_manifest(td / "sources.json", crawl_result.sources)
            _ensure_report_with_model(
                effective_settings, td, req.question.strip(), sources_meta, run_context
            )
            stored_memory = _update_memory_store(
                settings=effective_settings,
                repository=memory_repository,
                thread_id=thread_id,
                question=req.question.strip(),
                sources=crawl_result.sources,
            )
            for rel_path in ("memory_graph.json", "memory_graph.md"):
                path = td / rel_path
                run_context.artifact_written(
                    rel_path,
                    path.stat().st_size if path.exists() else None,
                )
            if stored_memory:
                run_context.log(
                    "memory_store_updated",
                    message=f"stored {len(stored_memory)} source memories",
                    metadata={"memory_ids": [record.memory_id for record in stored_memory]},
                )
            lifecycle.transition(RunStatus.BUILDING_EVIDENCE)
        except BudgetExceeded as e:
            run_context.budget_exceeded(e)
            run_context.log("run_failed", message=str(e), metadata={"reasons": e.reasons})
            run_repository.record_error(thread_id, e, fail_run=True)
            raise HTTPException(
                status_code=429,
                detail={"error": str(e), "thread_id": thread_id, "reasons": e.reasons},
            ) from e
        except Exception as e:
            log.exception("agent.invoke failed")
            if isinstance(e, RunCancelledError):
                run_context.log("run_failed", message=str(e), metadata={"cancelled": True})
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": str(e),
                        "thread_id": thread_id,
                        "limitation": (
                            "Cancellation is cooperative and checked between major stages; "
                            "it cannot interrupt an in-flight synchronous agent.invoke call."
                        ),
                    },
                ) from e
            run_context.log(
                "model_call_failed",
                message="agent orchestration",
                metadata={"error": f"{type(e).__name__}: {e}"},
            )
            if effective_settings.allow_mock_fallback:
                metadata = write_mock_research_artifacts(
                    thread_dir=td,
                    thread_id=thread_id,
                    question=req.question.strip(),
                    sources_meta=locals().get("sources_meta", []),
                )
                metadata["fallback_reason"] = f"{type(e).__name__}: {e}"
                run_context.log(
                    "run_completed", message="explicit mock fallback used", metadata=metadata
                )
                _advance_to_building_evidence(run_repository, lifecycle, thread_id)
                fallback_warnings = ["Explicit mock fallback used after model failure."]
                fallback_warnings.extend(ensure_required_artifacts(settings.runs_dir, thread_id))
                try:
                    rebuild_evidence_artifacts(td, thread_id=thread_id)
                except Exception as evidence_error:
                    log.exception("evidence rebuild failed")
                    fallback_warnings.append(
                        "Evidence artifacts were not generated: "
                        f"{type(evidence_error).__name__}: {evidence_error}"
                    )
                try:
                    rebuild_synthesis_artifacts(td, thread_id=thread_id)
                except Exception as synthesis_error:
                    log.exception("synthesis rebuild failed")
                    fallback_warnings.append(
                        "Synthesis artifacts were not generated: "
                        f"{type(synthesis_error).__name__}: {synthesis_error}"
                    )
                _try_rebuild_evaluation(td, thread_id, fallback_warnings, run_context)
                _try_rebuild_intelligence_summary(
                    settings=settings,
                    td=td,
                    thread_id=thread_id,
                    warnings=fallback_warnings,
                    run_context=run_context,
                )
                run_repository.set_warnings(thread_id, fallback_warnings)
                run_repository.refresh_artifacts(thread_id)
                run_repository.set_output_summary(
                    thread_id,
                    budget_summary=read_budget_file(td / "budget.json"),
                )
                lifecycle.complete(
                    require_review=require_review,
                    summary=("[MOCK OUTPUT] Explicit mock fallback completed after model failure."),
                )
                _try_rebuild_intelligence_summary(
                    settings=settings,
                    td=td,
                    thread_id=thread_id,
                    warnings=fallback_warnings,
                    run_context=run_context,
                )
                run_repository.refresh_artifacts(thread_id)
                fallback_summary = (
                    "[MOCK OUTPUT] Explicit mock fallback completed after model failure."
                )
                return {
                    "thread_id": thread_id,
                    "summary": fallback_summary,
                    "strategy": strategy.to_json_dict() if strategy else None,
                    "warnings": fallback_warnings,
                    "artifacts": [a.__dict__ for a in list_artifacts(settings.runs_dir, thread_id)],
                    "hint": f"Report should be at runs/{thread_id}/report.md",
                    "mock": True,
                    "budget": read_budget_file(td / "budget.json"),
                    "run": run_summary(run_repository.get(thread_id)),
                }
            run_repository.record_error(thread_id, e, fail_run=True)
            warnings = ensure_required_artifacts(settings.runs_dir, thread_id)
            for artifact in list_artifacts(settings.runs_dir, thread_id):
                run_context.artifact_written(artifact.path, artifact.size_bytes)
            run_context.log("run_failed", message=f"{type(e).__name__}: {e}")
            run_repository.set_warnings(thread_id, warnings)
            run_repository.refresh_artifacts(thread_id)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": f"{type(e).__name__}: {e}",
                    "thread_id": thread_id,
                    "warnings": warnings,
                },
            ) from e

        summary_text = ""
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            last = result["messages"][-1]
            if isinstance(last, dict):
                summary_text = last.get("content") or ""
            else:
                summary_text = getattr(last, "content", "") or ""

        warnings = ensure_required_artifacts(settings.runs_dir, thread_id)
        try:
            rebuild_evidence_artifacts(td, thread_id=thread_id)
        except Exception as e:
            log.exception("evidence rebuild failed")
            warnings.append(f"Evidence artifacts were not generated: {type(e).__name__}: {e}")
        try:
            synthesis = rebuild_synthesis_artifacts(td, thread_id=thread_id)
            run_context.log(
                "artifact_written",
                message="synthesis_output",
                metadata={
                    "findings": len(synthesis.findings),
                    "profile": synthesis.report_assembly_plan.profile,
                },
            )
        except Exception as e:
            log.exception("synthesis rebuild failed")
            warnings.append(f"Synthesis artifacts were not generated: {type(e).__name__}: {e}")
        _try_rebuild_evaluation(td, thread_id, warnings, run_context)
        _try_rebuild_intelligence_summary(
            settings=settings,
            td=td,
            thread_id=thread_id,
            warnings=warnings,
            run_context=run_context,
        )
        for artifact in list_artifacts(settings.runs_dir, thread_id):
            run_context.artifact_written(artifact.path, artifact.size_bytes)
        run_context.budget_warning()
        run_context.log("run_completed", message="run completed")
        run_repository.set_warnings(thread_id, warnings)
        run_repository.refresh_artifacts(thread_id)
        run_repository.set_output_summary(
            thread_id, budget_summary=read_budget_file(td / "budget.json")
        )
        lifecycle.complete(require_review=require_review, summary=summary_text)
        _try_rebuild_intelligence_summary(
            settings=settings,
            td=td,
            thread_id=thread_id,
            warnings=warnings,
            run_context=run_context,
        )
        run_repository.refresh_artifacts(thread_id)

        return {
            "thread_id": thread_id,
            "summary": summary_text,
            "strategy": strategy.to_json_dict() if strategy else None,
            "warnings": warnings,
            "artifacts": [a.__dict__ for a in list_artifacts(settings.runs_dir, thread_id)],
            "hint": f"Report should be at runs/{thread_id}/report.md",
            "mock": False,
            "budget": read_budget_file(td / "budget.json"),
            "run": run_summary(run_repository.get(thread_id)),
        }

    @app.post("/runs/{thread_id}/evidence/rebuild")
    def evidence_rebuild(thread_id: str) -> dict[str, Any]:
        try:
            td = ensure_thread_dir(settings.runs_dir, thread_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        warnings = ensure_required_artifacts(settings.runs_dir, thread_id)
        try:
            ledger = rebuild_evidence_artifacts(td, thread_id=thread_id)
        except Exception as e:
            log.exception("evidence rebuild failed")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

        return {
            "thread_id": thread_id,
            "warnings": warnings,
            "coverage": ledger.coverage.model_dump()
            if hasattr(ledger.coverage, "model_dump")
            else ledger.coverage.dict(),
            "artifacts": [a.__dict__ for a in list_artifacts(settings.runs_dir, thread_id)],
        }

    @app.post("/runs/{thread_id}/synthesis/rebuild")
    def synthesis_rebuild(thread_id: str) -> dict[str, Any]:
        try:
            td = ensure_thread_dir(settings.runs_dir, thread_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        warnings = ensure_required_artifacts(settings.runs_dir, thread_id)
        try:
            output = rebuild_synthesis_artifacts(td, thread_id=thread_id)
        except Exception as e:
            log.exception("synthesis rebuild failed")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

        if run_repository.get_or_none(thread_id) is not None:
            run_repository.set_warnings(thread_id, warnings + output.warnings)
            run_repository.refresh_artifacts(thread_id)

        return {
            "thread_id": thread_id,
            "warnings": warnings + output.warnings,
            "profile": output.report_assembly_plan.profile,
            "finding_count": len(output.findings),
            "comparison_detected": output.comparison_matrix.detected,
            "decision_detected": output.decision_memo.detected,
            "safe_to_replace_report": output.report_assembly_plan.safe_to_replace_report,
            "artifacts": [a.__dict__ for a in list_artifacts(settings.runs_dir, thread_id)],
        }

    @app.get("/runs/{thread_id}/argument-map")
    def argument_map_get(thread_id: str) -> dict[str, Any]:
        try:
            td = ensure_thread_dir(settings.runs_dir, thread_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return _read_json_artifact(td, "argument_map.json")

    @app.get("/runs/{thread_id}/comparison-matrix")
    def comparison_matrix_get(thread_id: str) -> dict[str, Any]:
        try:
            td = ensure_thread_dir(settings.runs_dir, thread_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return _read_json_artifact(td, "comparison_matrix.json")

    @app.get("/runs/{thread_id}/decision-memo")
    def decision_memo_get(thread_id: str) -> dict[str, Any]:
        try:
            td = ensure_thread_dir(settings.runs_dir, thread_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return _read_json_artifact(td, "decision_memo.json")

    @app.get("/runs/{thread_id}/uncertainty")
    def uncertainty_get(thread_id: str) -> dict[str, Any]:
        try:
            td = ensure_thread_dir(settings.runs_dir, thread_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return _read_json_artifact(td, "uncertainty_boundaries.json")

    @app.post("/runs/{thread_id}/evaluation/rebuild")
    def evaluation_rebuild(thread_id: str) -> dict[str, Any]:
        try:
            td = ensure_thread_dir(settings.runs_dir, thread_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        try:
            evaluation = rebuild_evaluation_artifacts(td, thread_id=thread_id)
        except Exception as e:
            log.exception("evaluation rebuild failed")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
        try:
            run_repository.refresh_artifacts(thread_id)
        except RunNotFoundError:
            pass
        return {
            "thread_id": thread_id,
            "evaluation": evaluation_model_to_plain(evaluation),
            "artifacts": [a.__dict__ for a in list_artifacts(settings.runs_dir, thread_id)],
        }

    @app.get("/runs/{thread_id}/evaluation")
    def evaluation_get(thread_id: str) -> dict[str, Any]:
        try:
            path = artifact_abs_path(settings.runs_dir, thread_id, "evaluation.json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if not path.exists():
            raise HTTPException(status_code=404, detail="Evaluation not found")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/runs/{thread_id}/synthesis")
    def synthesis_get(thread_id: str) -> dict[str, Any]:
        try:
            path = artifact_abs_path(settings.runs_dir, thread_id, "synthesis_output.json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if not path.exists():
            raise HTTPException(status_code=404, detail="Synthesis output not found")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/runs/{thread_id}/memory")
    def run_memory_get(thread_id: str) -> dict[str, Any]:
        try:
            td = ensure_thread_dir(settings.runs_dir, thread_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        context = _load_json_file(td / "memory_context.json")
        graph = _load_json_file(td / "memory_graph.json")
        records = memory_repository.records_for_thread(thread_id)
        if context is None and graph is None and not records:
            raise HTTPException(status_code=404, detail="Memory artifacts not found")
        return {
            "thread_id": thread_id,
            "context": context,
            "graph": graph,
            "records": [_model_dump_jsonable(record) for record in records],
        }

    @app.get("/runs/{thread_id}/intelligence-summary")
    def intelligence_summary_get(thread_id: str) -> dict[str, Any]:
        try:
            path = artifact_abs_path(settings.runs_dir, thread_id, "intelligence_summary.json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if not path.exists():
            raise HTTPException(status_code=404, detail="Intelligence summary not found")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/runs/{thread_id}/quality-score")
    def quality_score_get(thread_id: str) -> dict[str, Any]:
        try:
            path = artifact_abs_path(settings.runs_dir, thread_id, "quality_score.json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if not path.exists():
            raise HTTPException(status_code=404, detail="Quality score not found")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/benchmarks/cases")
    def benchmark_cases() -> list[dict[str, Any]]:
        return [evaluation_model_to_plain(case) for case in list_benchmark_cases()]

    @app.post("/benchmarks/run")
    def benchmark_run(req: BenchmarkRunRequest | None = None) -> dict[str, Any]:
        request = req or BenchmarkRunRequest()
        result = run_regression_suite(
            output_dir=settings.runs_dir / "_benchmarks",
            case_ids=request.case_ids or None,
        )
        return evaluation_model_to_plain(result)

    @app.get("/memory/search")
    def memory_search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)):
        return [_model_dump_jsonable(record) for record in memory_repository.search(q, limit=limit)]

    @app.get("/memory/sources")
    def memory_sources(
        domain: str | None = Query(default=None),
        limit: int = Query(100, ge=1, le=500),
    ):
        return [
            _model_dump_jsonable(record)
            for record in memory_repository.list_sources(domain=domain, limit=limit)
        ]

    @app.get("/memory/entities/{entity_name}")
    def memory_entity(entity_name: str, limit: int = Query(100, ge=1, le=500)):
        return [
            _model_dump_jsonable(record)
            for record in memory_repository.find_entity(entity_name, limit=limit)
        ]

    @app.post("/runs/{thread_id}/memory/rebuild")
    def memory_rebuild(thread_id: str) -> dict[str, Any]:
        try:
            run = run_repository.get(thread_id)
            stored = _rebuild_memory_for_thread(
                settings=settings,
                repository=memory_repository,
                thread_id=thread_id,
                question=run.question,
            )
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            log.exception("memory rebuild failed")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

        return {
            "thread_id": thread_id,
            "stored_memories": [_model_dump_jsonable(record) for record in stored],
            "artifacts": [a.__dict__ for a in list_artifacts(settings.runs_dir, thread_id)],
        }

    def _build_or_read_source_audit(thread_id: str):
        try:
            run = run_repository.get(thread_id)
            td = ensure_thread_dir(settings.runs_dir, thread_id)
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        audit_path = td / "source_audit.json"
        if audit_path.exists():
            try:
                return json.loads(audit_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        manifest = td / "sources.json"
        if not manifest.exists():
            raise HTTPException(status_code=404, detail="Sources manifest not found")
        batch = audit_sources_from_manifest(
            manifest,
            question=run.question,
            thread_id=thread_id,
        )
        write_source_audit_artifacts(td, batch)
        run_repository.refresh_artifacts(thread_id)
        return model_to_plain(batch)

    @app.post("/source-audit")
    def source_audit_create(req: SourceAuditRequest) -> dict[str, Any]:
        td = None
        if req.thread_id:
            try:
                td = ensure_thread_dir(settings.runs_dir, req.thread_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        if not req.sources and not req.thread_id:
            raise HTTPException(status_code=400, detail="Provide sources or a thread_id")

        if req.sources:
            batch = audit_sources(
                req.sources,
                question=req.question.strip(),
                thread_id=req.thread_id,
                thread_dir=td,
            )
        else:
            assert td is not None
            manifest = td / "sources.json"
            if not manifest.exists():
                raise HTTPException(status_code=404, detail="Sources manifest not found")
            batch = audit_sources_from_manifest(
                manifest,
                question=req.question.strip(),
                thread_id=req.thread_id,
            )

        if req.persist and td is not None:
            write_source_audit_artifacts(td, batch)
            if req.thread_id and run_repository.get_or_none(req.thread_id):
                run_repository.refresh_artifacts(req.thread_id)
        return model_to_plain(batch)

    @app.get("/runs/{thread_id}/source-audit")
    def source_audit_get(thread_id: str) -> dict[str, Any]:
        return _build_or_read_source_audit(thread_id)

    @app.get("/runs/{thread_id}/citation-readiness")
    def citation_readiness_get(thread_id: str) -> dict[str, Any]:
        try:
            td = ensure_thread_dir(settings.runs_dir, thread_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        path = td / "citation_readiness.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        data = _build_or_read_source_audit(thread_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "thread_id": thread_id,
            "question": data.get("question", ""),
            "generated_at": data.get("generated_at", ""),
            "sources": [
                {
                    "source_id": item.get("source_id"),
                    "url": item.get("url"),
                    "domain": item.get("domain"),
                    "title": item.get("title"),
                    "citation_readiness": item.get("citation_readiness_score"),
                    "recommended_usage": item.get("recommended_usage"),
                }
                for item in data.get("audits", [])
                if isinstance(item, dict)
            ],
            "citation_risks": data.get("summary", {}).get("citation_risks", []),
        }

    @app.get("/runs")
    def runs(
        status: RunStatus | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        has_errors: bool | None = None,
        review_status: ReviewState | None = None,
    ) -> list[dict[str, Any]]:
        filters = RunListFilters(
            status=status,
            created_after=_normalize_dt(created_after),
            created_before=_normalize_dt(created_before),
            has_errors=has_errors,
            review_status=review_status,
        )
        return [run_summary(item) for item in run_repository.list(filters)]

    @app.get("/runs/cleanup/plan")
    def cleanup_plan(
        stale_after_hours: int = Query(default=24, ge=1),
        max_dir_bytes: int = Query(default=50_000_000, ge=1),
    ) -> dict[str, Any]:
        return model_to_dict(
            build_cleanup_plan(
                run_repository,
                stale_after_hours=stale_after_hours,
                max_dir_bytes=max_dir_bytes,
            )
        )

    @app.post("/runs/cleanup/apply")
    def cleanup_apply(req: CleanupApplyRequest) -> dict[str, Any]:
        plan = build_cleanup_plan(run_repository)
        try:
            applied = apply_cleanup_plan(
                run_repository,
                plan,
                thread_ids=req.thread_ids,
                confirm_delete=req.confirm_delete,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return model_to_dict(applied)

    @app.get("/runs/{thread_id}")
    def run_get(thread_id: str) -> dict[str, Any]:
        try:
            return run_detail(run_repository.get(thread_id))
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found") from None

    @app.post("/runs/{thread_id}/cancel")
    def run_cancel(thread_id: str, req: RunCancellationRequest | None = None) -> dict[str, Any]:
        try:
            request = req or RunCancellationRequest()
            run = run_repository.request_cancellation(thread_id, request)
            return {
                **run_detail(run),
                "limitation": (
                    "Cancellation is cooperative and checked between major stages; "
                    "it cannot interrupt an in-flight synchronous agent.invoke call."
                ),
            }
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found") from None

    @app.get("/runs/{thread_id}/review")
    def review_get(thread_id: str) -> dict[str, Any]:
        try:
            return model_to_dict(get_review(run_repository, thread_id))
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found") from None

    @app.post("/runs/{thread_id}/review/approve")
    def review_approve(thread_id: str, req: ReviewActionRequest) -> dict[str, Any]:
        try:
            return model_to_dict(
                approve_review(
                    run_repository,
                    thread_id,
                    reviewer=req.reviewer,
                    notes=req.notes,
                )
            )
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except InvalidRunTransitionError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    @app.post("/runs/{thread_id}/review/request-changes")
    def review_request_changes(thread_id: str, req: ReviewActionRequest) -> dict[str, Any]:
        try:
            return model_to_dict(
                request_changes(
                    run_repository,
                    thread_id,
                    reviewer=req.reviewer,
                    notes=req.notes,
                    requested_changes=req.requested_changes,
                )
            )
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found") from None

    @app.post("/runs/{thread_id}/review/reject")
    def review_reject(thread_id: str, req: ReviewActionRequest) -> dict[str, Any]:
        try:
            return model_to_dict(
                reject_review(
                    run_repository,
                    thread_id,
                    reviewer=req.reviewer,
                    notes=req.notes,
                )
            )
        except RunNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found") from None

    @app.get("/runs/{thread_id}/events")
    def run_events(thread_id: str) -> list[dict[str, Any]]:
        try:
            td = ensure_thread_dir(settings.runs_dir, thread_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return read_event_file(td)

    @app.get("/runs/{thread_id}/budget")
    def run_budget(thread_id: str) -> dict[str, Any]:
        try:
            td = ensure_thread_dir(settings.runs_dir, thread_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        data = read_budget_file(td / "budget.json")
        if not data:
            raise HTTPException(status_code=404, detail="Budget not found")
        return data

    @app.get("/runs/{thread_id}/task-graph")
    def run_task_graph(thread_id: str) -> dict[str, Any]:
        try:
            return read_orchestration_json(settings.runs_dir, thread_id, "task_graph.json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Task graph not found") from None

    @app.get("/runs/{thread_id}/stage-outputs")
    def run_stage_outputs(thread_id: str) -> dict[str, Any]:
        try:
            return read_orchestration_json(settings.runs_dir, thread_id, "stage_outputs.json")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Stage outputs not found") from None

    @app.get("/runs/{thread_id}/artifacts")
    def run_artifacts(thread_id: str) -> list[dict[str, Any]]:
        return [a.__dict__ for a in list_artifacts(settings.runs_dir, thread_id)]

    @app.get("/runs/{thread_id}/artifacts/{rel_path:path}")
    def run_artifact_download(thread_id: str, rel_path: str):
        return _download_artifact(thread_id, rel_path)

    @app.post("/research-plan")
    def research_plan(req: PlanRequest) -> dict[str, Any]:
        urls = [u.strip() for u in req.urls if u and u.strip()]
        return _strategy_response(
            settings,
            question=req.question.strip(),
            urls=urls,
            thread_id=req.thread_id,
            persist=req.persist,
        )

    @app.post("/orchestration/preview")
    def orchestration_preview(req: OrchestrationPreviewRequest) -> dict[str, Any]:
        thread_id = req.thread_id or str(uuid.uuid4())
        urls = [u.strip() for u in req.urls if u and u.strip()]
        follow_links = (
            settings.default_follow_links if req.follow_links is None else bool(req.follow_links)
        )
        max_links_per_source = (
            settings.default_max_links_per_source
            if req.max_links_per_source is None
            else int(req.max_links_per_source)
        )
        strategy = (
            create_research_strategy(req.question.strip(), urls) if req.generate_strategy else None
        )
        graph = orchestration_executor.build_graph(
            thread_id=thread_id,
            question=req.question.strip(),
            urls=urls,
            strategy=strategy,
            follow_links=follow_links,
            max_links_per_source=max_links_per_source,
            available_source_count=req.available_source_count,
        )
        graph, outputs, summary = orchestration_executor.execute(
            graph,
            strategy=strategy.to_json_dict() if strategy else None,
            persist=False,
        )
        return {
            "thread_id": thread_id,
            "strategy": strategy.to_json_dict() if strategy else None,
            "task_graph": graph.to_json_dict(),
            "stage_outputs": [
                output.model_dump(mode="json")
                if hasattr(output, "model_dump")
                else output.dict()
                for output in outputs
            ],
            "summary": summary.to_json_dict(),
        }

    @app.post("/plan")
    def plan(req: PlanRequest) -> dict[str, Any]:
        urls = [u.strip() for u in req.urls if u and u.strip()]
        return _strategy_response(
            settings,
            question=req.question.strip(),
            urls=urls,
            thread_id=req.thread_id,
            persist=req.persist,
        )

    @app.get("/threads/{thread_id}/artifacts")
    def artifacts(thread_id: str) -> list[dict[str, Any]]:
        return [a.__dict__ for a in list_artifacts(settings.runs_dir, thread_id)]

    @app.get("/threads/{thread_id}/artifacts/{rel_path:path}")
    def artifact_download(thread_id: str, rel_path: str):
        return _download_artifact(thread_id, rel_path)

    def _download_artifact(thread_id: str, rel_path: str):
        if rel_path in INTERNAL_FILES:
            raise HTTPException(status_code=404, detail="Not found")
        try:
            ap = artifact_abs_path(settings.runs_dir, thread_id, rel_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        if not ap.exists() or ap.is_dir():
            raise HTTPException(status_code=404, detail="Not found")

        return FileResponse(str(ap))

    return app


app = create_app()
