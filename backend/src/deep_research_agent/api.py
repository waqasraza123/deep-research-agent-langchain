from __future__ import annotations

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
from .evidence import rebuild_evidence_artifacts
from .intelligence import ResearchStrategy, create_research_strategy
from .logging_config import configure_logging
from .model import create_chat_model
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
from .source_intelligence import CrawlBudgetStats, CrawlResult, SourceRecord, crawl_sources
from .source_intelligence.source_graph import write_source_graph_artifacts, write_sources_manifest

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


class ReviewActionRequest(BaseModel):
    reviewer: str = Field(..., min_length=1)
    notes: str = ""
    requested_changes: list[str] = Field(default_factory=list)


class CleanupApplyRequest(BaseModel):
    thread_ids: list[str]
    confirm_delete: bool = False


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


def create_app(*, settings: Settings | None = None, service: AgentService | None = None) -> FastAPI:
    configure_logging()
    settings = settings or Settings.load()
    service = service or AgentService(settings)
    run_repository = RunRepository(settings.runs_dir)

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
                _write_mock_source_artifacts(thread_dir=td, thread_id=thread_id, urls=urls)
                for rel_path in (
                    "plan.md",
                    "notes.md",
                    "sources.json",
                    "source_graph.json",
                    "source_graph.md",
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
                run_context.budget_warning()
                run_context.log("run_completed", message="mock run completed", metadata=metadata)
                run_repository.refresh_artifacts(thread_id)
                run_repository.set_output_summary(
                    thread_id,
                    budget_summary=read_budget_file(td / "budget.json"),
                )
                lifecycle.complete(
                    require_review=require_review,
                    summary="[MOCK OUTPUT] Deterministic offline run completed.",
                )
                return {
                    "thread_id": thread_id,
                    "summary": "[MOCK OUTPUT] Deterministic offline run completed.",
                    "strategy": strategy.to_json_dict() if strategy else None,
                    "warnings": [],
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
            agent_source_urls = [source.url for source in crawl_result.usable_sources()]
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
            user_msg += (
                "\n\nRules:\n"
                "- Use only fetched sources.\n"
                "- Do not use outside knowledge.\n"
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
                run_repository.set_warnings(
                    thread_id,
                    ["Explicit mock fallback used after model failure."],
                )
                run_repository.refresh_artifacts(thread_id)
                _advance_to_building_evidence(run_repository, lifecycle, thread_id)
                run_repository.set_output_summary(
                    thread_id,
                    budget_summary=read_budget_file(td / "budget.json"),
                )
                lifecycle.complete(
                    require_review=require_review,
                    summary=("[MOCK OUTPUT] Explicit mock fallback completed after model failure."),
                )
                fallback_summary = (
                    "[MOCK OUTPUT] Explicit mock fallback completed after model failure."
                )
                return {
                    "thread_id": thread_id,
                    "summary": fallback_summary,
                    "strategy": strategy.to_json_dict() if strategy else None,
                    "warnings": ["Explicit mock fallback used after model failure."],
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
