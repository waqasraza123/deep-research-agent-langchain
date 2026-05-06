from __future__ import annotations

import json
import time
import uuid
from dataclasses import replace
from hashlib import sha1
from typing import Any

from deep_research_agent.agent_factory import AgentService
from deep_research_agent.artifacts import ensure_required_artifacts, list_artifacts
from deep_research_agent.runtime.contracts import RunBudget
from deep_research_agent.runtime.mock_model import write_mock_research_artifacts
from deep_research_agent.runtime.run_context import RunContext
from deep_research_agent.settings import Settings
from deep_research_agent.tools import fetch_document

from .artifact_writer import RuntimeArtifactWriter
from .budgets import RuntimeBudgetTracker
from .contracts import (
    ORDERED_STAGES,
    ResearchJob,
    ResearchJobStatus,
    ResearchStage,
    RuntimeErrorRecord,
    RuntimeEventType,
)
from .events import RuntimeEventWriter
from .repository import RuntimeRepository


class StageExecutionCancelled(RuntimeError):
    pass


class StageExecutionPaused(RuntimeError):
    pass


class RuntimeStageExecutor:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: RuntimeRepository,
        service: AgentService | None = None,
    ):
        self.settings = settings
        self.repository = repository
        self.service = service
        self.events = RuntimeEventWriter(repository=repository, runs_dir=settings.runs_dir)

    def next_stage(self, job: ResearchJob) -> ResearchStage | None:
        records = self.repository.get_stage_records(job.job_id)
        completed = {r.stage for r in records if r.status in {"completed", "skipped"}}
        for stage in ORDERED_STAGES:
            if stage not in completed:
                return stage
        return None

    def execute_next_stage(self, job: ResearchJob) -> bool:
        stage = self.next_stage(job)
        if stage is None:
            return False
        self.execute_stage(job, stage)
        return True

    def execute_stage(self, job: ResearchJob, stage: ResearchStage) -> None:
        self._check_control(job)
        record = self.repository.mark_stage_started(job.job_id, stage)
        job = self.repository.get_job(job.job_id)
        job.stage = stage
        self.repository.update_job(job)
        self.events.emit(job, RuntimeEventType.STAGE_STARTED, stage=stage, message=stage.value)
        started = time.monotonic()
        writer = RuntimeArtifactWriter(runs_dir=self.settings.runs_dir, thread_id=job.thread_id)
        budget = RuntimeBudgetTracker(
            repository=self.repository, runs_dir=self.settings.runs_dir, job=job
        )
        outputs: list[str] = []
        metrics: dict[str, Any] = {}
        try:
            if stage == ResearchStage.INPUT_SNAPSHOT:
                outputs = self._input_snapshot(job, writer)
            elif stage == ResearchStage.PLANNING:
                outputs = self._planning(job, writer)
            elif stage == ResearchStage.SOURCE_FETCHING:
                outputs, metrics = self._source_fetching(job, writer)
                budget.increment_sources(int(metrics.get("source_count") or 0))
            elif stage == ResearchStage.SOURCE_PROCESSING:
                outputs = self._source_processing(job, writer)
            elif stage == ResearchStage.AGENT_EXECUTION:
                outputs = self._agent_execution(job, writer)
                budget.increment_model_calls(1)
            elif stage == ResearchStage.ARTIFACT_BACKFILL:
                outputs = self._artifact_backfill(job, writer)
            elif stage == ResearchStage.INTELLIGENCE_POSTPROCESSING:
                outputs = self._postprocessing(job, writer)
            elif stage == ResearchStage.VERIFICATION:
                outputs = self._verification(job, writer)
            elif stage == ResearchStage.FINALIZATION:
                outputs = self._finalization(job, writer)
            elapsed = time.monotonic() - started
            budget.stage_runtime(stage.value, elapsed)
            budget.check()
            record = self.repository.mark_stage_completed(
                record.stage_id,
                output_artifacts=outputs,
                metrics={**metrics, "runtime_seconds": elapsed},
            )
            self.events.emit(
                job,
                RuntimeEventType.STAGE_COMPLETED,
                stage=stage,
                message=stage.value,
                artifact_refs=outputs,
            )
            writer.write_stages(self.repository.get_stage_records(job.job_id))
            writer.write_job(self.repository.get_job(job.job_id))
            if stage == ResearchStage.FINALIZATION:
                self.repository.mark_completed(job.job_id)
                job = self.repository.get_job(job.job_id)
                self.events.emit(job, RuntimeEventType.JOB_COMPLETED, message="job completed")
                writer.write_job(job)
        except (StageExecutionCancelled, StageExecutionPaused):
            raise
        except Exception as exc:
            error = RuntimeErrorRecord(
                error_id=str(uuid.uuid4()),
                error_type=type(exc).__name__,
                message=str(exc),
                stage=stage,
                retryable=type(exc).__name__ not in {"ValueError", "PermissionError"},
            )
            self.repository.mark_stage_failed(record.stage_id, error)
            self.repository.set_error(job.job_id, error)
            writer.write_json("runtime_error.json", error)
            self.events.emit(
                job,
                RuntimeEventType.STAGE_FAILED,
                stage=stage,
                severity="error",
                message=f"{type(exc).__name__}: {exc}",
            )
            raise

    def _check_control(self, job: ResearchJob) -> None:
        fresh = self.repository.get_job(job.job_id)
        if fresh.status in {ResearchJobStatus.CANCELLING, ResearchJobStatus.CANCELLED}:
            self.repository.mark_job_status(
                fresh.job_id, ResearchJobStatus.CANCELLED, validate=False
            )
            self.events.emit(
                fresh,
                RuntimeEventType.JOB_CANCELLED,
                message="cancelled before stage",
            )
            raise StageExecutionCancelled("Job cancelled")
        if fresh.status in {ResearchJobStatus.PAUSING, ResearchJobStatus.PAUSED}:
            self.repository.mark_job_status(fresh.job_id, ResearchJobStatus.PAUSED, validate=False)
            self.events.emit(fresh, RuntimeEventType.JOB_PAUSED, message="paused before stage")
            raise StageExecutionPaused("Job paused")

    def _input_snapshot(self, job: ResearchJob, writer: RuntimeArtifactWriter) -> list[str]:
        writer.write_input_snapshot(job=job)
        writer.write_job(job)
        return ["runtime_input_snapshot.json", "runtime_job.json"]

    def _planning(self, job: ResearchJob, writer: RuntimeArtifactWriter) -> list[str]:
        plan = [
            "# Runtime Plan",
            "",
            f"Question: {job.question}",
            "",
            "1. Snapshot inputs and settings.",
            "2. Collect and process configured sources.",
            "3. Execute the research agent or configured mock runtime.",
            "4. Backfill required artifacts.",
            "5. Verify and finalize runtime metadata.",
        ]
        writer.write_text("runtime_plan.md", "\n".join(plan) + "\n")
        plan_path = writer.thread_dir / "plan.md"
        if not plan_path.exists():
            plan_path.write_text("# Plan\n\n- Runtime generated initial plan.\n", encoding="utf-8")
        return ["runtime_plan.md", "plan.md"]

    def _source_fetching(
        self, job: ResearchJob, writer: RuntimeArtifactWriter
    ) -> tuple[list[str], dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        source_outputs: list[str] = []
        for idx, url in enumerate(job.urls, start=1):
            if bool(job.metadata.get("mock_agent_execution")):
                sources.append(
                    {
                        "source_id": f"S{idx}",
                        "url": url,
                        "ok": False,
                        "skipped": True,
                        "skip_reason": "runtime_mock_agent_execution",
                        "title": "Runtime mock source placeholder",
                        "mock": True,
                    }
                )
                continue
            digest = sha1(url.encode("utf-8")).hexdigest()
            try:
                fetched = fetch_document(
                    url,
                    timeout_s=self.settings.http_timeout_s,
                    max_chars=self.settings.max_page_chars,
                    min_words=1,
                    min_chars=1,
                )
                text_rel = f"sources/{digest}.txt"
                meta_rel = f"sources/{digest}.json"
                writer.write_text(text_rel, fetched.extracted_text)
                meta = {
                    "source_id": f"S{idx}",
                    "url": fetched.url,
                    "final_url": fetched.final_url,
                    "title": fetched.title,
                    "content_type": fetched.content_type,
                    "status_code": fetched.status_code,
                    "ok": fetched.ok,
                    "skipped": not fetched.ok,
                    "truncated": fetched.truncated,
                    "strategy": fetched.strategy,
                    "word_count": fetched.word_count,
                    "char_count": fetched.char_count,
                    "canonical_url": fetched.canonical_url,
                    "document_kind": fetched.kind,
                    "local_path": f"runs/{job.thread_id}/{text_rel}",
                }
                writer.write_json(meta_rel, meta)
                sources.append(meta)
                source_outputs.extend([text_rel, meta_rel])
            except Exception as exc:
                sources.append(
                    {
                        "source_id": f"S{idx}",
                        "url": url,
                        "ok": False,
                        "skipped": True,
                        "skip_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        writer.write_json(
            "runtime_source_fetch_summary.json",
            {
                "source_count": len(sources),
                "urls": job.urls,
                "fetched_count": len([source for source in sources if source.get("ok") is True]),
                "mock": bool(job.metadata.get("mock_agent_execution")),
            },
        )
        if not (writer.thread_dir / "sources.json").exists():
            writer.write_json("sources.json", sources)
        return (
            ["runtime_source_fetch_summary.json", "sources.json", *source_outputs],
            {"source_count": len(sources)},
        )

    def _source_processing(self, job: ResearchJob, writer: RuntimeArtifactWriter) -> list[str]:
        writer.write_text(
            "runtime_source_processing_summary.md",
            "# Runtime Source Processing\n\n"
            "No optional runtime-specific source processing subsystem was required.\n",
        )
        return ["runtime_source_processing_summary.md"]

    def _agent_execution(self, job: ResearchJob, writer: RuntimeArtifactWriter) -> list[str]:
        use_mock = bool(
            job.metadata.get("mock_agent_execution")
            or getattr(self.settings, "runtime_mock_agent_execution_enabled", False)
        )
        if use_mock:
            sources_meta = []
            try:
                raw = json.loads((writer.thread_dir / "sources.json").read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    sources_meta = [item for item in raw if isinstance(item, dict)]
            except Exception:
                sources_meta = []
            write_mock_research_artifacts(
                thread_dir=writer.thread_dir,
                thread_id=job.thread_id,
                question=job.question,
                sources_meta=sources_meta,
            )
            self.repository.append_warning(job.job_id, "Runtime mock agent execution used.")
            job = self.repository.get_job(job.job_id)
            self.events.emit(
                job,
                RuntimeEventType.ARTIFACT_WRITTEN,
                severity="warning",
                message="runtime mock agent execution used",
                artifact_refs=["plan.md", "notes.md", "sources.json", "report.md"],
            )
        else:
            effective_settings = self.settings
            if self.service is None and getattr(self.settings, "model_provider", "") == "mock":
                effective_settings = replace(self.settings, model_provider="mock")
            service = self.service or AgentService(effective_settings)
            context = RunContext(
                thread_id=job.thread_id,
                thread_dir=writer.thread_dir,
                budget=RunBudget(
                    max_model_calls=max(1, job.budget.max_model_calls),
                    max_source_fetches=max(0, job.budget.max_source_fetches),
                    max_runtime_seconds=max(1, float(job.budget.max_runtime_seconds)),
                    max_artifacts_size=max(1, job.budget.max_artifact_bytes),
                ),
            )
            agent = service.build_agent(
                job.thread_id,
                max_sources=min(len(job.urls), job.budget.max_total_sources),
                run_context=context,
            )
            prompt = (
                job.question
                + "\n\nUse the configured source list and write plan.md, notes.md, "
                "sources.json, and report.md under the run directory."
            )
            agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": job.thread_id}},
            )
        return ["plan.md", "notes.md", "sources.json", "report.md"]

    def _artifact_backfill(self, job: ResearchJob, writer: RuntimeArtifactWriter) -> list[str]:
        warnings = ensure_required_artifacts(self.settings.runs_dir, job.thread_id)
        for warning in warnings:
            self.repository.append_warning(job.job_id, warning)
        return ["plan.md", "notes.md", "sources.json", "report.md"]

    def _postprocessing(self, job: ResearchJob, writer: RuntimeArtifactWriter) -> list[str]:
        if not getattr(self.settings, "runtime_run_postprocessing", True):
            writer.write_text(
                "runtime_postprocessing_summary.md",
                "# Runtime Postprocessing\n\nPostprocessing disabled by settings.\n",
            )
            return ["runtime_postprocessing_summary.md"]
        writer.write_text(
            "runtime_postprocessing_summary.md",
            "# Runtime Postprocessing\n\nNo additional runtime postprocessing was required.\n",
        )
        return ["runtime_postprocessing_summary.md"]

    def _verification(self, job: ResearchJob, writer: RuntimeArtifactWriter) -> list[str]:
        writer.write_text(
            "runtime_verification_skipped.md",
            "# Runtime Verification\n\n"
            "Runtime verification was skipped; existing verification rebuild APIs "
            "remain available.\n",
        )
        return ["runtime_verification_skipped.md"]

    def _finalization(self, job: ResearchJob, writer: RuntimeArtifactWriter) -> list[str]:
        warnings = ensure_required_artifacts(self.settings.runs_dir, job.thread_id)
        for warning in warnings:
            self.repository.append_warning(job.job_id, warning)
        artifacts = [
            item.__dict__ for item in list_artifacts(self.settings.runs_dir, job.thread_id)
        ]
        job = self.repository.get_job(job.job_id)
        job.artifact_summary = {
            "artifact_count": len(artifacts),
            "required": ["plan.md", "notes.md", "sources.json", "report.md"],
            "artifacts": artifacts,
        }
        self.repository.update_job(job)
        writer.write_json("runtime_final_summary.json", job.artifact_summary)
        writer.write_text(
            "runtime_final_summary.md",
            "# Runtime Final Summary\n\n"
            f"- Job: `{job.job_id}`\n"
            f"- Thread: `{job.thread_id}`\n"
            f"- Artifacts: {len(artifacts)}\n"
            f"- Warnings: {len(job.warnings)}\n",
        )
        writer.write_stages(self.repository.get_stage_records(job.job_id))
        writer.write_job(job)
        return [
            "runtime_final_summary.json",
            "runtime_final_summary.md",
            "runtime_stages.json",
            "runtime_stages.md",
            "runtime_job.json",
        ]
