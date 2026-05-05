from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import ensure_thread_dir, list_artifacts, safe_thread_id

from .contracts import (
    ResearchRun,
    ReviewState,
    RunCancellationRequest,
    RunError,
    RunInputSnapshot,
    RunReviewStatus,
    RunStage,
    RunStatus,
    model_to_dict,
    parse_run,
    utc_now,
)
from .resumability import inspect_resume_point
from .state_machine import STATUS_TO_STAGE, validate_transition

REGISTRY_FILE = ".run.json"
CANCELLATION_FILE = ".cancel.json"


@dataclass(frozen=True)
class RunListFilters:
    status: RunStatus | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    has_errors: bool | None = None
    review_status: ReviewState | None = None


class RunNotFoundError(KeyError):
    pass


class RunCancelledError(RuntimeError):
    pass


class RunRepository:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def registry_path(self, thread_id: str) -> Path:
        safe_thread_id(thread_id)
        return self.runs_dir / thread_id / REGISTRY_FILE

    def cancellation_path(self, thread_id: str) -> Path:
        safe_thread_id(thread_id)
        return self.runs_dir / thread_id / CANCELLATION_FILE

    def create(
        self,
        *,
        thread_id: str,
        question: str,
        urls: list[str],
        settings_snapshot: dict[str, Any],
        require_review: bool = False,
    ) -> ResearchRun:
        ensure_thread_dir(self.runs_dir, thread_id)
        cancel_path = self.cancellation_path(thread_id)
        if cancel_path.exists():
            cancel_path.unlink()
        now = utc_now()
        review = RunReviewStatus(
            status=ReviewState.PENDING if require_review else ReviewState.NOT_REQUIRED,
            created_at=now if require_review else None,
            updated_at=now if require_review else None,
        )
        run = ResearchRun(
            thread_id=thread_id,
            created_at=now,
            updated_at=now,
            question=question,
            urls=urls,
            input_snapshot=RunInputSnapshot(
                question=question,
                urls=urls,
                settings=settings_snapshot,
            ),
            review=review,
        )
        return self.save(run)

    def get(self, thread_id: str) -> ResearchRun:
        path = self.registry_path(thread_id)
        if not path.exists():
            raise RunNotFoundError(thread_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return parse_run(data)

    def get_or_none(self, thread_id: str) -> ResearchRun | None:
        try:
            return self.get(thread_id)
        except RunNotFoundError:
            return None

    def save(self, run: ResearchRun, *, touch: bool = True) -> ResearchRun:
        if touch:
            run.updated_at = utc_now()
        path = self.registry_path(run.thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(model_to_dict(run), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return run

    def transition(self, thread_id: str, status: RunStatus) -> ResearchRun:
        run = self.get(thread_id)
        validate_transition(run.status, status)
        run.status = status
        run.current_stage = STATUS_TO_STAGE[status]
        return self.save(run)

    def refresh_artifacts(self, thread_id: str) -> ResearchRun:
        run = self.get(thread_id)
        artifacts = [artifact.path for artifact in list_artifacts(self.runs_dir, thread_id)]
        run.artifacts = artifacts
        run.output_summary.artifacts = artifacts
        run.output_summary.artifact_count = len(artifacts)
        run.output_summary.report_path = (
            f"runs/{thread_id}/report.md" if "report.md" in artifacts else None
        )
        run.resume_point = inspect_resume_point(self.runs_dir, thread_id, run)
        return self.save(run)

    def set_output_summary(
        self,
        thread_id: str,
        *,
        summary: str = "",
        budget_summary: dict[str, Any] | None = None,
    ) -> ResearchRun:
        run = self.get(thread_id)
        run.output_summary.summary = summary
        if budget_summary is not None:
            run.output_summary.budget_summary = budget_summary
            run.budget_summary = budget_summary
        return self.save(run)

    def set_warnings(self, thread_id: str, warnings: list[str]) -> ResearchRun:
        run = self.get(thread_id)
        run.warnings = warnings
        return self.save(run)

    def record_error(
        self,
        thread_id: str,
        error: BaseException | str,
        *,
        stage: RunStage | None = None,
        details: dict[str, Any] | None = None,
        fail_run: bool = True,
    ) -> ResearchRun:
        run = self.get(thread_id)
        message = str(error)
        error_type = type(error).__name__ if isinstance(error, BaseException) else "RunError"
        run.errors.append(
            RunError(
                type=error_type,
                message=message,
                stage=stage or run.current_stage,
                status=run.status,
                details=details or {},
            )
        )
        if fail_run and run.status not in {
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.COMPLETED,
        }:
            run.status = RunStatus.FAILED
            run.current_stage = RunStage.FAILED
        return self.save(run)

    def request_cancellation(
        self,
        thread_id: str,
        request: RunCancellationRequest,
    ) -> ResearchRun:
        run = self.get(thread_id)
        run.cancellation = request
        path = self.cancellation_path(thread_id)
        path.write_text(
            json.dumps(model_to_dict(request), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if run.status == RunStatus.CREATED:
            run.status = RunStatus.CANCELLED
            run.current_stage = RunStage.CANCELLED
        return self.save(run)

    def cancellation_requested(self, thread_id: str) -> bool:
        return self.cancellation_path(thread_id).exists()

    def raise_if_cancelled(self, thread_id: str) -> None:
        if self.cancellation_requested(thread_id):
            run = self.get(thread_id)
            if run.status not in {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED}:
                run.status = RunStatus.CANCELLED
                run.current_stage = RunStage.CANCELLED
                self.save(run)
            raise RunCancelledError(f"Run {thread_id} has a cancellation marker")

    def update_review(self, thread_id: str, review: RunReviewStatus) -> ResearchRun:
        run = self.get(thread_id)
        review.updated_at = utc_now()
        if review.created_at is None:
            review.created_at = review.updated_at
        run.review = review
        return self.save(run)

    def list(self, filters: RunListFilters | None = None) -> list[ResearchRun]:
        filters = filters or RunListFilters()
        runs: list[ResearchRun] = []
        for path in sorted(self.runs_dir.glob(f"*/{REGISTRY_FILE}")):
            try:
                run = parse_run(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if filters.status and run.status != filters.status:
                continue
            if filters.created_after and run.created_at < filters.created_after:
                continue
            if filters.created_before and run.created_at > filters.created_before:
                continue
            if filters.has_errors is not None and bool(run.errors) != filters.has_errors:
                continue
            if filters.review_status and run.review.status != filters.review_status:
                continue
            runs.append(run)
        runs.sort(key=lambda item: item.created_at, reverse=True)
        return runs

    def delete_run_dir(self, thread_id: str) -> None:
        safe_thread_id(thread_id)
        run_dir = self.runs_dir / thread_id
        if run_dir.exists():
            shutil.rmtree(run_dir)


def settings_snapshot_from_object(settings: Any) -> dict[str, Any]:
    keys = (
        "model_provider",
        "temperature",
        "ollama_model",
        "ollama_num_predict",
        "openai_base_url",
        "openai_model",
        "openai_max_tokens",
        "openai_timeout_s",
        "openai_max_retries",
        "max_page_chars",
        "http_timeout_s",
    )
    snapshot: dict[str, Any] = {}
    for key in keys:
        if hasattr(settings, key):
            value = getattr(settings, key)
            snapshot[key] = str(value) if isinstance(value, Path) else value
    return snapshot
