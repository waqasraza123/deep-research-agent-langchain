from __future__ import annotations

import time
from pathlib import Path

from .artifact_writer import RuntimeArtifactWriter
from .contracts import ResearchJob, RuntimeBudgetUsage
from .errors import RuntimeBudgetExceeded
from .repository import RuntimeRepository


class RuntimeBudgetTracker:
    def __init__(self, *, repository: RuntimeRepository, runs_dir: Path, job: ResearchJob):
        self.repository = repository
        self.runs_dir = runs_dir
        self.job = job
        self.started = time.monotonic()

    def refresh(self) -> ResearchJob:
        self.job = self.repository.get_job(self.job.job_id)
        return self.job

    def update_runtime(self) -> None:
        self.job.budget_usage.runtime_seconds = max(0.0, time.monotonic() - self.started)

    def update_artifact_bytes(self) -> None:
        writer = RuntimeArtifactWriter(runs_dir=self.runs_dir, thread_id=self.job.thread_id)
        self.job.budget_usage.artifact_bytes = writer.artifact_bytes()

    def increment_events(self, count: int = 1) -> None:
        self.job.budget_usage.events += max(0, count)

    def increment_retries(self, count: int = 1) -> None:
        self.job.budget_usage.retries += max(0, count)

    def increment_sources(self, count: int) -> None:
        self.job.budget_usage.total_sources += max(0, count)
        self.job.budget_usage.source_fetches += max(0, count)

    def increment_model_calls(self, count: int = 1) -> None:
        self.job.budget_usage.model_calls += max(0, count)

    def increment_extracted_chars(self, count: int) -> None:
        self.job.budget_usage.total_extracted_chars += max(0, count)

    def stage_runtime(self, stage: str, seconds: float) -> None:
        self.job.budget_usage.stage_runtime_seconds[stage] = (
            self.job.budget_usage.stage_runtime_seconds.get(stage, 0.0) + max(0.0, seconds)
        )

    def exceeded_reasons(self) -> list[str]:
        self.update_runtime()
        self.update_artifact_bytes()
        usage = self.job.budget_usage
        budget = self.job.budget
        checks = [
            (
                usage.runtime_seconds > budget.max_runtime_seconds,
                f"runtime_seconds {usage.runtime_seconds:.2f} > {budget.max_runtime_seconds}",
            ),
            (
                any(v > budget.max_stage_seconds for v in usage.stage_runtime_seconds.values()),
                f"stage_runtime_seconds exceeded {budget.max_stage_seconds}",
            ),
            (
                usage.source_fetches > budget.max_source_fetches,
                f"source_fetches {usage.source_fetches} > {budget.max_source_fetches}",
            ),
            (
                usage.model_calls > budget.max_model_calls,
                f"model_calls {usage.model_calls} > {budget.max_model_calls}",
            ),
            (
                usage.artifact_bytes > budget.max_artifact_bytes,
                f"artifact_bytes {usage.artifact_bytes} > {budget.max_artifact_bytes}",
            ),
            (
                usage.retries > budget.max_retries,
                f"retries {usage.retries} > {budget.max_retries}",
            ),
            (
                usage.events > budget.max_events,
                f"events {usage.events} > {budget.max_events}",
            ),
            (
                usage.total_sources > budget.max_total_sources,
                f"total_sources {usage.total_sources} > {budget.max_total_sources}",
            ),
            (
                usage.total_extracted_chars > budget.max_total_extracted_chars,
                "total_extracted_chars "
                f"{usage.total_extracted_chars} > {budget.max_total_extracted_chars}",
            ),
        ]
        return [reason for exceeded, reason in checks if exceeded]

    def check(self) -> list[str]:
        reasons = self.exceeded_reasons()
        if reasons:
            self.job.budget_usage.exceeded = True
            for reason in reasons:
                if reason not in self.job.budget_usage.exceeded_reasons:
                    self.job.budget_usage.exceeded_reasons.append(reason)
            self.persist()
            if self.job.budget.fail_on_budget_exceeded:
                raise RuntimeBudgetExceeded(reasons)
        self.persist()
        return reasons

    def persist(self) -> None:
        self.repository.update_job(self.job)
        RuntimeArtifactWriter(runs_dir=self.runs_dir, thread_id=self.job.thread_id).write_budget(
            self.job.budget, self.job.budget_usage
        )

    def snapshot(self) -> RuntimeBudgetUsage:
        self.update_runtime()
        return self.job.budget_usage.copy(deep=True)

