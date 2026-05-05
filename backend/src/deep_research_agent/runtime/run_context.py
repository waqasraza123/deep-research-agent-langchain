from __future__ import annotations

from pathlib import Path
from typing import Any

from .budgets import BudgetExceeded, BudgetTracker
from .contracts import RunBudget
from .events import RunEventLogger


class RunContext:
    def __init__(self, *, thread_id: str, thread_dir: Path, budget: RunBudget):
        self.thread_id = thread_id
        self.thread_dir = thread_dir
        self.events = RunEventLogger(thread_id=thread_id, thread_dir=thread_dir)
        self.budget = BudgetTracker(budget)
        self.budget_path = thread_dir / "budget.json"
        self.persist_budget()

    def persist_budget(self) -> None:
        self.budget.write(self.budget_path)

    def log(
        self,
        event_type: str,
        *,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.events.log(event_type, message=message, metadata=metadata or {})  # type: ignore[arg-type]

    def model_call_started(self, *, provider: str, model_name: str, purpose: str) -> None:
        self.budget.increment_model_calls()
        self.persist_budget()
        self.log(
            "model_call_started",
            message=purpose,
            metadata={"provider": provider, "model_name": model_name},
        )

    def model_call_completed(
        self,
        *,
        generated_chars: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.budget.increment_generated_chars(generated_chars)
        self.persist_budget()
        self.log(
            "model_call_completed",
            message="model call completed",
            metadata={"generated_chars": generated_chars, **(metadata or {})},
        )

    def source_fetch_started(self, url: str) -> None:
        self.budget.increment_source_fetches()
        self.persist_budget()
        self.log("source_fetch_started", message=url, metadata={"url": url})

    def source_fetch_completed(self, url: str, metadata: dict[str, Any]) -> None:
        self.log("source_fetch_completed", message=url, metadata=metadata)
        self.persist_budget()

    def source_fetch_failed(self, url: str, error: str) -> None:
        self.log("source_fetch_failed", message=url, metadata={"url": url, "error": error})
        self.persist_budget()

    def artifact_written(self, path: str, size_bytes: int | None = None) -> None:
        self.log(
            "artifact_written",
            message=path,
            metadata={"path": path, "size_bytes": size_bytes},
        )
        self.budget.update_artifacts_size(self.thread_dir)
        self.persist_budget()

    def budget_warning(self) -> None:
        warnings = self.budget.warning_reasons()
        new_warnings = [w for w in warnings if w not in self.budget.usage.warning_reasons]
        for warning in new_warnings:
            self.budget.usage.warning_reasons.append(warning)
            self.log("budget_warning", message=warning)
        self.persist_budget()

    def budget_exceeded(self, exc: BudgetExceeded) -> None:
        self.log("budget_exceeded", message=str(exc), metadata={"reasons": exc.reasons})
        self.persist_budget()
