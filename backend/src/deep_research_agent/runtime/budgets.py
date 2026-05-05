from __future__ import annotations

import json
import time
from pathlib import Path

from .contracts import RunBudget, RunBudgetUsage


class BudgetExceeded(RuntimeError):
    def __init__(self, reasons: list[str]):
        super().__init__("Budget exceeded: " + "; ".join(reasons))
        self.reasons = reasons


class BudgetTracker:
    def __init__(self, budget: RunBudget, *, started_at: float | None = None):
        self.budget = budget
        self.started_at = started_at or time.monotonic()
        self.usage = RunBudgetUsage()

    def increment_model_calls(self, count: int = 1) -> None:
        self.usage.model_calls += max(0, count)
        self.check()

    def increment_source_fetches(self, count: int = 1) -> None:
        self.usage.source_fetches += max(0, count)
        self.check()

    def increment_generated_chars(self, count: int) -> None:
        self.usage.generated_chars += max(0, count)
        self.check()

    def increment_crawl_expansion(self, count: int = 1) -> None:
        self.usage.crawl_expansion += max(0, count)
        self.check()

    def update_runtime(self) -> None:
        self.usage.runtime_seconds = max(0.0, time.monotonic() - self.started_at)

    def update_artifacts_size(self, thread_dir: Path) -> None:
        size = 0
        if thread_dir.exists():
            for path in thread_dir.rglob("*"):
                if path.is_file():
                    try:
                        size += path.stat().st_size
                    except OSError:
                        pass
        self.usage.artifacts_size = size
        self.check()

    def current_reasons(self) -> list[str]:
        self.update_runtime()
        checks = [
            (
                self.usage.model_calls > self.budget.max_model_calls,
                f"model_calls {self.usage.model_calls} > {self.budget.max_model_calls}",
            ),
            (
                self.usage.source_fetches > self.budget.max_source_fetches,
                f"source_fetches {self.usage.source_fetches} > {self.budget.max_source_fetches}",
            ),
            (
                self.usage.generated_chars > self.budget.max_generated_chars,
                f"generated_chars {self.usage.generated_chars} > {self.budget.max_generated_chars}",
            ),
            (
                self.usage.runtime_seconds > self.budget.max_runtime_seconds,
                "runtime_seconds "
                f"{self.usage.runtime_seconds:.2f} > {self.budget.max_runtime_seconds:.2f}",
            ),
            (
                self.usage.artifacts_size > self.budget.max_artifacts_size,
                f"artifacts_size {self.usage.artifacts_size} > {self.budget.max_artifacts_size}",
            ),
            (
                self.usage.crawl_expansion > self.budget.max_crawl_expansion,
                f"crawl_expansion {self.usage.crawl_expansion} > {self.budget.max_crawl_expansion}",
            ),
        ]
        return [reason for exceeded, reason in checks if exceeded]

    def warning_reasons(self) -> list[str]:
        self.update_runtime()
        warnings: list[str] = []
        pairs = [
            ("model_calls", self.usage.model_calls, self.budget.max_model_calls),
            ("source_fetches", self.usage.source_fetches, self.budget.max_source_fetches),
            ("generated_chars", self.usage.generated_chars, self.budget.max_generated_chars),
            ("runtime_seconds", self.usage.runtime_seconds, self.budget.max_runtime_seconds),
            ("artifacts_size", self.usage.artifacts_size, self.budget.max_artifacts_size),
            ("crawl_expansion", self.usage.crawl_expansion, self.budget.max_crawl_expansion),
        ]
        for name, used, max_allowed in pairs:
            if max_allowed > 0 and used >= max_allowed * 0.8 and used <= max_allowed:
                warnings.append(f"{name} is at {used}/{max_allowed}")
        return warnings

    def check(self) -> None:
        reasons = self.current_reasons()
        if reasons:
            for reason in reasons:
                if reason not in self.usage.exceeded_reasons:
                    self.usage.exceeded_reasons.append(reason)
            raise BudgetExceeded(reasons)

    def snapshot(self) -> RunBudgetUsage:
        self.update_runtime()
        return self.usage.copy(deep=True)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"budget": self.budget.dict(), "usage": self.snapshot().dict()}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_budget_file(path: Path) -> dict:
    if not path.exists() or path.is_dir():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
