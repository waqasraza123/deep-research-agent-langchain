from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import artifact_abs_path, ensure_thread_dir

from .contracts import ResearchJob, ResearchStageRecord, RuntimeBudget, RuntimeBudgetUsage


def _jsonable(value: Any) -> Any:
    if hasattr(value, "dict"):
        return value.dict()
    return value


class RuntimeArtifactWriter:
    def __init__(self, *, runs_dir: Path, thread_id: str):
        self.runs_dir = runs_dir
        self.thread_id = thread_id
        self.thread_dir = ensure_thread_dir(runs_dir, thread_id)

    def write_text(self, rel_path: str, content: str) -> str:
        path = artifact_abs_path(self.runs_dir, self.thread_id, rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return rel_path

    def write_json(self, rel_path: str, payload: Any) -> str:
        data = _jsonable(payload)
        return self.write_text(
            rel_path,
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        )

    def append_jsonl(self, rel_path: str, payload: Any) -> str:
        path = artifact_abs_path(self.runs_dir, self.thread_id, rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, default=str)
                + "\n"
            )
        return rel_path

    def write_job(self, job: ResearchJob) -> None:
        self.write_json("runtime_job.json", job)

    def write_input_snapshot(
        self,
        *,
        job: ResearchJob,
        request: dict[str, Any] | None = None,
    ) -> None:
        self.write_json(
            "runtime_input_snapshot.json",
            {
                "job_id": job.job_id,
                "thread_id": job.thread_id,
                "question": job.question,
                "urls": job.urls,
                "settings_snapshot": job.settings_snapshot,
                "request": request or {},
            },
        )

    def write_stages(self, stages: list[ResearchStageRecord]) -> None:
        self.write_json("runtime_stages.json", [stage.dict() for stage in stages])
        lines = ["# Runtime Stages", ""]
        for stage in stages:
            lines.append(
                f"- `{stage.stage}`: **{stage.status}** "
                f"(attempts={stage.attempts}, required={stage.required})"
            )
        self.write_text("runtime_stages.md", "\n".join(lines).rstrip() + "\n")

    def write_budget(self, budget: RuntimeBudget, usage: RuntimeBudgetUsage) -> None:
        self.write_json("runtime_budget.json", {"budget": budget, "usage": usage})
        reasons = ", ".join(usage.exceeded_reasons) if usage.exceeded_reasons else "none"
        self.write_text(
            "runtime_budget.md",
            "\n".join(
                [
                    "# Runtime Budget",
                    "",
                    f"- Runtime seconds: {usage.runtime_seconds:.2f}/{budget.max_runtime_seconds}",
                    f"- Source fetches: {usage.source_fetches}/{budget.max_source_fetches}",
                    f"- Model calls: {usage.model_calls}/{budget.max_model_calls}",
                    f"- Artifact bytes: {usage.artifact_bytes}/{budget.max_artifact_bytes}",
                    f"- Events: {usage.events}/{budget.max_events}",
                    f"- Retries: {usage.retries}/{budget.max_retries}",
                    f"- Exceeded: {usage.exceeded}",
                    f"- Reasons: {reasons}",
                ]
            )
            + "\n",
        )

    def artifact_bytes(self) -> int:
        total = 0
        for path in self.thread_dir.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total
