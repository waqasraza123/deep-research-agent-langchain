from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .artifact_writer import RuntimeArtifactWriter
from .contracts import ResearchJob, ResearchStage, RuntimeEvent, RuntimeEventType
from .repository import RuntimeRepository


class RuntimeEventWriter:
    def __init__(self, *, repository: RuntimeRepository, runs_dir: Path):
        self.repository = repository
        self.runs_dir = runs_dir

    def emit(
        self,
        job: ResearchJob,
        event_type: RuntimeEventType,
        *,
        stage: ResearchStage | None = None,
        severity: str = "info",
        message: str = "",
        data: dict[str, Any] | None = None,
        artifact_refs: list[str] | None = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            event_id=str(uuid.uuid4()),
            job_id=job.job_id,
            thread_id=job.thread_id,
            event_type=event_type,
            stage=stage,
            severity=severity,
            message=message,
            data=data or {},
            artifact_refs=artifact_refs or [],
        )
        self.repository.append_event(event)
        writer = RuntimeArtifactWriter(runs_dir=self.runs_dir, thread_id=job.thread_id)
        writer.append_jsonl("runtime_events.jsonl", event)
        self.render_markdown(job)
        return event

    def render_markdown(self, job: ResearchJob) -> None:
        events = self.repository.list_events(job.job_id)
        lines = ["# Runtime Events", ""]
        for event in events:
            suffix = f" - {event.message}" if event.message else ""
            stage = f" `{event.stage}`" if event.stage else ""
            detail = ""
            if event.data:
                detail = f" `{json.dumps(event.data, sort_keys=True, default=str)}`"
            lines.append(
                f"- `{event.timestamp.isoformat()}` **{event.event_type}**{stage}{suffix}{detail}"
            )
        RuntimeArtifactWriter(runs_dir=self.runs_dir, thread_id=job.thread_id).write_text(
            "runtime_events.md", "\n".join(lines).rstrip() + "\n"
        )

