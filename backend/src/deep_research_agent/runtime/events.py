from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import now_iso_utc

from .contracts import RunEvent, RunEventType


class RunEventLogger:
    def __init__(self, *, thread_id: str, thread_dir: Path):
        self.thread_id = thread_id
        self.thread_dir = thread_dir
        self.jsonl_path = thread_dir / "events.jsonl"
        self.markdown_path = thread_dir / "events.md"
        thread_dir.mkdir(parents=True, exist_ok=True)
        if not self.markdown_path.exists():
            self.markdown_path.write_text("# Run Events\n\n", encoding="utf-8")

    def log(
        self,
        event_type: RunEventType,
        *,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RunEvent:
        event = RunEvent(
            event_type=event_type,
            thread_id=self.thread_id,
            timestamp=now_iso_utc(),
            message=message,
            metadata=metadata or {},
        )
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.dict(), sort_keys=True) + "\n")
        with self.markdown_path.open("a", encoding="utf-8") as f:
            detail = f" `{json.dumps(event.metadata, sort_keys=True)}`" if event.metadata else ""
            suffix = f" - {event.message}" if event.message else ""
            f.write(f"- `{event.timestamp}` **{event.event_type}**{suffix}{detail}\n")
        return event

    def read_events(self) -> list[RunEvent]:
        if not self.jsonl_path.exists():
            return []
        events: list[RunEvent] = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(RunEvent.parse_raw(line))
        return events


def read_event_file(thread_dir: Path) -> list[dict[str, Any]]:
    logger = RunEventLogger(thread_id=thread_dir.name, thread_dir=thread_dir)
    return [event.dict() for event in logger.read_events()]
