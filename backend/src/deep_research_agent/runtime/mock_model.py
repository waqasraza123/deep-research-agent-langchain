from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class _Message:
    def __init__(self, content: str):
        self.content = content


class DeterministicMockChatModel:
    provider = "mock"
    model_name = "deterministic-mock-research-model"

    def invoke(self, messages: list[dict[str, str]] | Any, *_args, **_kwargs):
        content = _extract_prompt(messages)
        question = _first_nonempty_line(content)
        return _Message(_mock_response(question))


def _extract_prompt(messages: list[dict[str, str]] | Any) -> str:
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "")
        return str(getattr(last, "content", "") or "")
    return str(messages or "")


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s[:160]
    return "Mock research request"


def _mock_response(question: str) -> str:
    return (
        "[MOCK OUTPUT]\n"
        f"Question: {question}\n\n"
        "Plan:\n"
        "1. Restate the request.\n"
        "2. Inspect provided sources only.\n"
        "3. Produce a cautious report skeleton.\n\n"
        "Limitations: deterministic mock model; no external reasoning or factual validation.\n"
    )


def write_mock_research_artifacts(
    *,
    thread_dir: Path,
    thread_id: str,
    question: str,
    sources_meta: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    thread_dir.mkdir(parents=True, exist_ok=True)
    sources_meta = sources_meta or []
    normalized_sources = []
    for idx, item in enumerate(sources_meta, start=1):
        normalized_sources.append(
            {
                "source_id": f"S{idx}",
                "url": item.get("final_url") or item.get("url") or "mock://source",
                "title": item.get("title") or "Mock source",
                "ok": bool(item.get("ok", False)),
                "summary": "Mock summary placeholder derived from captured source metadata.",
                "mock": True,
            }
        )
    if not normalized_sources:
        normalized_sources.append(
            {
                "source_id": "S1",
                "url": "mock://no-source-provided",
                "title": "No source provided",
                "ok": False,
                "summary": "No external source was fetched in mock mode.",
                "mock": True,
            }
        )

    metadata = {
        "mock": True,
        "model_provider": "mock",
        "model_name": DeterministicMockChatModel.model_name,
        "thread_id": thread_id,
    }

    (thread_dir / "plan.md").write_text(
        "# Plan\n\n"
        "> MOCK OUTPUT: deterministic offline plan.\n\n"
        f"- Restate the question: {question.strip()}\n"
        "- Review only supplied source metadata and stored source text if present.\n"
        "- Draft notes, source summaries, limitations, and a final report skeleton.\n",
        encoding="utf-8",
    )
    (thread_dir / "notes.md").write_text(
        "# Notes\n\n"
        "> MOCK OUTPUT: deterministic offline notes.\n\n"
        "- S1: Placeholder note. The mock model does not verify facts.\n"
        "- Use this run to validate orchestration, artifacts, budgets, and event logs.\n",
        encoding="utf-8",
    )
    (thread_dir / "sources.json").write_text(
        json.dumps(normalized_sources, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (thread_dir / "report.md").write_text(
        "# Mock Research Report\n\n"
        "> MOCK OUTPUT: this report is generated without real model credentials and must not be "
        "treated as factual research.\n\n"
        "## Summary\n\n"
        f"- Request: {question.strip()}\n"
        "- This is a deterministic report skeleton for smoke tests and CI.\n"
        "- Source handling, artifact writing, event logging, and budget persistence "
        "were exercised.\n\n"
        "## Source Summaries\n\n"
        "- [S1] Mock source summary placeholder.\n\n"
        "## Limitations\n\n"
        "- No live model inference was performed.\n"
        "- No outside knowledge was used.\n"
        "- Claims are placeholders unless backed by separately inspected source artifacts.\n",
        encoding="utf-8",
    )
    (thread_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


class MockResearchAgent:
    def __init__(self, *, thread_dir: Path, thread_id: str, question_hint: str = ""):
        self.thread_dir = thread_dir
        self.thread_id = thread_id
        self.question_hint = question_hint

    def invoke(self, payload: dict[str, Any], *_args, **_kwargs) -> dict[str, Any]:
        question = self.question_hint or "Mock research request"
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                question = str(last.get("content") or question).splitlines()[0][:200]
        write_mock_research_artifacts(
            thread_dir=self.thread_dir,
            thread_id=self.thread_id,
            question=question,
            sources_meta=[],
        )
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"[MOCK OUTPUT] Done. Report at runs/{self.thread_id}/report.md",
                }
            ]
        }
