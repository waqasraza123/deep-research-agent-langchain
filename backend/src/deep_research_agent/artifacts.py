from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from .intelligence import ResearchStrategy


@dataclass
class Artifact:
    path: str
    size_bytes: int
    mtime_epoch: float


REQUIRED_FILES = ("plan.md", "notes.md", "sources.json", "report.md")
INTERNAL_FILES = {".run.json", ".cancel.json"}


def safe_thread_id(thread_id: str) -> str:
    if not thread_id or "/" in thread_id or "\\" in thread_id or ".." in thread_id:
        raise ValueError("Invalid thread_id")
    return thread_id


def ensure_thread_dir(runs_dir: Path, thread_id: str) -> Path:
    safe_thread_id(thread_id)
    runs_dir.mkdir(parents=True, exist_ok=True)
    td = (runs_dir / thread_id).resolve()
    td.mkdir(parents=True, exist_ok=True)
    return td


def list_artifacts(runs_dir: Path, thread_id: str) -> List[Artifact]:
    td = ensure_thread_dir(runs_dir, thread_id)
    out: List[Artifact] = []
    for p in td.rglob("*"):
        if p.is_dir():
            continue
        rel = str(p.relative_to(td)).replace(os.sep, "/")
        if rel in INTERNAL_FILES:
            continue
        out.append(
            Artifact(
                path=rel,
                size_bytes=p.stat().st_size,
                mtime_epoch=p.stat().st_mtime,
            )
        )
    out.sort(key=lambda a: a.path)
    return out


def artifact_abs_path(runs_dir: Path, thread_id: str, rel_path: str) -> Path:
    td = ensure_thread_dir(runs_dir, thread_id)
    if rel_path.startswith("/") or ".." in rel_path or "\\" in rel_path:
        raise ValueError("Invalid path")
    ap = (td / rel_path).resolve()
    if not str(ap).startswith(str(td)):
        raise ValueError("Invalid path")
    return ap


def ensure_required_artifacts(runs_dir: Path, thread_id: str) -> list[str]:
    """
    Production-grade: guarantee deliverables exist.
    Returns warnings if we had to backfill.
    """
    td = ensure_thread_dir(runs_dir, thread_id)
    warnings: list[str] = []

    plan = td / "plan.md"
    notes = td / "notes.md"
    sources = td / "sources.json"
    report = td / "report.md"

    if not plan.exists():
        plan.write_text("# Plan\n\n- (Agent did not write plan)\n", encoding="utf-8")
        warnings.append("Backfilled plan.md (agent did not create it).")

    if not notes.exists():
        notes.write_text("# Notes\n\n(Agent did not write notes)\n", encoding="utf-8")
        warnings.append("Backfilled notes.md (agent did not create it).")

    if not sources.exists():
        sources.write_text("[]\n", encoding="utf-8")
        warnings.append("Backfilled sources.json (agent did not create it).")
    else:
        # validate JSON so consumers don’t break
        try:
            json.loads(sources.read_text(encoding="utf-8"))
        except Exception:
            sources.write_text("[]\n", encoding="utf-8")
            warnings.append("Reset sources.json to [] (invalid JSON).")

    if not report.exists():
        report.write_text(
            "# Report\n\n(Agent did not write report)\n",
            encoding="utf-8",
        )
        warnings.append("Backfilled report.md (agent did not create it).")

    return warnings


def _strategy_subquestions_payload(strategy: "ResearchStrategy") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sq in strategy.subquestions:
        if hasattr(sq, "model_dump"):
            out.append(sq.model_dump(mode="json"))  # type: ignore[attr-defined]
        else:
            out.append(sq.dict())
    return out


def _verification_plan_markdown(strategy: "ResearchStrategy") -> str:
    lines = [
        "# Verification Plan",
        "",
        f"Research ID: `{strategy.research_id}`",
        "",
    ]
    for idx, step in enumerate(strategy.verification_plan, start=1):
        cats = ", ".join(c.value for c in step.required_sources) or "source-appropriate"
        lines.extend(
            [
                f"## {idx}. {step.claim_area}",
                "",
                f"- Priority: P{step.priority}",
                f"- Method: {step.method}",
                f"- Required source categories: {cats}",
                "",
            ]
        )
    return "\n".join(lines)


def write_strategy_artifacts(
    runs_dir: Path,
    thread_id: str,
    strategy: "ResearchStrategy",
) -> list[Artifact]:
    """
    Persist deterministic planning artifacts under runs/<thread_id>/.
    The thread id and all generated paths go through the same safety checks as downloads.
    """
    td = ensure_thread_dir(runs_dir, thread_id)

    files = {
        "strategy.json": strategy.to_json(),
        "strategy.md": strategy.to_markdown(),
        "subquestions.json": json.dumps(
            _strategy_subquestions_payload(strategy),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        "verification_plan.md": _verification_plan_markdown(strategy),
    }
    for rel_path, content in files.items():
        path = artifact_abs_path(runs_dir, thread_id, rel_path)
        path.write_text(content, encoding="utf-8")

    return [
        Artifact(
            path=rel_path,
            size_bytes=(td / rel_path).stat().st_size,
            mtime_epoch=(td / rel_path).stat().st_mtime,
        )
        for rel_path in sorted(files)
    ]


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
