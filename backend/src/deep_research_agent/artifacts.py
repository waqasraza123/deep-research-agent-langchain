from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List
import os
import time


@dataclass
class Artifact:
    path: str
    size_bytes: int
    mtime_epoch: float


REQUIRED_FILES = ("plan.md", "notes.md", "sources.json", "report.md")


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
        out.append(
            Artifact(
                path=str(p.relative_to(td)).replace(os.sep, "/"),
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


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())