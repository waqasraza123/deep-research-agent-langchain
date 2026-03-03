from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent_factory import AgentService
from .artifacts import artifact_abs_path, ensure_required_artifacts, ensure_thread_dir, list_artifacts
from .model import create_chat_model
from .logging_config import configure_logging
from .settings import Settings
from .tools import fetch_document


log = logging.getLogger("deep_research_agent.api")


class RunRequest(BaseModel):
    question: str = Field(..., min_length=5)
    urls: list[str] = []
    thread_id: str | None = None

    max_sources: int = Field(default=1, ge=0, le=3)
    max_links_per_source: int = Field(default=0, ge=0, le=10)
    follow_links: bool = False


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _prefetch_sources(settings: Settings, td, thread_id: str, urls: list[str]) -> list[dict[str, Any]]:
    sources_dir = (td / "sources").resolve()
    sources_dir.mkdir(parents=True, exist_ok=True)

    out: list[dict[str, Any]] = []
    for u in urls:
        h = _sha1(u)
        txt_path = sources_dir / f"{h}.txt"
        meta_path = sources_dir / f"{h}.json"

        need = True
        if txt_path.exists() and meta_path.exists():
            try:
                size_ok = int(txt_path.stat().st_size) >= 1200
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if size_ok and meta.get("ok") is True:
                    need = False
            except Exception:
                need = True

        if need:
            fr = fetch_document(
                u,
                timeout_s=settings.http_timeout_s,
                max_chars=settings.max_page_chars,
                min_words=160,
                min_chars=1200,
            )
            txt_path.write_text(fr.extracted_text, encoding="utf-8")
            meta = {
                "ok": True,
                "url": fr.url,
                "final_url": fr.final_url,
                "title": fr.title,
                "content_type": fr.content_type,
                "status_code": fr.status_code,
                "truncated": fr.truncated,
                "local_path": f"runs/{thread_id}/sources/{h}.txt",
                "strategy": fr.strategy,
                "word_count": fr.word_count,
                "char_count": fr.char_count,
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        try:
            out.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            out.append({"ok": False, "url": u, "local_path": f"runs/{thread_id}/sources/{h}.txt"})
    return out


def _extract_json_object(text: str) -> dict[str, Any] | None:
    s = text.strip()
    if not s:
        return None
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except Exception:
        return None


def _fill_missing_with_model(settings: Settings, td, question: str, sources_meta: list[dict[str, Any]]) -> None:
    plan_path = td / "plan.md"
    notes_path = td / "notes.md"
    sources_path = td / "sources.json"
    report_path = td / "report.md"

    missing = [p for p in (plan_path, notes_path, sources_path, report_path) if not p.exists()]
    if not missing:
        return

    sources_text_blocks: list[str] = []
    for i, m in enumerate(sources_meta, start=1):
        lp = m.get("local_path")
        if not isinstance(lp, str):
            continue
        local_rel = lp.split("runs/", 1)[-1]
        local_fs = (settings.runs_dir / local_rel).resolve()
        if not local_fs.exists():
            continue
        text = local_fs.read_text(encoding="utf-8", errors="ignore")
        text = text.strip()
        if not text:
            continue
        if len(text) > 12000:
            text = text[:12000] + "\n\n[TRUNCATED]\n"
        sources_text_blocks.append(f"S{i} URL: {m.get('final_url') or m.get('url')}\n{text}")

    context = "\n\n".join(sources_text_blocks).strip()
    if not context:
        return

    prompt = (
        "Using only the source text below, produce strict JSON with keys:\n"
        "plan_md (string), notes_md (string), sources_json (array), report_md (string).\n"
        "sources_json items must include: source_id, url, title, local_path.\n"
        "report_md must cite like [S1]. Do not use outside knowledge.\n\n"
        f"Question:\n{question}\n\n"
        f"Sources:\n{context}\n"
    )

    model = create_chat_model(settings)
    msg = model.invoke([{"role": "user", "content": prompt}])
    content = getattr(msg, "content", "") or ""
    data = _extract_json_object(content)
    if not isinstance(data, dict):
        return

    if not plan_path.exists() and isinstance(data.get("plan_md"), str) and data["plan_md"].strip():
        plan_path.write_text(data["plan_md"].strip() + "\n", encoding="utf-8")

    if not notes_path.exists() and isinstance(data.get("notes_md"), str) and data["notes_md"].strip():
        notes_path.write_text(data["notes_md"].strip() + "\n", encoding="utf-8")

    if not sources_path.exists() and isinstance(data.get("sources_json"), list):
        sources_path.write_text(json.dumps(data["sources_json"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not report_path.exists() and isinstance(data.get("report_md"), str) and data["report_md"].strip():
        report_path.write_text(data["report_md"].strip() + "\n", encoding="utf-8")


def create_app(*, settings: Settings | None = None, service: AgentService | None = None) -> FastAPI:
    configure_logging()
    settings = settings or Settings.load()
    service = service or AgentService(settings)

    app = FastAPI(title="Deep Research Agent")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.post("/run")
    def run(req: RunRequest) -> dict[str, Any]:
        thread_id = req.thread_id or str(uuid.uuid4())
        td = ensure_thread_dir(settings.runs_dir, thread_id)

        urls = [u.strip() for u in req.urls if u and u.strip()]
        urls = urls[: max(0, min(req.max_sources, 3))] if urls else []

        sources_meta = _prefetch_sources(settings, td, thread_id, urls)

        user_msg = req.question.strip()
        if urls:
            user_msg += "\n\nSources (call fetch_and_store on these):\n" + "\n".join(f"- {u}" for u in urls)

        agent = service.build_agent(
            thread_id,
            max_sources=max(0, min(req.max_sources, 3)),
            max_links_per_source=max(0, min(req.max_links_per_source, 10)),
            follow_links=bool(req.follow_links),
        )

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_msg}]},
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as e:
            log.exception("agent.invoke failed")
            warnings = ensure_required_artifacts(settings.runs_dir, thread_id)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": f"{type(e).__name__}: {e}",
                    "thread_id": thread_id,
                    "warnings": warnings,
                },
            )

        _fill_missing_with_model(settings, td, req.question.strip(), sources_meta)

        summary_text = None
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            last = result["messages"][-1]
            if isinstance(last, dict):
                summary_text = last.get("content")
            else:
                summary_text = getattr(last, "content", None)

        warnings = ensure_required_artifacts(settings.runs_dir, thread_id)

        return {
            "thread_id": thread_id,
            "summary": summary_text or "",
            "warnings": warnings,
            "artifacts": [a.__dict__ for a in list_artifacts(settings.runs_dir, thread_id)],
            "hint": f"Report should be at runs/{thread_id}/report.md",
        }

    @app.get("/threads/{thread_id}/artifacts")
    def artifacts(thread_id: str) -> list[dict[str, Any]]:
        return [a.__dict__ for a in list_artifacts(settings.runs_dir, thread_id)]

    @app.get("/threads/{thread_id}/artifacts/{rel_path:path}")
    def artifact_download(thread_id: str, rel_path: str):
        try:
            ap = artifact_abs_path(settings.runs_dir, thread_id, rel_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not ap.exists() or ap.is_dir():
            raise HTTPException(status_code=404, detail="Not found")

        return FileResponse(str(ap))

    return app


app = create_app()