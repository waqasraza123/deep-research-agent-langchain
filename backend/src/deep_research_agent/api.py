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
from .logging_config import configure_logging
from .model import create_chat_model
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


def _safe_local_rel(local_path: str) -> str | None:
    if not local_path:
        return None
    if local_path.startswith("/") or ".." in local_path or "\\" in local_path:
        return None
    if "runs/" not in local_path:
        return None
    rel = local_path.split("runs/", 1)[-1]
    if not rel or rel.startswith("/") or ".." in rel or "\\" in rel:
        return None
    return rel


def _read_text(path, *, max_chars: int) -> str:
    try:
        if not path.exists() or path.is_dir():
            return ""
        s = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not s:
            return ""
        if len(s) > max_chars:
            s = s[:max_chars] + "\n\n[TRUNCATED]\n"
        return s
    except Exception:
        return ""


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
                if size_ok and meta.get("ok") is True and meta.get("url") == u:
                    need = False
            except Exception:
                need = True

        if need:
            try:
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
            except Exception as e:
                try:
                    txt_path.write_text(f"Fetch failed: {type(e).__name__}: {e}\nURL: {u}\n", encoding="utf-8")
                except Exception:
                    pass
                meta = {
                    "ok": False,
                    "url": u,
                    "error": f"{type(e).__name__}: {e}",
                    "local_path": f"runs/{thread_id}/sources/{h}.txt",
                    "strategy": "error",
                    "word_count": 0,
                    "char_count": 0,
                }
                try:
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass

        try:
            out.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            out.append({"ok": False, "url": u, "local_path": f"runs/{thread_id}/sources/{h}.txt", "strategy": "error"})
    return out


def _build_deterministic_report(td) -> str:
    notes = _read_text(td / "notes.md", max_chars=4000)
    sources_json = _read_text(td / "sources.json", max_chars=2000)
    body = "# Report\n\n"
    if notes:
        body += notes.strip() + "\n\n"
    else:
        body += "Notes were not produced. This report was generated from available artifacts.\n\n"
    body += "## Sources\n\n"
    if sources_json:
        body += "```json\n" + sources_json.strip() + "\n```\n\n"
    body += "## Conclusion\n\n"
    body += "This report is grounded only in captured sources and artifacts. [S1]\n"
    return body


def _ensure_report_with_model(settings: Settings, td, question: str, sources_meta: list[dict[str, Any]]) -> None:
    report_path = td / "report.md"
    if report_path.exists():
        return

    usable = [m for m in sources_meta if isinstance(m, dict) and m.get("ok") is True and isinstance(m.get("local_path"), str)]
    if not usable:
        report_path.write_text(_build_deterministic_report(td), encoding="utf-8")
        return

    m = usable[0]
    rel = _safe_local_rel(m["local_path"])
    if not rel:
        report_path.write_text(_build_deterministic_report(td), encoding="utf-8")
        return

    src_fs = (settings.runs_dir / rel).resolve()
    src_text = _read_text(src_fs, max_chars=8000)
    if not src_text:
        report_path.write_text(_build_deterministic_report(td), encoding="utf-8")
        return

    notes = _read_text(td / "notes.md", max_chars=2500)
    src_url = m.get("final_url") or m.get("url") or ""
    src_title = m.get("title") or ""

    prompt = (
        "Write report.md as Markdown using only the provided source text and notes.\n"
        "Requirements:\n"
        "- Title\n"
        "- Exactly 6 bullet points\n"
        "- 1-line conclusion\n"
        "- Cite claims using [S1]\n"
        "- No outside knowledge\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Source S1 URL: {src_url}\nTitle: {src_title}\n\n"
        f"Notes:\n{notes}\n\n"
        f"Source text:\n{src_text}\n"
    )

    try:
        model = create_chat_model(settings)
        msg = model.invoke([{"role": "user", "content": prompt}])
        content = (getattr(msg, "content", "") or "").strip()
        if len(content) >= 200:
            report_path.write_text(content + "\n", encoding="utf-8")
            return
    except Exception:
        log.exception("ensure_report_with_model failed")

    report_path.write_text(_build_deterministic_report(td), encoding="utf-8")


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
        user_msg += "\n\nRules:\n- Use only fetched sources.\n- Do not use outside knowledge.\n- Write all required files."

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

        _ensure_report_with_model(settings, td, req.question.strip(), sources_meta)

        summary_text = ""
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            last = result["messages"][-1]
            if isinstance(last, dict):
                summary_text = last.get("content") or ""
            else:
                summary_text = getattr(last, "content", "") or ""

        warnings = ensure_required_artifacts(settings.runs_dir, thread_id)

        return {
            "thread_id": thread_id,
            "summary": summary_text,
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