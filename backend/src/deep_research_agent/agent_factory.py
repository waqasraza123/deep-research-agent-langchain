from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langgraph.checkpoint.sqlite import SqliteSaver

from .model import create_chat_model
from .settings import Settings
from .tools import fetch_document


@dataclass(frozen=True)
class ResearchLimits:
    max_sources: int = 1
    max_links_per_source: int = 0
    follow_links: bool = False


class AgentService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.runs_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.settings.runs_dir / "checkpoints.sqlite"
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._checkpointer = SqliteSaver(conn)

    def _backend(self, rt):
        return CompositeBackend(
            default=StateBackend(rt),
            routes={
                "/runs/": FilesystemBackend(
                    root_dir=str(self.settings.runs_dir),
                    virtual_mode=True,
                ),
            },
        )

    def build_agent(
        self,
        thread_id: str,
        *,
        max_sources: int = 1,
        max_links_per_source: int = 0,
        follow_links: bool = False,
    ):
        max_sources = max(0, min(int(max_sources), 3))
        max_links_per_source = max(0, min(int(max_links_per_source), 10))
        follow_links = bool(follow_links and max_sources > 1 and max_links_per_source > 0)

        limits = ResearchLimits(
            max_sources=max_sources,
            max_links_per_source=max_links_per_source,
            follow_links=follow_links,
        )

        run_dir = f"/runs/{thread_id}"
        thread_dir = (self.settings.runs_dir / thread_id).resolve()
        sources_dir = (thread_dir / "sources").resolve()
        sources_dir.mkdir(parents=True, exist_ok=True)

        seen_urls: set[str] = set()

        def _now_iso_utc() -> str:
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        def _sha1(s: str) -> str:
            return hashlib.sha1(s.encode("utf-8")).hexdigest()

        def fetch_and_store(url: str) -> str:
            """Fetch a URL, store extracted text to runs/<thread_id>/sources, return compact JSON metadata."""
            if url not in seen_urls and len(seen_urls) >= limits.max_sources:
                return json.dumps({"ok": False, "error": "source limit reached", "url": url})

            url_hash = _sha1(url)
            txt_path = sources_dir / f"{url_hash}.txt"
            meta_path = sources_dir / f"{url_hash}.json"

            if meta_path.exists() and txt_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    wc = int(meta.get("word_count") or 0)
                    cc = int(meta.get("char_count") or 0)
                    if meta.get("ok") is True and wc >= 160 and cc >= 1200:
                        seen_urls.add(url)
                        return json.dumps(meta, ensure_ascii=False)
                except Exception:
                    pass

            try:
                fr = fetch_document(
                    url,
                    timeout_s=self.settings.http_timeout_s,
                    max_chars=self.settings.max_page_chars,
                    min_words=160,
                    min_chars=1200,
                )
            except Exception as e:
                return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "url": url})

            txt_path.write_text(fr.extracted_text, encoding="utf-8")

            meta: dict[str, Any] = {
                "ok": fr.ok,
                "url": fr.url,
                "final_url": fr.final_url,
                "title": fr.title,
                "content_type": fr.content_type,
                "status_code": fr.status_code,
                "truncated": fr.truncated,
                "fetched_at": _now_iso_utc(),
                "local_path": f"runs/{thread_id}/sources/{url_hash}.txt",
                "strategy": fr.strategy,
                "word_count": fr.word_count,
                "char_count": fr.char_count,
            }

            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            seen_urls.add(url)
            return json.dumps(meta, ensure_ascii=False)

        system_prompt = f"""You must create these files in {run_dir}:
- plan.md
- notes.md
- sources.json
- report.md

Rules:
- Use only information from fetched sources.
- For each provided URL, call fetch_and_store(url).
- notes.md must include source_id labels S1, S2...
- sources.json must be valid JSON.
- report.md must cite sources like [S1].
- If sources are insufficient, write report.md explaining that and cite [S1].

Output only after all files are written."""

        agent = create_deep_agent(
            model=create_chat_model(self.settings),
            tools=[fetch_and_store],
            system_prompt=system_prompt,
            backend=self._backend,
            checkpointer=self._checkpointer,
        )
        return agent