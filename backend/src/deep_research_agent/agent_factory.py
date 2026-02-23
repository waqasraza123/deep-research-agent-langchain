from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langgraph.checkpoint.sqlite import SqliteSaver

from .model import create_chat_model
from .settings import Settings
from .tools import extract_links, extract_title, html_to_text


@dataclass(frozen=True)
class ResearchLimits:
    max_sources: int = 8
    max_links_per_source: int = 12
    follow_links: bool = True


class AgentService:
    """
    Shared resources + agent construction.

    Production considerations:
    - Limits enforced to prevent runaway tool usage
    - Fetch results cached to disk per thread
    - Content-type checked, size capped, and stored as text
    """

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
        max_sources: int = 8,
        max_links_per_source: int = 12,
        follow_links: bool = True,
    ):
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
            """
            Tool: fetch a URL, store to runs/<thread_id>/sources/<sha>.txt and <sha>.json,
            and return compact metadata JSON.
            """
            if not (url.startswith("http://") or url.startswith("https://")):
                return json.dumps({"error": "Blocked: only http(s) URLs are allowed.", "url": url})

            # enforce limits on unique sources
            if url not in seen_urls and len(seen_urls) >= limits.max_sources:
                return json.dumps(
                    {"error": "Source limit reached", "max_sources": limits.max_sources, "url": url}
                )

            url_hash = _sha1(url)
            txt_path = sources_dir / f"{url_hash}.txt"
            meta_path = sources_dir / f"{url_hash}.json"

            # cache hit
            if meta_path.exists() and txt_path.exists():
                seen_urls.add(url)
                try:
                    return meta_path.read_text(encoding="utf-8")
                except Exception:
                    pass

            try:
                with httpx.Client(timeout=self.settings.http_timeout_s, follow_redirects=True) as client:
                    r = client.get(url, headers={"User-Agent": "deep-research-agent/0.1"})
                    status = r.status_code
                    final_url = str(r.url)
                    ctype = (r.headers.get("content-type") or "").lower()
                    raw = r.text or ""
            except Exception as e:
                return json.dumps({"error": f"Fetch failed: {type(e).__name__}: {e}", "url": url})

            # hard cap size
            truncated = False
            if len(raw) > self.settings.max_page_chars:
                raw = raw[: self.settings.max_page_chars] + "\n\n[TRUNCATED]\n"
                truncated = True

            title = extract_title(raw) if "html" in ctype else None

            links: list[str] = []
            text: str = raw

            if "html" in ctype:
                text = html_to_text(raw)
                if limits.follow_links:
                    links = extract_links(raw, final_url, limit=limits.max_links_per_source)

            if not ctype.startswith("text/") and "html" not in ctype:
                # store a readable placeholder instead of binary
                text = f"Non-text content-type: {ctype}\nURL: {final_url}\nStatus: {status}\n"

            txt_path.write_text(text, encoding="utf-8")

            meta: dict[str, Any] = {
                "url": url,
                "final_url": final_url,
                "status_code": status,
                "content_type": ctype,
                "fetched_at": _now_iso_utc(),
                "title": title,
                "local_text_path": f"runs/{thread_id}/sources/{url_hash}.txt",
                "truncated": truncated,
                "links": links,
                "preview": text[:50],
            }

            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            seen_urls.add(url)
            return json.dumps(meta, ensure_ascii=False, indent=2)

        system_prompt = f"""\
You are an expert research analyst.

You MUST write working artifacts to: {run_dir}

Deliverables (always):
1) {run_dir}/plan.md
2) {run_dir}/notes.md
3) {run_dir}/sources.json as a JSON array of objects:
   {{
     "source_id": "S1",
     "url": "...",
     "title": "...",
     "local_path": "runs/{thread_id}/sources/<hash>.txt",
     "summary": "...",
     "key_points": ["..."],
     "quotes": ["..."]
   }}
4) {run_dir}/report.md a polished Markdown report with citations like [S1], [S2]

Limits:
- Max sources: {limits.max_sources}
- Max links per source: {limits.max_links_per_source}
- Follow links: {str(limits.follow_links).lower()}

Workflow:
A) Write plan.md first. Include what you will read and why.
B) For each provided URL:
   - call fetch_and_store(url)
   - read its local_text_path if needed
C) If follow_links is true:
   - pick only the most relevant links returned by fetch_and_store
   - fetch_and_store a few until max_sources is reached
D) Write notes.md with per-source sections:
   - title, url, 5-10 bullet key facts, 1-3 short quotes
E) Write sources.json using the schema above.
F) Write report.md:
   - clear structure, conclusion, explicit assumptions if any
   - cite every important claim using [Sx]

Return in chat:
- 5-10 line executive summary and confirm report.md location.
"""

        agent = create_deep_agent(
            model=create_chat_model(self.settings),
            tools=[fetch_and_store],
            system_prompt=system_prompt,
            backend=self._backend,
            checkpointer=self._checkpointer,
        )
        return agent