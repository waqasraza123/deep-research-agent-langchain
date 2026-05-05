from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .contracts import ArtifactReference, ExtractedEntity, ExtractedTopic, MemoryRecord


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_question(question: str) -> str:
    text = re.sub(r"[^a-z0-9\s]+", " ", (question or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def source_domain(url: str | None) -> str | None:
    if not url:
        return None
    host = urlsplit(url).hostname
    return host.lower() if host else None


def _model_dump_jsonable(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


class MemoryRepository:
    def __init__(self, data_dir: Path, *, max_records: int = 20_000) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "memory.sqlite"
        self.max_records = max(100, int(max_records))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    normalized_url TEXT NOT NULL,
                    canonical_url TEXT,
                    source_title TEXT,
                    source_domain TEXT,
                    content_hash TEXT NOT NULL,
                    extracted_text_hash TEXT NOT NULL,
                    source_type TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    run_count INTEGER NOT NULL DEFAULT 1,
                    quality_score REAL,
                    entities_json TEXT NOT NULL DEFAULT '[]',
                    topics_json TEXT NOT NULL DEFAULT '[]',
                    summary TEXT NOT NULL DEFAULT '',
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    artifacts_json TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_normalized_url "
                "ON memories(normalized_url)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_canonical_url "
                "ON memories(canonical_url)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_content_hash "
                "ON memories(content_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_source_domain "
                "ON memories(source_domain)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_normalized_question "
                "ON memories(normalized_question)"
            )

    def upsert(self, record: MemoryRecord) -> MemoryRecord:
        now = now_iso_utc()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (record.memory_id,)
            ).fetchone()
            if existing:
                previous = self._row_to_record(existing)
                record = record.copy(
                    update={
                        "first_seen_at": previous.first_seen_at,
                        "last_seen_at": now,
                        "run_count": max(previous.run_count + 1, record.run_count),
                    }
                )
            conn.execute(
                """
                INSERT INTO memories (
                    memory_id, thread_id, question, normalized_question, source_url,
                    normalized_url, canonical_url, source_title, source_domain, content_hash,
                    extracted_text_hash, source_type, first_seen_at, last_seen_at, run_count,
                    quality_score, entities_json, topics_json, summary, warnings_json,
                    artifacts_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    thread_id=excluded.thread_id,
                    question=excluded.question,
                    normalized_question=excluded.normalized_question,
                    source_url=excluded.source_url,
                    normalized_url=excluded.normalized_url,
                    canonical_url=excluded.canonical_url,
                    source_title=excluded.source_title,
                    source_domain=excluded.source_domain,
                    content_hash=excluded.content_hash,
                    extracted_text_hash=excluded.extracted_text_hash,
                    source_type=excluded.source_type,
                    last_seen_at=excluded.last_seen_at,
                    run_count=excluded.run_count,
                    quality_score=excluded.quality_score,
                    entities_json=excluded.entities_json,
                    topics_json=excluded.topics_json,
                    summary=excluded.summary,
                    warnings_json=excluded.warnings_json,
                    artifacts_json=excluded.artifacts_json
                """,
                self._record_to_tuple(record),
            )
            self._prune(conn)
        return record

    def list_records(self, *, limit: int = 500) -> list[MemoryRecord]:
        limit = max(1, min(int(limit), self.max_records))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY last_seen_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def search(self, query: str, *, limit: int = 20) -> list[MemoryRecord]:
        normalized = normalize_question(query)
        tokens = [t for t in normalized.split() if len(t) >= 3]
        if not tokens:
            return []
        records = self.list_records(limit=min(self.max_records, 2000))
        scored: list[tuple[float, MemoryRecord]] = []
        query_set = set(tokens)
        for record in records:
            haystack = " ".join(
                [
                    record.normalized_question,
                    record.source_title or "",
                    record.summary,
                    " ".join(topic.name for topic in record.topics),
                    " ".join(entity.name for entity in record.entities),
                ]
            ).lower()
            hit_count = sum(1 for token in query_set if token in haystack)
            if hit_count == 0:
                continue
            question_tokens = set(record.normalized_question.split())
            overlap = len(query_set & question_tokens) / max(1, len(query_set | question_tokens))
            score = min(1.0, (hit_count / max(1, len(query_set))) * 0.6 + overlap * 0.4)
            scored.append((score, record))
        scored.sort(key=lambda item: (item[0], item[1].last_seen_at), reverse=True)
        return [record for _, record in scored[: max(1, int(limit))]]

    def find_by_exact_url(self, url: str, *, limit: int = 20) -> list[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE source_url = ? ORDER BY last_seen_at DESC LIMIT ?",
                (url, max(1, int(limit))),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_by_normalized_url(self, normalized_url: str, *, limit: int = 20) -> list[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE normalized_url = ? "
                "ORDER BY last_seen_at DESC LIMIT ?",
                (normalized_url, max(1, int(limit))),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_by_canonical_url(self, canonical_url: str, *, limit: int = 20) -> list[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE canonical_url = ? "
                "ORDER BY last_seen_at DESC LIMIT ?",
                (canonical_url, max(1, int(limit))),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_by_content_hash(self, digest: str, *, limit: int = 20) -> list[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE content_hash = ? OR extracted_text_hash = ? "
                "ORDER BY last_seen_at DESC LIMIT ?",
                (digest, digest, max(1, int(limit))),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_by_domain_title(
        self, domain: str, title_terms: list[str], *, limit: int = 20
    ) -> list[MemoryRecord]:
        if not domain or not title_terms:
            return []
        records = self.list_sources(domain=domain, limit=200)
        terms = {t.lower() for t in title_terms if len(t) >= 3}
        scored: list[tuple[int, MemoryRecord]] = []
        for record in records:
            title = (record.source_title or "").lower()
            hits = sum(1 for term in terms if term in title)
            if hits:
                scored.append((hits, record))
        scored.sort(key=lambda item: (item[0], item[1].last_seen_at), reverse=True)
        return [record for _, record in scored[: max(1, int(limit))]]

    def list_sources(self, *, domain: str | None = None, limit: int = 100) -> list[MemoryRecord]:
        with self._connect() as conn:
            if domain:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE source_domain = ? "
                    "ORDER BY last_seen_at DESC LIMIT ?",
                    (domain.lower(), max(1, int(limit))),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY last_seen_at DESC LIMIT ?",
                    (max(1, int(limit)),),
                ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_entity(self, entity_name: str, *, limit: int = 100) -> list[MemoryRecord]:
        needle = entity_name.strip().lower()
        if not needle:
            return []
        matches = []
        for record in self.list_records(limit=min(self.max_records, 2000)):
            if any(entity.name.lower() == needle for entity in record.entities):
                matches.append(record)
            if len(matches) >= limit:
                break
        return matches

    def records_for_thread(self, thread_id: str) -> list[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE thread_id = ? ORDER BY source_url", (thread_id,)
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def _prune(self, conn: sqlite3.Connection) -> None:
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        excess = int(count) - self.max_records
        if excess <= 0:
            return
        conn.execute(
            """
            DELETE FROM memories
            WHERE memory_id IN (
                SELECT memory_id FROM memories ORDER BY last_seen_at ASC LIMIT ?
            )
            """,
            (excess,),
        )

    def _record_to_tuple(self, record: MemoryRecord) -> tuple[Any, ...]:
        return (
            record.memory_id,
            record.thread_id,
            record.question,
            record.normalized_question,
            record.source_url,
            record.normalized_url,
            record.canonical_url,
            record.source_title,
            record.source_domain,
            record.content_hash,
            record.extracted_text_hash,
            record.source_type,
            record.first_seen_at,
            record.last_seen_at,
            record.run_count,
            record.quality_score,
            _json_dumps([_model_dump_jsonable(entity) for entity in record.entities]),
            _json_dumps([_model_dump_jsonable(topic) for topic in record.topics]),
            record.summary,
            _json_dumps(record.warnings),
            _json_dumps([_model_dump_jsonable(artifact) for artifact in record.artifacts]),
        )

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        entities = [
            ExtractedEntity(**item)
            for item in _json_loads(row["entities_json"], [])
            if isinstance(item, dict)
        ]
        topics = [
            ExtractedTopic(**item)
            for item in _json_loads(row["topics_json"], [])
            if isinstance(item, dict)
        ]
        artifacts = [
            ArtifactReference(**item)
            for item in _json_loads(row["artifacts_json"], [])
            if isinstance(item, dict)
        ]
        return MemoryRecord(
            memory_id=row["memory_id"],
            thread_id=row["thread_id"],
            question=row["question"],
            normalized_question=row["normalized_question"],
            source_url=row["source_url"],
            normalized_url=row["normalized_url"],
            canonical_url=row["canonical_url"],
            source_title=row["source_title"],
            source_domain=row["source_domain"],
            content_hash=row["content_hash"],
            extracted_text_hash=row["extracted_text_hash"],
            source_type=row["source_type"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            run_count=int(row["run_count"]),
            quality_score=row["quality_score"],
            entities=entities,
            topics=topics,
            summary=row["summary"] or "",
            warnings=_json_loads(row["warnings_json"], []),
            artifacts=artifacts,
        )
