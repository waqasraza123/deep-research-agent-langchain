from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ..tools import _word_count, html_to_text
from .case_loader import build_source_url_map
from .contracts import BenchmarkCase, BenchmarkSource
from .errors import FixtureCorpusError, UnsafeBenchmarkPathError


@dataclass(frozen=True)
class FixtureDocument:
    case_id: str
    source: BenchmarkSource
    text: str
    local_path: Path

    @property
    def metadata(self) -> dict:
        return {
            "title": self.source.title,
            "source_type": self.source.source_type,
            "content_hash": self.source.content_hash,
            "local_path": str(self.local_path),
            "published_at": self.source.published_at,
            "updated_at": self.source.updated_at,
            "trust_level": self.source.trust_level,
            "benchmark_case_id": self.case_id,
            "benchmark_source_id": self.source.source_id,
            "warnings": list(self.source.warnings),
        }


class FixtureCorpus:
    def __init__(self, cases: list[BenchmarkCase]):
        self.cases = {case.case_id: case for case in cases}
        self._url_map: dict[str, tuple[BenchmarkCase, BenchmarkSource]] = {}
        for case in cases:
            for url, source in build_source_url_map(case).items():
                self._url_map[url] = (case, source)

    def resolve(self, url: str) -> FixtureDocument:
        self._validate_benchmark_url(url)
        item = self._url_map.get(url)
        if item is None:
            raise FixtureCorpusError(f"Unknown benchmark source URL: {url}")
        case, source = item
        if not case.case_dir:
            raise FixtureCorpusError(f"Case {case.case_id} has no case_dir")
        case_dir = Path(case.case_dir).resolve()
        local_path = (case_dir / source.local_path).resolve()
        if case_dir not in local_path.parents and local_path != case_dir:
            raise UnsafeBenchmarkPathError(
                f"Fixture path escapes case directory: {source.local_path}"
            )
        if not local_path.exists() or local_path.is_dir():
            raise FixtureCorpusError(f"Fixture source not found: {source.local_path}")
        raw = local_path.read_text(encoding="utf-8", errors="ignore")
        if source.source_type.lower() == "html" or local_path.suffix.lower() in {".html", ".htm"}:
            text = html_to_text(raw)
        else:
            text = raw.strip()
        return FixtureDocument(
            case_id=case.case_id, source=source, text=text, local_path=local_path
        )

    def metadata_for_url(self, url: str) -> dict:
        return self.resolve(url).metadata

    @staticmethod
    def _validate_benchmark_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "benchmark":
            raise FixtureCorpusError("FixtureCorpus only resolves benchmark:// URLs")
        if not parsed.netloc or not parsed.path.strip("/"):
            raise FixtureCorpusError("Benchmark URL must be benchmark://case_id/source_id")
        if ".." in parsed.netloc or ".." in parsed.path or "\\" in url:
            raise UnsafeBenchmarkPathError(f"Unsafe benchmark URL: {url}")
        if "/" in parsed.netloc or parsed.path.count("/") != 1:
            raise UnsafeBenchmarkPathError(f"Unsafe benchmark URL: {url}")

    @staticmethod
    def word_count(text: str) -> int:
        return _word_count(text)
