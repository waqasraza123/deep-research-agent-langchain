from __future__ import annotations

from ..tools import FetchResult, _word_count
from .fixture_corpus import FixtureCorpus


class OfflineFetcher:
    def __init__(self, corpus: FixtureCorpus, *, allow_benchmark_scheme: bool = True):
        self.corpus = corpus
        self.allow_benchmark_scheme = allow_benchmark_scheme

    def fetch_document(self, url: str, *, max_chars: int = 250_000) -> FetchResult:
        if not self.allow_benchmark_scheme:
            raise ValueError("benchmark:// fetching is disabled outside evaluation lab mode")
        document = self.corpus.resolve(url)
        text = document.text
        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[TRUNCATED]\n"
            truncated = True
        source_type = document.source.source_type.lower()
        content_type = {
            "html": "text/html",
            "md": "text/markdown",
            "txt": "text/plain",
            "csv": "text/csv",
        }.get(source_type, "text/plain")
        return FetchResult(
            ok=True,
            url=url,
            final_url=url,
            status_code=200,
            content_type=content_type,
            extracted_text=text,
            title=document.source.title,
            truncated=truncated,
            strategy="benchmark_fixture",
            word_count=_word_count(text),
            char_count=len(text),
            kind=source_type or "txt",
            canonical_url=url,
            extracted_links=(),
        )

    def fetch_metadata(self, url: str) -> dict:
        return self.corpus.metadata_for_url(url)
