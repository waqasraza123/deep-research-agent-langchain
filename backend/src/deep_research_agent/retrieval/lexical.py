from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .contracts import RetrievalChunk

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-'.%]*")


def normalize_token(token: str) -> str:
    token = (token or "").strip("'\"“”‘’.,;:!?()[]{}").lower()
    if token.endswith("'s"):
        token = token[:-2]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text or ""):
        token = normalize_token(raw)
        if not token or len(token) <= 1:
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def extract_quoted_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for match in re.finditer(r"[\"'“”]([^\"'“”]{3,80})[\"'“”]", text or ""):
        phrase = " ".join(match.group(1).split())
        if phrase and phrase.lower() not in {p.lower() for p in phrases}:
            phrases.append(phrase)
    return phrases


def phrase_match_score(text: str, phrases: Iterable[str]) -> tuple[float, list[str]]:
    lower = (text or "").lower()
    matched: list[str] = []
    for phrase in phrases:
        clean = " ".join((phrase or "").split())
        if len(clean) < 3:
            continue
        if clean.lower() in lower:
            matched.append(clean)
    if not matched:
        return 0.0, []
    return min(1.0, 0.35 + 0.2 * len(matched)), matched


@dataclass
class LexicalSearchHit:
    chunk: RetrievalChunk
    score: float
    matched_terms: list[str] = field(default_factory=list)


class LexicalIndex:
    def __init__(self, chunks: list[RetrievalChunk]) -> None:
        self.chunks = chunks
        self.doc_count = len(chunks)
        self.term_freqs: dict[str, Counter[str]] = {}
        self.doc_freqs: Counter[str] = Counter()
        self.doc_lengths: dict[str, int] = {}
        self.avg_doc_length = 1.0
        self._build()

    def _build(self) -> None:
        total_length = 0
        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            self.term_freqs[chunk.chunk_id] = counts
            self.doc_lengths[chunk.chunk_id] = max(1, sum(counts.values()))
            total_length += self.doc_lengths[chunk.chunk_id]
            for token in counts:
                self.doc_freqs[token] += 1
        if self.chunks:
            self.avg_doc_length = max(1.0, total_length / len(self.chunks))

    def idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log(1.0 + ((self.doc_count - df + 0.5) / (df + 0.5))) if self.doc_count else 0.0

    def score_chunk(self, query: str, chunk: RetrievalChunk) -> tuple[float, list[str]]:
        query_terms = tokenize(query)
        if not query_terms:
            return 0.0, []
        counts = self.term_freqs.get(chunk.chunk_id, Counter())
        if not counts:
            return 0.0, []

        k1 = 1.35
        b = 0.72
        length = self.doc_lengths.get(chunk.chunk_id, 1)
        score = 0.0
        matched: list[str] = []
        seen: set[str] = set()
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf <= 0:
                continue
            if term not in seen:
                matched.append(term)
                seen.add(term)
            denom = tf + k1 * (1 - b + b * (length / self.avg_doc_length))
            score += self.idf(term) * ((tf * (k1 + 1)) / denom)
        norm = min(1.0, score / max(2.5, len(set(query_terms)) * 0.9))
        return norm, matched

    def search(self, query: str, *, limit: int = 50) -> list[LexicalSearchHit]:
        hits: list[LexicalSearchHit] = []
        for chunk in self.chunks:
            score, matched = self.score_chunk(query, chunk)
            if score <= 0:
                continue
            hits.append(LexicalSearchHit(chunk=chunk, score=score, matched_terms=matched))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]


def overlap_score(left: Iterable[str], right: Iterable[str]) -> tuple[float, list[str]]:
    left_norm = {normalize_token(item) for item in left if normalize_token(item)}
    right_norm = {normalize_token(item) for item in right if normalize_token(item)}
    if not left_norm or not right_norm:
        return 0.0, []
    overlap = sorted(left_norm & right_norm)
    if not overlap:
        return 0.0, []
    return min(1.0, len(overlap) / max(1, min(len(left_norm), len(right_norm)))), overlap


def group_duplicate_texts(chunks: list[RetrievalChunk]) -> dict[str, set[str]]:
    by_fingerprint: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        terms = sorted(tokenize(chunk.text)[:80])
        fingerprint = " ".join(terms[:40])
        if fingerprint:
            by_fingerprint[fingerprint].add(chunk.chunk_id)
    return {key: value for key, value in by_fingerprint.items() if len(value) > 1}
