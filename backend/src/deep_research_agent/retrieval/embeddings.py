from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    name: str = "base"

    @property
    def enabled(self) -> bool:
        return False

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class DisabledEmbeddingProvider(EmbeddingProvider):
    name = "disabled"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


class MockEmbeddingProvider(EmbeddingProvider):
    name = "mock"

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = max(4, int(dimensions))

    @property
    def enabled(self) -> bool:
        return True

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for raw in (text or "").lower().split():
            token = raw.strip(".,;:!?()[]{}\"'")
            if not token:
                continue
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = digest[0] % self.dimensions
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class PlaceholderEmbeddingProvider(EmbeddingProvider):
    name = "placeholder"

    def __init__(self, provider_name: str = "future") -> None:
        self.name = provider_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError(
            "Live embedding providers are intentionally not configured in the "
            "offline retrieval layer."
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    denom = math.sqrt(sum(v * v for v in left)) * math.sqrt(sum(v * v for v in right))
    if denom <= 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denom
