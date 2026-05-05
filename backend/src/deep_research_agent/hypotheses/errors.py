from __future__ import annotations


class HypothesisError(Exception):
    """Base error for deterministic hypothesis analysis."""


class HypothesisArtifactError(HypothesisError):
    """Raised when hypothesis artifacts cannot be read or written safely."""
