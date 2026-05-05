from __future__ import annotations


class DocumentIntelligenceError(Exception):
    """Base error for deterministic document intelligence failures."""


class UnsafeArtifactPathError(DocumentIntelligenceError):
    """Raised when an artifact path escapes the run directory."""
