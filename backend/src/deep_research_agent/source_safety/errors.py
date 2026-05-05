from __future__ import annotations


class SourceSafetyError(Exception):
    """Base error for deterministic source-safety processing."""


class SourceSafetyArtifactError(SourceSafetyError):
    """Raised when source safety artifacts cannot be read or written safely."""
