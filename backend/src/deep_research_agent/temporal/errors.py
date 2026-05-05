from __future__ import annotations


class TemporalError(RuntimeError):
    """Base error raised by the temporal intelligence subsystem."""


class TemporalArtifactError(TemporalError):
    """Raised when temporal artifacts cannot be read or written safely."""
