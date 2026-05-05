from __future__ import annotations


class VerificationError(RuntimeError):
    """Base error for deterministic verification failures."""


class VerificationArtifactError(VerificationError):
    """Raised when verification artifacts cannot be read or written safely."""
