from __future__ import annotations


class RetrievalError(Exception):
    """Base retrieval subsystem error."""


class RetrievalIndexError(RetrievalError):
    """Raised when a run cannot be indexed."""


class RetrievalArtifactError(RetrievalError):
    """Raised when retrieval artifacts cannot be read or written."""
