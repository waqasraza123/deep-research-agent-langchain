from __future__ import annotations


class ProvenanceError(RuntimeError):
    """Base error for provenance generation failures."""


class ProvenancePathError(ProvenanceError, ValueError):
    """Raised when a run or artifact path would escape the configured runs directory."""
