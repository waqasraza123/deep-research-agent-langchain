from __future__ import annotations


class KernelError(RuntimeError):
    """Base error for deterministic intelligence-kernel failures."""


class KernelArtifactError(KernelError):
    """Raised when an artifact cannot be safely read or written."""


class KernelPassError(KernelError):
    """Raised when a required intelligence pass fails."""
