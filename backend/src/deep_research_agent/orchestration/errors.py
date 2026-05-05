from __future__ import annotations


class OrchestrationError(Exception):
    """Base exception for local research orchestration failures."""


class TaskGraphError(OrchestrationError):
    """Raised when a task graph is invalid or cannot be executed."""


class SpecialistExecutionError(OrchestrationError):
    """Raised by deterministic specialist stages when their contract is violated."""
