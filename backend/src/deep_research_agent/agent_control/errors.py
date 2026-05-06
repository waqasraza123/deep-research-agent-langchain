from __future__ import annotations


class AgentControlError(Exception):
    """Base error for deterministic agent control-plane failures."""


class AgentControlValidationError(AgentControlError):
    """Raised when a typed control-plane contract is invalid."""


class AgentControlPolicyError(AgentControlError):
    """Raised when policy strict mode requires a hard failure."""
