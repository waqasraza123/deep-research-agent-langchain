from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .contracts import ResearchAgentRole
from .policy import PolicyEngine


def governed_tool(
    *,
    func: Callable[..., Any],
    role: ResearchAgentRole,
    tool_name: str,
    tool_category: str,
    policy_engine: PolicyEngine,
) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not policy_engine.check_tool(role, tool_name, tool_category):
            return {"ok": False, "error": f"Tool denied by agent control policy: {tool_name}"}
        return func(*args, **kwargs)

    wrapper.__name__ = getattr(func, "__name__", tool_name)
    wrapper.__doc__ = getattr(func, "__doc__", "") or f"Governed tool wrapper for {tool_name}."
    return wrapper
