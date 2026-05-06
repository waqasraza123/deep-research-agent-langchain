from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .contracts import AgentControlSettings


class AgentControlPreviewRequest(BaseModel):
    question: str = Field(..., min_length=5)
    urls: list[str] = Field(default_factory=list)
    thread_id: str | None = None
    settings: AgentControlSettings | None = None
    settings_overrides: dict[str, Any] = Field(default_factory=dict)


class AgentControlPreviewResponse(BaseModel):
    thread_id: str
    selected_roles: list[dict[str, Any]]
    selected_skills: list[dict[str, Any]]
    planned_subagents: list[dict[str, Any]]
    policies: dict[str, Any]
    warnings: list[str]
    expected_artifacts: list[str]
