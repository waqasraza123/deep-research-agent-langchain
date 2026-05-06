from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .contracts import ResearchJobStatus, ResearchStage, RuntimeBudget


class RuntimeJobSubmitRequest(BaseModel):
    question: str = Field(..., min_length=5)
    urls: list[str] = Field(default_factory=list)
    thread_id: str | None = None
    idempotency_key: str | None = None
    priority: int = 0
    settings: dict[str, Any] = Field(default_factory=dict)
    budget: RuntimeBudget | None = None
    max_attempts: int | None = Field(default=None, ge=1, le=20)
    run_now: bool = False
    mock_agent_execution: bool | None = None
    resubmit_completed: bool = False


class RuntimeJobControlRequest(BaseModel):
    requested_by: str = "operator"
    reason: str = ""
    force: bool = False


class RuntimeJobListFilters(BaseModel):
    status: ResearchJobStatus | None = None
    stage: ResearchStage | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    has_errors: bool | None = None
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

