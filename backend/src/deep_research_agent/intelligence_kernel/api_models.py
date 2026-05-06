from __future__ import annotations

from pydantic import BaseModel, Field

from .contracts import ResearchBlueprint, ResearchComplexity, ResearchIntent, ResearchKernelSettings


class IntelligenceAnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=5)
    urls: list[str] = Field(default_factory=list)
    settings: ResearchKernelSettings | None = None
    thread_id: str | None = None


class IntelligenceAnalyzeResponse(BaseModel):
    intent: ResearchIntent
    complexity: ResearchComplexity
    blueprint: ResearchBlueprint
    warnings: list[dict] = Field(default_factory=list)


class IntelligenceRebuildResponse(BaseModel):
    thread_id: str
    warnings: list[str] = Field(default_factory=list)
    summary: dict
    artifacts: list[dict] = Field(default_factory=list)
