from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ModelProvider = Literal["openai", "ollama", "llamacpp", "mock"]
Locality = Literal["local", "remote", "mock"]
HealthStatus = Literal["configured", "degraded", "unavailable", "mock"]
RunEventType = Literal[
    "run_started",
    "strategy_created",
    "protocol_selected",
    "source_discovery_completed",
    "source_discovery_selected",
    "source_fetch_started",
    "source_fetch_completed",
    "source_fetch_failed",
    "model_call_started",
    "model_call_completed",
    "model_call_failed",
    "artifact_written",
    "budget_warning",
    "budget_exceeded",
    "run_completed",
    "run_failed",
]


class ModelCapability(BaseModel):
    provider: ModelProvider
    model_name: str
    endpoint: str
    supports_tools: bool
    supports_json_mode: bool
    supports_streaming: bool
    max_context_tokens: int
    recommended_temperature: float
    local_or_remote: Locality
    requires_api_key: bool
    health_status: HealthStatus
    warnings: list[str] = Field(default_factory=list)


class RunBudget(BaseModel):
    max_model_calls: int = Field(default=25, ge=0)
    max_source_fetches: int = Field(default=3, ge=0)
    max_generated_chars: int = Field(default=80_000, ge=0)
    max_runtime_seconds: float = Field(default=180.0, ge=0)
    max_artifacts_size: int = Field(default=5_000_000, ge=0)
    max_crawl_expansion: int = Field(default=10, ge=0)


class RunBudgetUsage(BaseModel):
    model_calls: int = 0
    source_fetches: int = 0
    generated_chars: int = 0
    runtime_seconds: float = 0.0
    artifacts_size: int = 0
    crawl_expansion: int = 0
    exceeded_reasons: list[str] = Field(default_factory=list)
    warning_reasons: list[str] = Field(default_factory=list)


class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=2, ge=1, le=8)
    initial_backoff_s: float = Field(default=0.25, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1)
    max_backoff_s: float = Field(default=3.0, ge=0)
    jitter_s: float = Field(default=0.0, ge=0)


class RunEvent(BaseModel):
    event_type: RunEventType
    thread_id: str
    timestamp: str
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeDiagnostics(BaseModel):
    ok: bool
    runs_dir: str
    active_provider: str
    mock_enabled: bool
    mock_fallback_enabled: bool
    models: list[ModelCapability]
    default_budget: RunBudget
    warnings: list[str] = Field(default_factory=list)
