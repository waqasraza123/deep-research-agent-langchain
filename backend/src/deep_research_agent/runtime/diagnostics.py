from __future__ import annotations

from deep_research_agent.settings import Settings

from .contracts import RuntimeDiagnostics
from .model_registry import build_model_registry


def build_runtime_diagnostics(settings: Settings) -> RuntimeDiagnostics:
    warnings: list[str] = []
    if settings.model_provider == "openai" and not settings.openai_api_key:
        warnings.append("MODEL_PROVIDER=openai but OPENAI_API_KEY is not configured.")
    if settings.allow_mock_fallback:
        warnings.append("Mock fallback is enabled; failures may produce explicit mock artifacts.")
    return RuntimeDiagnostics(
        ok=not warnings or settings.model_provider == "mock",
        runs_dir=str(settings.runs_dir),
        active_provider=settings.model_provider,
        mock_enabled=settings.model_provider == "mock",
        mock_fallback_enabled=settings.allow_mock_fallback,
        models=build_model_registry(settings),
        default_budget=settings.default_budget(),
        warnings=warnings,
    )
