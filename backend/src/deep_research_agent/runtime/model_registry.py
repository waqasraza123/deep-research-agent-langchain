from __future__ import annotations

from deep_research_agent.settings import Settings

from .contracts import ModelCapability


def build_model_registry(settings: Settings) -> list[ModelCapability]:
    provider = settings.model_provider
    models: list[ModelCapability] = []

    openai_warnings: list[str] = []
    openai_status = "configured"
    if not settings.openai_api_key:
        openai_warnings.append("OPENAI_API_KEY is not set.")
        openai_status = "unavailable"
    models.append(
        ModelCapability(
            provider="openai",
            model_name=settings.openai_model,
            endpoint=settings.openai_base_url,
            supports_tools=True,
            supports_json_mode=True,
            supports_streaming=True,
            max_context_tokens=settings.openai_max_context_tokens,
            recommended_temperature=settings.temperature,
            local_or_remote="remote",
            requires_api_key=True,
            health_status=openai_status if provider == "openai" else openai_status,
            warnings=openai_warnings,
        )
    )

    llama_warnings: list[str] = []
    if not settings.openai_base_url:
        llama_warnings.append(
            "OPENAI_BASE_URL must point to the llama.cpp OpenAI-compatible server."
        )
    models.append(
        ModelCapability(
            provider="llamacpp",
            model_name=settings.openai_model,
            endpoint=settings.openai_base_url,
            supports_tools=False,
            supports_json_mode=False,
            supports_streaming=True,
            max_context_tokens=settings.llamacpp_max_context_tokens,
            recommended_temperature=min(settings.temperature, 0.2),
            local_or_remote="local",
            requires_api_key=False,
            health_status="configured",
            warnings=llama_warnings,
        )
    )

    models.append(
        ModelCapability(
            provider="ollama",
            model_name=settings.ollama_model,
            endpoint=settings.ollama_base_url,
            supports_tools=False,
            supports_json_mode=True,
            supports_streaming=True,
            max_context_tokens=settings.ollama_max_context_tokens,
            recommended_temperature=min(settings.temperature, 0.2),
            local_or_remote="local",
            requires_api_key=False,
            health_status="configured",
            warnings=["Health is configuration-only; no startup network probe is performed."],
        )
    )

    models.append(
        ModelCapability(
            provider="mock",
            model_name=settings.mock_model_name,
            endpoint="offline",
            supports_tools=True,
            supports_json_mode=True,
            supports_streaming=False,
            max_context_tokens=16_000,
            recommended_temperature=0.0,
            local_or_remote="mock",
            requires_api_key=False,
            health_status="mock",
            warnings=["Deterministic mock output. Not suitable for production research claims."],
        )
    )

    return models
