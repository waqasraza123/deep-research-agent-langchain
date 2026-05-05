from __future__ import annotations

from .runtime.mock_model import DeterministicMockChatModel
from .settings import Settings


def create_chat_model(settings: Settings):
    if settings.model_provider == "mock":
        return DeterministicMockChatModel()

    if settings.model_provider in ("openai", "llamacpp"):
        if settings.model_provider == "openai" and not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key or "not-needed",
            temperature=settings.temperature,
            max_tokens=settings.openai_max_tokens,
            timeout=settings.openai_timeout_s,
            max_retries=settings.openai_max_retries,
        )

    if settings.model_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            num_predict=settings.ollama_num_predict,
        )

    raise RuntimeError("Unsupported MODEL_PROVIDER. Use openai, llamacpp, ollama, or mock.")
