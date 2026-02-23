from __future__ import annotations

from .settings import Settings


def create_chat_model(settings: Settings):
    """
    Phase 1: free local via Ollama.
    Phase 5: swap to Anthropic here only.
    """
    if settings.model_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            temperature=settings.temperature,
        )

    raise RuntimeError("Unsupported MODEL_PROVIDER for Phase 1. Use MODEL_PROVIDER=ollama.")