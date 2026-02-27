from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / ".env"

if load_dotenv and ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=False)


def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v.strip() if v and v.strip() else default


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def _clamp_int(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@dataclass(frozen=True)
class Settings:
    model_provider: str
    temperature: float

    ollama_model: str
    ollama_num_predict: int

    openai_base_url: str
    openai_api_key: str
    openai_model: str
    openai_max_tokens: int
    openai_timeout_s: float
    openai_max_retries: int

    runs_dir: Path
    max_page_chars: int
    http_timeout_s: float

    host: str
    port: int

    @staticmethod
    def load() -> "Settings":
        model_provider = _env_str("MODEL_PROVIDER", "openai").lower()

        openai_base_url = _env_str("OPENAI_BASE_URL", "https://api.openai.com/v1")
        openai_api_key = _env_str("OPENAI_API_KEY", "")
        openai_model = _env_str("OPENAI_MODEL", "gpt-5-mini")

        openai_max_tokens = _clamp_int(_env_int("OPENAI_MAX_TOKENS", 350), 50, 800)
        openai_timeout_s = _env_float("OPENAI_TIMEOUT_S", 60.0)
        openai_max_retries = _clamp_int(_env_int("OPENAI_MAX_RETRIES", 1), 0, 2)

        max_page_chars = _clamp_int(_env_int("MAX_PAGE_CHARS", 15000), 2000, 50000)
        http_timeout_s = _env_float("HTTP_TIMEOUT_S", 20.0)

        return Settings(
            model_provider=model_provider,
            temperature=_env_float("TEMPERATURE", 0.2),

            ollama_model=_env_str("OLLAMA_MODEL", "llama3.1"),
            ollama_num_predict=_clamp_int(_env_int("OLLAMA_NUM_PREDICT", 220), 50, 800),

            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_max_tokens=openai_max_tokens,
            openai_timeout_s=openai_timeout_s,
            openai_max_retries=openai_max_retries,

            runs_dir=REPO_ROOT / "runs",
            max_page_chars=max_page_chars,
            http_timeout_s=http_timeout_s,

            host=_env_str("HOST", "127.0.0.1"),
            port=_env_int("PORT", 8000),
        )