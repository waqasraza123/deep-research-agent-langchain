from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_dotenv_loader: Any
try:
    from dotenv import load_dotenv as _dotenv_loader
except Exception:
    _dotenv_loader = None

load_dotenv: Any = _dotenv_loader


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / ".env"

if load_dotenv is not None and ENV_PATH.exists():
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
    model_provider: str = "openai"
    temperature: float = 0.2

    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_predict: int = 220
    ollama_max_context_tokens: int = 8_192

    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    openai_max_tokens: int = 350
    openai_timeout_s: float = 60.0
    openai_max_retries: int = 1
    openai_max_context_tokens: int = 128_000
    llamacpp_max_context_tokens: int = 8_192

    runs_dir: Path = REPO_ROOT / "runs"
    memory_data_dir: Path | None = None
    memory_enabled: bool = True
    source_reuse_enabled: bool = True
    source_audit_enabled: bool = True
    orchestration_enabled: bool = True
    synthesis_enabled: bool = True
    evaluation_enabled: bool = True
    benchmark_path: Path | None = None
    memory_stale_after_days: int = 30
    max_memory_results: int = 20
    source_scoring_threshold: float = 0.45
    evaluation_threshold: float = 0.65
    checkpoint_path: Path | None = None
    max_page_chars: int = 15_000
    http_timeout_s: float = 20.0
    default_follow_links: bool = False
    default_max_links_per_source: int = 0

    host: str = "127.0.0.1"
    port: int = 8000

    mock_model_name: str = "deterministic-mock-research-model"
    allow_mock_fallback: bool = False
    review_gate_default: bool = False
    evidence_citation_threshold: float = 0.34

    budget_max_model_calls: int = 25
    budget_max_source_fetches: int = 3
    budget_max_generated_chars: int = 80_000
    budget_max_runtime_seconds: float = 180.0
    budget_max_artifacts_size: int = 5_000_000
    budget_max_crawl_expansion: int = 10

    def default_budget(self):
        from .runtime.contracts import RunBudget

        return RunBudget(
            max_model_calls=self.budget_max_model_calls,
            max_source_fetches=self.budget_max_source_fetches,
            max_generated_chars=self.budget_max_generated_chars,
            max_runtime_seconds=self.budget_max_runtime_seconds,
            max_artifacts_size=self.budget_max_artifacts_size,
            max_crawl_expansion=self.budget_max_crawl_expansion,
        )

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
        runs_dir = REPO_ROOT / "runs"
        memory_data_dir_raw = _env_str("MEMORY_DATA_DIR", "")
        benchmark_path_raw = _env_str("BENCHMARK_PATH", "")
        checkpoint_path_raw = _env_str("CHECKPOINT_PATH", "")

        return Settings(
            model_provider=model_provider,
            temperature=_env_float("TEMPERATURE", 0.2),

            ollama_model=_env_str("OLLAMA_MODEL", "llama3.1"),
            ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_num_predict=_clamp_int(_env_int("OLLAMA_NUM_PREDICT", 220), 50, 800),
            ollama_max_context_tokens=_clamp_int(
                _env_int("OLLAMA_MAX_CONTEXT_TOKENS", 8192), 1024, 262144
            ),

            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_max_tokens=openai_max_tokens,
            openai_timeout_s=openai_timeout_s,
            openai_max_retries=openai_max_retries,
            openai_max_context_tokens=_clamp_int(
                _env_int("OPENAI_MAX_CONTEXT_TOKENS", 128000), 4096, 2_000_000
            ),
            llamacpp_max_context_tokens=_clamp_int(
                _env_int("LLAMACPP_MAX_CONTEXT_TOKENS", 8192), 1024, 262144
            ),

            runs_dir=runs_dir,
            memory_data_dir=Path(memory_data_dir_raw)
            if memory_data_dir_raw
            else runs_dir / "_memory",
            memory_enabled=_env_str("MEMORY_ENABLED", "true").lower() in ("1", "true", "yes"),
            source_reuse_enabled=_env_str("SOURCE_REUSE_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            source_audit_enabled=_env_str("SOURCE_AUDIT_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            orchestration_enabled=_env_str("ORCHESTRATION_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            synthesis_enabled=_env_str("SYNTHESIS_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            evaluation_enabled=_env_str("EVALUATION_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            benchmark_path=Path(benchmark_path_raw)
            if benchmark_path_raw
            else REPO_ROOT / "benchmarks",
            memory_stale_after_days=_clamp_int(
                _env_int("MEMORY_STALE_AFTER_DAYS", 30), 1, 3650
            ),
            max_memory_results=_clamp_int(_env_int("MAX_MEMORY_RESULTS", 20), 1, 500),
            source_scoring_threshold=_env_float("SOURCE_SCORING_THRESHOLD", 0.45),
            evaluation_threshold=_env_float("EVALUATION_THRESHOLD", 0.65),
            checkpoint_path=Path(checkpoint_path_raw)
            if checkpoint_path_raw
            else runs_dir / "checkpoints.sqlite",
            max_page_chars=max_page_chars,
            http_timeout_s=http_timeout_s,
            default_follow_links=_env_str("DEFAULT_FOLLOW_LINKS", "false").lower()
            in ("1", "true", "yes"),
            default_max_links_per_source=_clamp_int(
                _env_int("DEFAULT_MAX_LINKS_PER_SOURCE", 0), 0, 10
            ),

            host=_env_str("HOST", "127.0.0.1"),
            port=_env_int("PORT", 8000),

            mock_model_name=_env_str("MOCK_MODEL_NAME", "deterministic-mock-research-model"),
            allow_mock_fallback=_env_str("ALLOW_MOCK_FALLBACK", "false").lower()
            in ("1", "true", "yes"),
            review_gate_default=_env_str("REVIEW_GATE_DEFAULT", "false").lower()
            in ("1", "true", "yes"),
            evidence_citation_threshold=_env_float("EVIDENCE_CITATION_THRESHOLD", 0.34),
            budget_max_model_calls=_clamp_int(_env_int("BUDGET_MAX_MODEL_CALLS", 25), 0, 1000),
            budget_max_source_fetches=_clamp_int(_env_int("BUDGET_MAX_SOURCE_FETCHES", 3), 0, 1000),
            budget_max_generated_chars=_clamp_int(
                _env_int("BUDGET_MAX_GENERATED_CHARS", 80000), 0, 10_000_000
            ),
            budget_max_runtime_seconds=_env_float("BUDGET_MAX_RUNTIME_SECONDS", 180.0),
            budget_max_artifacts_size=_clamp_int(
                _env_int("BUDGET_MAX_ARTIFACTS_SIZE", 5_000_000), 0, 500_000_000
            ),
            budget_max_crawl_expansion=_clamp_int(
                _env_int("BUDGET_MAX_CRAWL_EXPANSION", 10), 0, 1000
            ),
        )
