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
    protocol_selection_enabled: bool = True
    intelligence_profile: str = "balanced_research"
    source_discovery_enabled: bool = False
    source_discovery_provider: str = "disabled"
    source_discovery_max_queries: int = 8
    source_discovery_max_candidates_per_query: int = 5
    source_discovery_max_selected_sources: int = 3
    source_discovery_allow_secondary_sources: bool = True
    source_discovery_allow_forums: bool = False
    source_discovery_require_primary: bool = True
    source_discovery_freshness_required: bool | None = None
    max_discovery_queries: int = 8
    max_selected_discovered_sources: int = 3
    document_intelligence_enabled: bool = True
    chunk_max_chars: int = 3200
    chunk_overlap_chars: int = 300
    retrieval_enabled: bool = True
    embedding_provider: str = "disabled"
    context_pack_max_chars: int = 11_000
    orchestration_enabled: bool = True
    synthesis_enabled: bool = True
    evaluation_enabled: bool = True
    verification_enabled: bool = True
    hypothesis_engine_enabled: bool = True
    temporal_intelligence_enabled: bool = True
    source_safety_enabled: bool = True
    quantitative_intelligence_enabled: bool = True
    provenance_enabled: bool = True
    verification_gate_enabled: bool = False
    max_verification_tasks: int = 12
    high_risk_source_policy: str = "quote_high_exclude_critical"
    max_hypotheses: int = 12
    max_hypothesis_evidence_items: int = 6
    freshness_warning_threshold_days: int = 365
    quantitative_extraction_enabled: bool = True
    provenance_manifest_enabled: bool = True
    replay_plan_enabled: bool = True
    confidence_threshold_for_review: float = 0.55
    benchmark_path: Path | None = None
    evaluation_lab_enabled: bool = True
    evaluation_lab_cases_dir: Path | None = None
    evaluation_lab_runs_dir: Path | None = None
    evaluation_lab_allow_benchmark_scheme: bool = False
    evaluation_lab_default_use_mock_agent: bool = True
    evaluation_lab_default_use_offline_fetcher: bool = True
    evaluation_lab_fail_on_invalid_case: bool = True
    evaluation_lab_max_cases_per_run: int = 25
    evaluation_lab_copy_run_artifacts: bool = True
    evaluation_lab_minimum_passing_score: float = 0.75
    evaluation_lab_strict_adversarial_checks: bool = True
    evaluation_lab_strict_numeric_checks: bool = True
    evaluation_lab_strict_temporal_checks: bool = True
    evaluation_lab_strict_citation_checks: bool = False
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
    intelligence_kernel_enabled: bool = True
    intelligence_offline_mode: bool = True
    intelligence_mock_model_allowed: bool = True
    intelligence_source_reasoning_enabled: bool = True
    intelligence_critique_enabled: bool = True
    intelligence_verification_enabled: bool = True
    intelligence_confidence_enabled: bool = True
    intelligence_strict_citation_mode: bool = False
    intelligence_max_reasoning_passes: int = 8
    intelligence_max_source_units: int = 100
    intelligence_max_evidence_units: int = 500
    intelligence_max_claims: int = 200
    intelligence_max_claims_to_verify: int = 50
    intelligence_max_artifact_bytes: int = 5_000_000
    intelligence_fail_on_critical_warnings: bool = False
    intelligence_sensitive_domain_review_required: bool = True
    intelligence_produce_markdown_artifacts: bool = True
    intelligence_produce_json_artifacts: bool = True

    budget_max_model_calls: int = 25
    budget_max_source_fetches: int = 3
    budget_max_generated_chars: int = 80_000
    budget_max_runtime_seconds: float = 180.0
    budget_max_artifacts_size: int = 5_000_000
    budget_max_crawl_expansion: int = 10

    runtime_control_enabled: bool = True
    runtime_async_enabled: bool = False
    runtime_sqlite_path: Path | None = None
    runtime_worker_poll_interval_seconds: float = 1.0
    runtime_lease_seconds: int = 120
    runtime_heartbeat_seconds: int = 30
    runtime_max_attempts: int = 3
    runtime_retry_backoff_initial_seconds: float = 2.0
    runtime_retry_backoff_multiplier: float = 2.0
    runtime_retry_backoff_max_seconds: float = 60.0
    runtime_fail_on_budget_exceeded: bool = False
    runtime_max_runtime_seconds: int = 900
    runtime_max_stage_seconds: int = 300
    runtime_max_events: int = 5000
    runtime_max_artifact_bytes: int = 50_000_000
    runtime_allow_force_cancel: bool = True
    runtime_resume_enabled: bool = True
    runtime_dead_letter_enabled: bool = True
    runtime_run_postprocessing: bool = True
    runtime_mock_agent_execution_enabled: bool = False

    agent_control_enabled: bool = True
    agent_control_strict_role_isolation: bool = True
    agent_control_source_context_quarantine_enabled: bool = True
    agent_control_tool_governance_enabled: bool = True
    agent_control_filesystem_governance_enabled: bool = True
    agent_control_skill_selection_enabled: bool = True
    agent_control_subagent_planning_enabled: bool = True
    agent_control_trace_analysis_enabled: bool = True
    agent_control_artifact_validation_enabled: bool = True
    agent_control_max_compiled_instruction_chars: int = 12_000
    agent_control_max_subagents: int = 8
    agent_control_max_handoffs: int = 32
    agent_control_max_context_chars_per_role: int = 16_000
    agent_control_fail_on_policy_violation: bool = False
    agent_control_fail_on_missing_required_artifact: bool = False
    agent_control_allow_mock_subagents: bool = True
    agent_control_produce_markdown_artifacts: bool = True
    agent_control_produce_json_artifacts: bool = True

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
        max_discovery_queries = _clamp_int(
            _env_int(
                "MAX_DISCOVERY_QUERIES",
                _env_int("SOURCE_DISCOVERY_MAX_QUERIES", 8),
            ),
            0,
            20,
        )
        max_selected_discovered_sources = _clamp_int(
            _env_int(
                "MAX_SELECTED_DISCOVERED_SOURCES",
                _env_int("SOURCE_DISCOVERY_MAX_SELECTED_SOURCES", 3),
            ),
            0,
            20,
        )
        runs_dir = REPO_ROOT / "runs"
        memory_data_dir_raw = _env_str("MEMORY_DATA_DIR", "")
        benchmark_path_raw = _env_str("BENCHMARK_PATH", "")
        evaluation_lab_cases_dir_raw = _env_str("EVALUATION_LAB_CASES_DIR", "")
        evaluation_lab_runs_dir_raw = _env_str("EVALUATION_LAB_RUNS_DIR", "")
        checkpoint_path_raw = _env_str("CHECKPOINT_PATH", "")
        runtime_sqlite_path_raw = _env_str("RUNTIME_SQLITE_PATH", "")

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
            protocol_selection_enabled=_env_str("PROTOCOL_SELECTION_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            intelligence_profile=_env_str("INTELLIGENCE_PROFILE", "balanced_research"),
            source_discovery_enabled=_env_str("SOURCE_DISCOVERY_ENABLED", "false").lower()
            in ("1", "true", "yes"),
            source_discovery_provider=_env_str("SOURCE_DISCOVERY_PROVIDER", "disabled").lower(),
            source_discovery_max_queries=max_discovery_queries,
            source_discovery_max_candidates_per_query=_clamp_int(
                _env_int("SOURCE_DISCOVERY_MAX_CANDIDATES_PER_QUERY", 5), 0, 20
            ),
            source_discovery_max_selected_sources=max_selected_discovered_sources,
            source_discovery_allow_secondary_sources=_env_str(
                "SOURCE_DISCOVERY_ALLOW_SECONDARY_SOURCES", "true"
            ).lower()
            in ("1", "true", "yes"),
            source_discovery_allow_forums=_env_str("SOURCE_DISCOVERY_ALLOW_FORUMS", "false").lower()
            in ("1", "true", "yes"),
            source_discovery_require_primary=_env_str(
                "SOURCE_DISCOVERY_REQUIRE_PRIMARY", "true"
            ).lower()
            in ("1", "true", "yes"),
            source_discovery_freshness_required=(
                True
                if _env_str("SOURCE_DISCOVERY_FRESHNESS_REQUIRED", "").lower()
                in ("1", "true", "yes")
                else False
                if _env_str("SOURCE_DISCOVERY_FRESHNESS_REQUIRED", "").lower()
                in ("0", "false", "no")
                else None
            ),
            max_discovery_queries=max_discovery_queries,
            max_selected_discovered_sources=max_selected_discovered_sources,
            document_intelligence_enabled=_env_str("DOCUMENT_INTELLIGENCE_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            chunk_max_chars=_clamp_int(_env_int("CHUNK_MAX_CHARS", 3200), 400, 20_000),
            chunk_overlap_chars=_clamp_int(_env_int("CHUNK_OVERLAP_CHARS", 300), 0, 5000),
            retrieval_enabled=_env_str("RETRIEVAL_ENABLED", "true").lower() in ("1", "true", "yes"),
            embedding_provider=_env_str("EMBEDDING_PROVIDER", "disabled").lower(),
            context_pack_max_chars=_clamp_int(
                _env_int("CONTEXT_PACK_MAX_CHARS", 11000), 1000, 80_000
            ),
            orchestration_enabled=_env_str("ORCHESTRATION_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            synthesis_enabled=_env_str("SYNTHESIS_ENABLED", "true").lower() in ("1", "true", "yes"),
            evaluation_enabled=_env_str("EVALUATION_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            verification_enabled=_env_str("VERIFICATION_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            hypothesis_engine_enabled=_env_str("HYPOTHESIS_ENGINE_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            temporal_intelligence_enabled=_env_str("TEMPORAL_INTELLIGENCE_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            source_safety_enabled=_env_str("SOURCE_SAFETY_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            quantitative_intelligence_enabled=_env_str(
                "QUANTITATIVE_INTELLIGENCE_ENABLED", "true"
            ).lower()
            in ("1", "true", "yes"),
            provenance_enabled=_env_str("PROVENANCE_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            verification_gate_enabled=_env_str("VERIFICATION_GATE_ENABLED", "false").lower()
            in ("1", "true", "yes"),
            max_verification_tasks=_clamp_int(_env_int("MAX_VERIFICATION_TASKS", 12), 0, 100),
            high_risk_source_policy=_env_str(
                "HIGH_RISK_SOURCE_POLICY", "quote_high_exclude_critical"
            ),
            max_hypotheses=_clamp_int(_env_int("MAX_HYPOTHESES", 12), 1, 100),
            max_hypothesis_evidence_items=_clamp_int(
                _env_int("MAX_HYPOTHESIS_EVIDENCE_ITEMS", 6), 1, 50
            ),
            freshness_warning_threshold_days=_clamp_int(
                _env_int("FRESHNESS_WARNING_THRESHOLD_DAYS", 365), 1, 3650
            ),
            quantitative_extraction_enabled=_env_str(
                "QUANTITATIVE_EXTRACTION_ENABLED", "true"
            ).lower()
            in ("1", "true", "yes"),
            provenance_manifest_enabled=_env_str("PROVENANCE_MANIFEST_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            replay_plan_enabled=_env_str("REPLAY_PLAN_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            confidence_threshold_for_review=_env_float("CONFIDENCE_THRESHOLD_FOR_REVIEW", 0.55),
            benchmark_path=Path(benchmark_path_raw)
            if benchmark_path_raw
            else REPO_ROOT / "benchmarks",
            evaluation_lab_enabled=_env_str("EVALUATION_LAB_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            evaluation_lab_cases_dir=Path(evaluation_lab_cases_dir_raw)
            if evaluation_lab_cases_dir_raw
            else REPO_ROOT / "backend" / "benchmarks" / "cases",
            evaluation_lab_runs_dir=Path(evaluation_lab_runs_dir_raw)
            if evaluation_lab_runs_dir_raw
            else REPO_ROOT / "backend" / "benchmark_runs",
            evaluation_lab_allow_benchmark_scheme=_env_str(
                "EVALUATION_LAB_ALLOW_BENCHMARK_SCHEME", "false"
            ).lower()
            in ("1", "true", "yes"),
            evaluation_lab_default_use_mock_agent=_env_str(
                "EVALUATION_LAB_DEFAULT_USE_MOCK_AGENT", "true"
            ).lower()
            in ("1", "true", "yes"),
            evaluation_lab_default_use_offline_fetcher=_env_str(
                "EVALUATION_LAB_DEFAULT_USE_OFFLINE_FETCHER", "true"
            ).lower()
            in ("1", "true", "yes"),
            evaluation_lab_fail_on_invalid_case=_env_str(
                "EVALUATION_LAB_FAIL_ON_INVALID_CASE", "true"
            ).lower()
            in ("1", "true", "yes"),
            evaluation_lab_max_cases_per_run=_clamp_int(
                _env_int("EVALUATION_LAB_MAX_CASES_PER_RUN", 25), 1, 500
            ),
            evaluation_lab_copy_run_artifacts=_env_str(
                "EVALUATION_LAB_COPY_RUN_ARTIFACTS", "true"
            ).lower()
            in ("1", "true", "yes"),
            evaluation_lab_minimum_passing_score=_env_float(
                "EVALUATION_LAB_MINIMUM_PASSING_SCORE", 0.75
            ),
            evaluation_lab_strict_adversarial_checks=_env_str(
                "EVALUATION_LAB_STRICT_ADVERSARIAL_CHECKS", "true"
            ).lower()
            in ("1", "true", "yes"),
            evaluation_lab_strict_numeric_checks=_env_str(
                "EVALUATION_LAB_STRICT_NUMERIC_CHECKS", "true"
            ).lower()
            in ("1", "true", "yes"),
            evaluation_lab_strict_temporal_checks=_env_str(
                "EVALUATION_LAB_STRICT_TEMPORAL_CHECKS", "true"
            ).lower()
            in ("1", "true", "yes"),
            evaluation_lab_strict_citation_checks=_env_str(
                "EVALUATION_LAB_STRICT_CITATION_CHECKS", "false"
            ).lower()
            in ("1", "true", "yes"),
            memory_stale_after_days=_clamp_int(_env_int("MEMORY_STALE_AFTER_DAYS", 30), 1, 3650),
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
            intelligence_kernel_enabled=_env_str("INTELLIGENCE_KERNEL_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            intelligence_offline_mode=_env_str("INTELLIGENCE_OFFLINE_MODE", "true").lower()
            in ("1", "true", "yes"),
            intelligence_mock_model_allowed=_env_str(
                "INTELLIGENCE_MOCK_MODEL_ALLOWED", "true"
            ).lower()
            in ("1", "true", "yes"),
            intelligence_source_reasoning_enabled=_env_str(
                "INTELLIGENCE_SOURCE_REASONING_ENABLED", "true"
            ).lower()
            in ("1", "true", "yes"),
            intelligence_critique_enabled=_env_str("INTELLIGENCE_CRITIQUE_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            intelligence_verification_enabled=_env_str(
                "INTELLIGENCE_VERIFICATION_ENABLED", "true"
            ).lower()
            in ("1", "true", "yes"),
            intelligence_confidence_enabled=_env_str(
                "INTELLIGENCE_CONFIDENCE_ENABLED", "true"
            ).lower()
            in ("1", "true", "yes"),
            intelligence_strict_citation_mode=_env_str(
                "INTELLIGENCE_STRICT_CITATION_MODE", "false"
            ).lower()
            in ("1", "true", "yes"),
            intelligence_max_reasoning_passes=_clamp_int(
                _env_int("INTELLIGENCE_MAX_REASONING_PASSES", 8), 1, 50
            ),
            intelligence_max_source_units=_clamp_int(
                _env_int("INTELLIGENCE_MAX_SOURCE_UNITS", 100), 1, 1000
            ),
            intelligence_max_evidence_units=_clamp_int(
                _env_int("INTELLIGENCE_MAX_EVIDENCE_UNITS", 500), 1, 5000
            ),
            intelligence_max_claims=_clamp_int(_env_int("INTELLIGENCE_MAX_CLAIMS", 200), 1, 2000),
            intelligence_max_claims_to_verify=_clamp_int(
                _env_int("INTELLIGENCE_MAX_CLAIMS_TO_VERIFY", 50), 0, 500
            ),
            intelligence_max_artifact_bytes=_clamp_int(
                _env_int("INTELLIGENCE_MAX_ARTIFACT_BYTES", 5_000_000),
                1_000,
                500_000_000,
            ),
            intelligence_fail_on_critical_warnings=_env_str(
                "INTELLIGENCE_FAIL_ON_CRITICAL_WARNINGS", "false"
            ).lower()
            in ("1", "true", "yes"),
            intelligence_sensitive_domain_review_required=_env_str(
                "INTELLIGENCE_SENSITIVE_DOMAIN_REVIEW_REQUIRED", "true"
            ).lower()
            in ("1", "true", "yes"),
            intelligence_produce_markdown_artifacts=_env_str(
                "INTELLIGENCE_PRODUCE_MARKDOWN_ARTIFACTS", "true"
            ).lower()
            in ("1", "true", "yes"),
            intelligence_produce_json_artifacts=_env_str(
                "INTELLIGENCE_PRODUCE_JSON_ARTIFACTS", "true"
            ).lower()
            in ("1", "true", "yes"),
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
            runtime_control_enabled=_env_str("RUNTIME_CONTROL_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            runtime_async_enabled=_env_str("RUNTIME_ASYNC_ENABLED", "false").lower()
            in ("1", "true", "yes"),
            runtime_sqlite_path=Path(runtime_sqlite_path_raw)
            if runtime_sqlite_path_raw
            else None,
            runtime_worker_poll_interval_seconds=_env_float(
                "RUNTIME_WORKER_POLL_INTERVAL_SECONDS", 1.0
            ),
            runtime_lease_seconds=_clamp_int(_env_int("RUNTIME_LEASE_SECONDS", 120), 1, 86_400),
            runtime_heartbeat_seconds=_clamp_int(
                _env_int("RUNTIME_HEARTBEAT_SECONDS", 30), 1, 86_400
            ),
            runtime_max_attempts=_clamp_int(_env_int("RUNTIME_MAX_ATTEMPTS", 3), 1, 20),
            runtime_retry_backoff_initial_seconds=_env_float(
                "RUNTIME_RETRY_BACKOFF_INITIAL_SECONDS", 2.0
            ),
            runtime_retry_backoff_multiplier=_env_float(
                "RUNTIME_RETRY_BACKOFF_MULTIPLIER", 2.0
            ),
            runtime_retry_backoff_max_seconds=_env_float(
                "RUNTIME_RETRY_BACKOFF_MAX_SECONDS", 60.0
            ),
            runtime_fail_on_budget_exceeded=_env_str(
                "RUNTIME_FAIL_ON_BUDGET_EXCEEDED", "false"
            ).lower()
            in ("1", "true", "yes"),
            runtime_max_runtime_seconds=_clamp_int(
                _env_int("RUNTIME_MAX_RUNTIME_SECONDS", 900), 0, 86_400
            ),
            runtime_max_stage_seconds=_clamp_int(
                _env_int("RUNTIME_MAX_STAGE_SECONDS", 300), 0, 86_400
            ),
            runtime_max_events=_clamp_int(_env_int("RUNTIME_MAX_EVENTS", 5000), 0, 1_000_000),
            runtime_max_artifact_bytes=_clamp_int(
                _env_int("RUNTIME_MAX_ARTIFACT_BYTES", 50_000_000), 0, 1_000_000_000
            ),
            runtime_allow_force_cancel=_env_str("RUNTIME_ALLOW_FORCE_CANCEL", "true").lower()
            in ("1", "true", "yes"),
            runtime_resume_enabled=_env_str("RUNTIME_RESUME_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            runtime_dead_letter_enabled=_env_str("RUNTIME_DEAD_LETTER_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            runtime_run_postprocessing=_env_str("RUNTIME_RUN_POSTPROCESSING", "true").lower()
            in ("1", "true", "yes"),
            runtime_mock_agent_execution_enabled=_env_str(
                "RUNTIME_MOCK_AGENT_EXECUTION_ENABLED", "false"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_enabled=_env_str("AGENT_CONTROL_ENABLED", "true").lower()
            in ("1", "true", "yes"),
            agent_control_strict_role_isolation=_env_str(
                "AGENT_CONTROL_STRICT_ROLE_ISOLATION", "true"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_source_context_quarantine_enabled=_env_str(
                "AGENT_CONTROL_SOURCE_CONTEXT_QUARANTINE_ENABLED", "true"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_tool_governance_enabled=_env_str(
                "AGENT_CONTROL_TOOL_GOVERNANCE_ENABLED", "true"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_filesystem_governance_enabled=_env_str(
                "AGENT_CONTROL_FILESYSTEM_GOVERNANCE_ENABLED", "true"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_skill_selection_enabled=_env_str(
                "AGENT_CONTROL_SKILL_SELECTION_ENABLED", "true"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_subagent_planning_enabled=_env_str(
                "AGENT_CONTROL_SUBAGENT_PLANNING_ENABLED", "true"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_trace_analysis_enabled=_env_str(
                "AGENT_CONTROL_TRACE_ANALYSIS_ENABLED", "true"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_artifact_validation_enabled=_env_str(
                "AGENT_CONTROL_ARTIFACT_VALIDATION_ENABLED", "true"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_max_compiled_instruction_chars=_clamp_int(
                _env_int("AGENT_CONTROL_MAX_COMPILED_INSTRUCTION_CHARS", 12000),
                1000,
                100_000,
            ),
            agent_control_max_subagents=_clamp_int(
                _env_int("AGENT_CONTROL_MAX_SUBAGENTS", 8), 0, 32
            ),
            agent_control_max_handoffs=_clamp_int(
                _env_int("AGENT_CONTROL_MAX_HANDOFFS", 32), 0, 200
            ),
            agent_control_max_context_chars_per_role=_clamp_int(
                _env_int("AGENT_CONTROL_MAX_CONTEXT_CHARS_PER_ROLE", 16000),
                1000,
                200_000,
            ),
            agent_control_fail_on_policy_violation=_env_str(
                "AGENT_CONTROL_FAIL_ON_POLICY_VIOLATION", "false"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_fail_on_missing_required_artifact=_env_str(
                "AGENT_CONTROL_FAIL_ON_MISSING_REQUIRED_ARTIFACT", "false"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_allow_mock_subagents=_env_str(
                "AGENT_CONTROL_ALLOW_MOCK_SUBAGENTS", "true"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_produce_markdown_artifacts=_env_str(
                "AGENT_CONTROL_PRODUCE_MARKDOWN_ARTIFACTS", "true"
            ).lower()
            in ("1", "true", "yes"),
            agent_control_produce_json_artifacts=_env_str(
                "AGENT_CONTROL_PRODUCE_JSON_ARTIFACTS", "true"
            ).lower()
            in ("1", "true", "yes"),
        )
