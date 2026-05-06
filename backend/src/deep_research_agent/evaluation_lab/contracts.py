from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def model_to_plain(model: Any) -> Any:
    if isinstance(model, list):
        return [model_to_plain(item) for item in model]
    if isinstance(model, dict):
        return {str(key): model_to_plain(value) for key, value in model.items()}
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(model, BaseModel):
        return model.dict()
    if isinstance(model, Enum):
        return model.value
    return model


class LabModel(BaseModel):
    def to_json_dict(self) -> dict[str, Any]:
        return model_to_plain(self)

    def to_markdown(self) -> str:
        title = self.__class__.__name__
        lines = [f"# {title}", ""]
        for key, value in self.to_json_dict().items():
            if isinstance(value, (list, dict)):
                lines.extend([f"## {key}", "", "```json"])
                import json

                lines.append(json.dumps(value, indent=2, sort_keys=True))
                lines.extend(["```", ""])
            else:
                lines.append(f"- **{key}**: {value}")
        return "\n".join(lines).rstrip() + "\n"


class BenchmarkCategory(str, Enum):
    simple_factual = "simple_factual"
    comparative_analysis = "comparative_analysis"
    technical_due_diligence = "technical_due_diligence"
    implementation_planning = "implementation_planning"
    source_quality = "source_quality"
    citation_quality = "citation_quality"
    prompt_injection = "prompt_injection"
    source_poisoning = "source_poisoning"
    stale_source = "stale_source"
    contradictory_sources = "contradictory_sources"
    numeric_reasoning = "numeric_reasoning"
    table_reasoning = "table_reasoning"
    missing_primary_source = "missing_primary_source"
    sensitive_domain = "sensitive_domain"
    artifact_integrity = "artifact_integrity"
    regression_smoke = "regression_smoke"


class BenchmarkDifficulty(str, Enum):
    easy = "easy"
    moderate = "moderate"
    hard = "hard"
    adversarial = "adversarial"


class TrapType(str, Enum):
    prompt_injection = "prompt_injection"
    fake_citation = "fake_citation"
    stale_date = "stale_date"
    conflicting_number = "conflicting_number"
    unsupported_recommendation = "unsupported_recommendation"
    missing_primary_source = "missing_primary_source"
    irrelevant_source = "irrelevant_source"
    duplicate_source = "duplicate_source"
    marketing_bias = "marketing_bias"
    absolute_language = "absolute_language"
    source_instruction_leakage = "source_instruction_leakage"


class CheckType(str, Enum):
    artifact_exists = "artifact_exists"
    artifact_valid_json = "artifact_valid_json"
    artifact_nonempty = "artifact_nonempty"
    must_mention = "must_mention"
    must_not_mention = "must_not_mention"
    entity_coverage = "entity_coverage"
    numeric_support = "numeric_support"
    date_support = "date_support"
    citation_support = "citation_support"
    prompt_injection_resistance = "prompt_injection_resistance"
    stale_source_warning = "stale_source_warning"
    contradiction_handling = "contradiction_handling"
    uncertainty_handling = "uncertainty_handling"
    source_traceability = "source_traceability"
    confidence_calibration = "confidence_calibration"
    report_completeness = "report_completeness"


class CheckSeverity(str, Enum):
    info = "info"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class BenchmarkSource(LabModel):
    source_id: str
    url: str
    local_path: str
    title: str = ""
    source_type: str = "md"
    trust_level: str = "untrusted"
    published_at: str | None = None
    updated_at: str | None = None
    content_hash: str | None = None
    expected_role: str = "evidence"
    warnings: list[str] = Field(default_factory=list)


class ExpectedResearchOutput(LabModel):
    must_mention: list[str] = Field(default_factory=list)
    must_not_mention: list[str] = Field(default_factory=list)
    expected_entities: list[str] = Field(default_factory=list)
    expected_numbers: list[str] = Field(default_factory=list)
    expected_dates: list[str] = Field(default_factory=list)
    expected_claims: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    required_uncertainty_phrases: list[str] = Field(default_factory=list)
    required_warnings: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(
        default_factory=lambda: ["plan.md", "notes.md", "sources.json", "report.md"]
    )
    optional_artifacts: list[str] = Field(default_factory=list)
    expected_citation_sources: list[str] = Field(default_factory=list)
    expected_missing_evidence_warnings: list[str] = Field(default_factory=list)
    expected_confidence_max: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_confidence_min: float | None = Field(default=None, ge=0.0, le=1.0)


class BenchmarkTrap(LabModel):
    trap_id: str
    trap_type: TrapType
    description: str
    source_id: str | None = None
    expected_detection: str = ""
    severity: CheckSeverity = CheckSeverity.high


DEFAULT_WEIGHTS: dict[str, float] = {
    "artifact_integrity": 0.12,
    "answer_relevance": 0.12,
    "expected_content_coverage": 0.14,
    "source_traceability": 0.12,
    "citation_support": 0.10,
    "numeric_accuracy": 0.10,
    "temporal_handling": 0.08,
    "contradiction_handling": 0.08,
    "adversarial_resistance": 0.12,
    "uncertainty_handling": 0.07,
    "overclaiming_control": 0.05,
}


class ScoringProfile(LabModel):
    profile_id: str = "default"
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    minimum_passing_score: float = Field(default=0.75, ge=0.0, le=1.0)
    fail_on_critical_trap_missed: bool = True
    strict_citations: bool = False
    strict_temporal: bool = True
    strict_numeric: bool = True
    strict_artifact_validation: bool = True
    sensitive_domain_mode: bool = False


class BenchmarkCase(LabModel):
    case_id: str
    title: str
    description: str = ""
    category: BenchmarkCategory
    question: str
    urls: list[str] = Field(default_factory=list)
    local_sources: list[BenchmarkSource] = Field(default_factory=list)
    settings_overrides: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    difficulty: BenchmarkDifficulty = BenchmarkDifficulty.moderate
    expected: ExpectedResearchOutput = Field(default_factory=ExpectedResearchOutput)
    traps: list[BenchmarkTrap] = Field(default_factory=list)
    scoring_profile: ScoringProfile = Field(default_factory=ScoringProfile)
    created_at: str = Field(default_factory=now_iso_utc)
    updated_at: str = Field(default_factory=now_iso_utc)
    case_dir: str | None = None


class BenchmarkRunRequest(LabModel):
    case_ids: list[str] = Field(default_factory=list)
    categories: list[BenchmarkCategory] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    run_all: bool = False
    dry_run: bool = False
    use_mock_agent: bool = True
    use_offline_fetcher: bool = True
    max_cases: int | None = Field(default=None, ge=1)
    settings_overrides: dict[str, Any] = Field(default_factory=dict)
    output_dir: str | None = None


class CheckResult(LabModel):
    check_id: str
    check_type: CheckType
    name: str
    passed: bool
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    severity: CheckSeverity = CheckSeverity.medium
    message: str = ""
    expected: Any = None
    actual: Any = None
    affected_artifacts: list[str] = Field(default_factory=list)
    affected_sources: list[str] = Field(default_factory=list)
    recommendation: str = ""


class BenchmarkCaseResult(LabModel):
    case_id: str
    title: str
    status: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    passed: bool = False
    duration_seconds: float = 0.0
    thread_id: str = ""
    run_dir: str = ""
    check_results: list[CheckResult] = Field(default_factory=list)
    missed_traps: list[str] = Field(default_factory=list)
    detected_traps: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BenchmarkRunResult(LabModel):
    run_id: str
    started_at: str
    completed_at: str | None = None
    status: str = "running"
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    errored_cases: int = 0
    skipped_cases: int = 0
    average_score: float = 0.0
    case_results: list[BenchmarkCaseResult] = Field(default_factory=list)
    report_artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RegressionSuite(LabModel):
    suite_id: str
    title: str
    description: str = ""
    case_ids: list[str] = Field(default_factory=list)
    categories: list[BenchmarkCategory] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    scoring_profile: ScoringProfile = Field(default_factory=ScoringProfile)
    enabled: bool = True


class RegressionSuiteResult(LabModel):
    suite_id: str
    run_id: str
    status: str
    started_at: str
    completed_at: str | None = None
    case_results: list[BenchmarkCaseResult] = Field(default_factory=list)
    pass_rate: float = 0.0
    average_score: float = 0.0
    regressions: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RegressionComparison(LabModel):
    baseline_run_id: str
    current_run_id: str
    compared_at: str = Field(default_factory=now_iso_utc)
    score_delta: float = 0.0
    pass_rate_delta: float = 0.0
    newly_failed_cases: list[str] = Field(default_factory=list)
    newly_passed_cases: list[str] = Field(default_factory=list)
    changed_cases: list[str] = Field(default_factory=list)
    artifact_diffs: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class EvaluationLabSummary(LabModel):
    run_id: str
    total_cases: int
    pass_rate: float
    average_score: float
    highest_risk_failures: list[str] = Field(default_factory=list)
    most_common_failures: list[str] = Field(default_factory=list)
    missed_critical_traps: list[str] = Field(default_factory=list)
    artifact_quality_summary: str = ""
    recommended_fixes: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=now_iso_utc)


class ScoringDimension(LabModel):
    dimension: str
    score: float = Field(ge=0.0, le=1.0)
    weighted_score: float = Field(ge=0.0)
    severity: CheckSeverity = CheckSeverity.info
    reasons: list[str] = Field(default_factory=list)
    suggested_fix: str = ""


class ScoreReport(LabModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    minimum_passing_score: float = Field(ge=0.0, le=1.0)
    dimensions: list[ScoringDimension] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)


class QualityGateStatus(str, Enum):
    passed = "passed"
    failed = "failed"
    warning = "warning"
    skipped = "skipped"
    errored = "errored"


class RegressionFindingType(str, Enum):
    newly_failed_case = "newly_failed_case"
    score_drop = "score_drop"
    threshold_failure = "threshold_failure"
    newly_missed_trap = "newly_missed_trap"
    new_critical_warning = "new_critical_warning"
    new_artifact_failure = "new_artifact_failure"
    new_prompt_injection_failure = "new_prompt_injection_failure"
    new_numeric_failure = "new_numeric_failure"
    new_temporal_failure = "new_temporal_failure"
    new_citation_failure = "new_citation_failure"
    warning_growth = "warning_growth"
    output_forbidden_phrase_added = "output_forbidden_phrase_added"
    required_phrase_removed = "required_phrase_removed"


class QualityGateProfile(LabModel):
    gate_id: str
    name: str
    description: str = ""
    enabled: bool = True
    categories: list[BenchmarkCategory] = Field(default_factory=list)
    case_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    excluded_case_ids: list[str] = Field(default_factory=list)
    scoring_profile_id: str = "default"
    minimum_average_score: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_case_score: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    max_failed_cases: int | None = Field(default=None, ge=0)
    max_critical_failures: int | None = Field(default=None, ge=0)
    max_high_severity_failures: int | None = Field(default=None, ge=0)
    max_missed_critical_traps: int | None = Field(default=None, ge=0)
    max_new_regressions: int | None = Field(default=None, ge=0)
    max_warning_count: int | None = Field(default=None, ge=0)
    max_warning_growth_ratio: float | None = Field(default=None, ge=0.0)
    max_serious_warnings: int | None = Field(default=None, ge=0)
    require_no_prompt_injection_failures: bool = False
    require_no_path_safety_failures: bool = False
    require_no_artifact_integrity_failures: bool = False
    require_no_numeric_regressions: bool = False
    require_no_temporal_regressions: bool = False
    require_no_citation_regressions: bool = False
    allow_mock_agent: bool = True
    use_offline_fetcher: bool = True
    compare_against_baseline: bool = False
    baseline_id: str | None = None
    fail_on_missing_baseline: bool = False
    fail_on_invalid_cases: bool = True
    fail_on_flaky_cases: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityGateRunRequest(LabModel):
    gate_id: str = "smoke"
    case_ids: list[str] = Field(default_factory=list)
    categories: list[BenchmarkCategory] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    run_all: bool = False
    baseline_id: str | None = None
    compare_against_baseline: bool | None = None
    dry_run: bool = False
    use_mock_agent: bool | None = None
    use_offline_fetcher: bool | None = None
    settings_overrides: dict[str, Any] = Field(default_factory=dict)
    output_dir: str | None = None
    update_baseline: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class QualityGateThresholdResult(LabModel):
    threshold_id: str
    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    severity: CheckSeverity = CheckSeverity.high
    message: str = ""
    affected_cases: list[str] = Field(default_factory=list)
    recommendation: str = ""


class BaselineCaseSnapshot(LabModel):
    case_id: str
    status: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    check_results_summary: dict[str, int] = Field(default_factory=dict)
    missed_traps: list[str] = Field(default_factory=list)
    detected_traps: list[str] = Field(default_factory=list)
    warning_count: int = 0
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    report_fingerprint: str = ""


class BenchmarkBaseline(LabModel):
    baseline_id: str
    name: str
    description: str = ""
    created_at: str = Field(default_factory=now_iso_utc)
    created_from_run_id: str
    gate_id: str | None = None
    git_commit: str | None = None
    case_results: list[BaselineCaseSnapshot] = Field(default_factory=list)
    suite_score: float = Field(default=0.0, ge=0.0, le=1.0)
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    average_score: float = Field(default=0.0, ge=0.0, le=1.0)
    case_scores: dict[str, float] = Field(default_factory=dict)
    check_fingerprints: dict[str, str] = Field(default_factory=dict)
    warning_fingerprint: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegressionFinding(LabModel):
    finding_id: str
    case_id: str = ""
    category: str = "regression"
    severity: CheckSeverity = CheckSeverity.high
    type: RegressionFindingType
    message: str
    baseline_value: Any = None
    current_value: Any = None
    score_delta: float = 0.0
    affected_checks: list[str] = Field(default_factory=list)
    recommendation: str = ""


class ImprovementFinding(LabModel):
    finding_id: str
    case_id: str = ""
    type: str
    message: str
    baseline_value: Any = None
    current_value: Any = None
    score_delta: float = 0.0
    affected_checks: list[str] = Field(default_factory=list)


class FlakyCaseSignal(LabModel):
    case_id: str
    signal_type: str
    severity: CheckSeverity = CheckSeverity.info
    message: str
    recent_statuses: list[str] = Field(default_factory=list)
    recent_scores: list[float] = Field(default_factory=list)
    recommendation: str = ""


class WarningAudit(LabModel):
    total_warnings: int = 0
    warnings_by_category: dict[str, int] = Field(default_factory=dict)
    warnings_by_file: dict[str, int] = Field(default_factory=dict)
    warnings_by_origin: dict[str, int] = Field(default_factory=dict)
    new_warning_count: int = 0
    repeated_warning_count: int = 0
    ignored_warning_count: int = 0
    serious_warning_count: int = 0
    budget_exceeded: bool = False
    recommendations: list[str] = Field(default_factory=list)


class BenchmarkCoverageSummary(LabModel):
    total_cases: int = 0
    categories_covered: list[str] = Field(default_factory=list)
    categories_missing: list[str] = Field(default_factory=list)
    trap_types_covered: list[str] = Field(default_factory=list)
    trap_types_missing: list[str] = Field(default_factory=list)
    check_types_covered: list[str] = Field(default_factory=list)
    check_types_missing: list[str] = Field(default_factory=list)
    difficulty_distribution: dict[str, int] = Field(default_factory=dict)
    source_type_distribution: dict[str, int] = Field(default_factory=dict)
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendations: list[str] = Field(default_factory=list)


class GateTriageSummary(LabModel):
    highest_priority_failures: list[str] = Field(default_factory=list)
    likely_root_causes: list[str] = Field(default_factory=list)
    suggested_engineering_tasks: list[str] = Field(default_factory=list)
    cases_to_inspect_first: list[str] = Field(default_factory=list)
    quick_wins: list[str] = Field(default_factory=list)
    risky_regressions: list[str] = Field(default_factory=list)


class QualityGateRunResult(LabModel):
    gate_run_id: str
    gate_id: str
    status: QualityGateStatus
    started_at: str
    completed_at: str | None = None
    benchmark_run_id: str = ""
    baseline_id: str | None = None
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    errored_cases: int = 0
    skipped_cases: int = 0
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    average_score: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_case_score: float = Field(default=0.0, ge=0.0, le=1.0)
    failed_thresholds: list[QualityGateThresholdResult] = Field(default_factory=list)
    passed_thresholds: list[QualityGateThresholdResult] = Field(default_factory=list)
    regressions: list[RegressionFinding] = Field(default_factory=list)
    improvements: list[ImprovementFinding] = Field(default_factory=list)
    flaky_cases: list[FlakyCaseSignal] = Field(default_factory=list)
    warning_audit: WarningAudit = Field(default_factory=WarningAudit)
    coverage_summary: BenchmarkCoverageSummary = Field(default_factory=BenchmarkCoverageSummary)
    triage_summary: GateTriageSummary = Field(default_factory=GateTriageSummary)
    report_artifacts: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
