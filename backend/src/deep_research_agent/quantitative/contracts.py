from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from deep_research_agent.source_identity import WarningSeverity


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def model_to_plain(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


NumericKind = Literal[
    "number",
    "percentage",
    "currency",
    "range",
    "ratio",
    "benchmark",
    "date",
    "version",
]

class NumericValue(BaseModel):
    raw_text: str
    normalized_value: float | None = None
    unit: str | None = None
    currency: str | None = None
    percentage: bool = False
    magnitude: str | None = None
    source_id: str | None = None
    source_url: str | None = None
    context: str = ""
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    kind: NumericKind = "number"
    range_start: float | None = None
    range_end: float | None = None
    ratio_left: float | None = None
    ratio_right: float | None = None
    metric_name: str | None = None
    start_char: int | None = None
    end_char: int | None = None


class NumericClaim(BaseModel):
    claim_id: str
    text: str
    origin: Literal["source", "report", "notes"] = "source"
    origin_ref: str | None = None
    source_id: str | None = None
    source_url: str | None = None
    metric_name: str | None = None
    values: list[NumericValue] = Field(default_factory=list)
    comparative_operator: str | None = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    support_status: str = "unchecked"
    warnings: list[str] = Field(default_factory=list)


class MetricDefinition(BaseModel):
    name: str
    normalized_name: str
    unit: str | None = None
    currency: str | None = None
    category: str = "general"
    aliases: list[str] = Field(default_factory=list)
    higher_is_better: bool | None = None
    source_ids: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ColumnProfile(BaseModel):
    name: str
    index: int
    empty_count: int = 0
    numeric_count: int = 0
    date_count: int = 0
    distinct_count: int = 0
    detected_unit: str | None = None
    inferred_type: Literal["numeric", "date", "categorical", "identifier", "empty", "mixed"]
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    sample_values: list[str] = Field(default_factory=list)


class TableProfile(BaseModel):
    table_id: str
    source_id: str | None = None
    source_url: str | None = None
    caption: str | None = None
    column_names: list[str] = Field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    empty_values: int = 0
    numeric_columns: list[str] = Field(default_factory=list)
    date_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    detected_units: dict[str, str] = Field(default_factory=dict)
    column_profiles: list[ColumnProfile] = Field(default_factory=list)
    duplicate_rows: int = 0
    possible_identifier_columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CSVProfile(BaseModel):
    csv_id: str
    source_id: str | None = None
    source_url: str | None = None
    path: str | None = None
    delimiter: str = ","
    column_names: list[str] = Field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    empty_values: int = 0
    numeric_columns: list[str] = Field(default_factory=list)
    date_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    detected_units: dict[str, str] = Field(default_factory=dict)
    column_profiles: list[ColumnProfile] = Field(default_factory=list)
    duplicate_rows: int = 0
    possible_identifier_columns: list[str] = Field(default_factory=list)
    malformed_rows: int = 0
    warnings: list[str] = Field(default_factory=list)


class ComparisonValue(BaseModel):
    entity: str
    value: NumericValue
    source_id: str | None = None
    source_url: str | None = None


class QuantitativeComparison(BaseModel):
    comparison_id: str
    metric_name: str
    unit: str | None = None
    currency: str | None = None
    values: list[ComparisonValue] = Field(default_factory=list)
    winner: str | None = None
    direction: str | None = None
    comparable: bool = True
    warnings: list[str] = Field(default_factory=list)


class QuantitativeConsistencyCheck(BaseModel):
    check_id: str
    claim_id: str | None = None
    status: Literal["pass", "warning", "fail", "unchecked"] = "unchecked"
    message: str
    expected_value: NumericValue | None = None
    observed_value: NumericValue | None = None
    source_ids: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CalculationResult(BaseModel):
    calculation_id: str
    operation: str
    inputs: list[NumericValue] = Field(default_factory=list)
    result_value: float | None = None
    unit: str | None = None
    currency: str | None = None
    expression: str = ""
    valid: bool = True
    warnings: list[str] = Field(default_factory=list)


class QuantitativeWarning(BaseModel):
    subsystem: str = "quantitative"
    warning_id: str
    code: str
    message: str
    severity: WarningSeverity = WarningSeverity.MEDIUM
    affected_artifacts: list[str] = Field(default_factory=list)
    affected_sources: list[str] = Field(default_factory=list)
    recommended_action: str = "Review numeric evidence and source context."
    source_id: str | None = None
    claim_id: str | None = None
    context: str = ""


class QuantitativeEvidence(BaseModel):
    evidence_id: str
    thread_id: str | None = None
    generated_at: datetime = Field(default_factory=now_utc)
    numeric_values: list[NumericValue] = Field(default_factory=list)
    numeric_claims: list[NumericClaim] = Field(default_factory=list)
    metrics: list[MetricDefinition] = Field(default_factory=list)
    table_profiles: list[TableProfile] = Field(default_factory=list)
    csv_profiles: list[CSVProfile] = Field(default_factory=list)
    comparisons: list[QuantitativeComparison] = Field(default_factory=list)
    consistency_checks: list[QuantitativeConsistencyCheck] = Field(default_factory=list)
    calculations: list[CalculationResult] = Field(default_factory=list)
    warnings: list[QuantitativeWarning] = Field(default_factory=list)


class QuantitativeSummary(BaseModel):
    thread_id: str
    generated_at: datetime = Field(default_factory=now_utc)
    value_count: int = 0
    claim_count: int = 0
    metric_count: int = 0
    table_count: int = 0
    csv_count: int = 0
    comparison_count: int = 0
    warning_count: int = 0
    consistency_failures: int = 0
    evidence: QuantitativeEvidence | None = None
    warnings: list[QuantitativeWarning] = Field(default_factory=list)
