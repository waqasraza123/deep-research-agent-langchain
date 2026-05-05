from __future__ import annotations

import json

from deep_research_agent.artifacts import ensure_thread_dir
from deep_research_agent.quantitative import (
    calculate,
    extract_numeric_claims,
    extract_numeric_values,
    profile_csv_text,
    profile_rows,
    rebuild_quantitative_artifacts,
)
from deep_research_agent.quantitative.comparison import build_comparisons
from deep_research_agent.quantitative.consistency_checker import check_report_claims


def test_number_extraction_core_types():
    text = (
        "Acme has 99.9% uptime, $20 per user per month, a 128k context window, "
        "3.2 seconds latency, and 15 requests per second."
    )
    values = extract_numeric_values(text, source_id="S1")
    raw = [value.raw_text for value in values]

    assert "99.9%" in raw
    assert any(value.currency == "USD" and value.normalized_value == 20 for value in values)
    assert any(
        value.raw_text.startswith("128k") and value.normalized_value == 128000
        for value in values
    )
    assert any(value.unit == "seconds" and value.normalized_value == 3.2 for value in values)
    assert any(
        value.unit == "requests per second" and value.normalized_value == 15
        for value in values
    )


def test_version_is_not_metric_value():
    values = extract_numeric_values("The SDK v0.2.14 scored 91 points on the benchmark.")
    version = [value for value in values if value.raw_text == "v0.2.14"]
    benchmark = [value for value in values if value.raw_text.startswith("91")]

    assert version
    assert version[0].kind == "version"
    assert benchmark
    assert benchmark[0].kind == "benchmark"


def test_metric_name_detection_near_values():
    values = extract_numeric_values("The service reports 3.2 seconds latency and 99.9% uptime.")

    assert any(value.metric_name == "latency" for value in values)
    assert any(value.metric_name == "uptime" for value in values)


def test_csv_profiling_detects_column_types_and_duplicates():
    profile = profile_csv_text(
        "id,name,latency_ms,release_date\n"
        "1,Alpha,120,2025-01-01\n"
        "2,Beta,90,2025-02-01\n"
        "2,Beta,90,2025-02-01\n"
    )

    assert profile.row_count == 3
    assert "latency_ms" in profile.numeric_columns
    assert "release_date" in profile.date_columns
    assert profile.duplicate_rows == 1
    assert "id" in profile.possible_identifier_columns


def test_table_profiling_detects_units_and_empty_values():
    profile = profile_rows(
        [
            ["Framework", "Stars", "Latency"],
            ["Alpha", "100 stars", "120 ms"],
            ["Beta", "150 stars", ""],
        ],
        table_id="tbl1",
    )

    assert profile.row_count == 2
    assert "Stars" in profile.numeric_columns
    assert profile.empty_values == 1
    assert profile.detected_units["Latency"] == "ms"


def test_comparison_building_and_safe_calculations():
    values = extract_numeric_values(
        "Alpha latency is 120 ms. Beta latency is 90 ms. Alpha costs $20. Beta costs $15."
    )
    comparisons = build_comparisons(values)
    latency = next(item for item in comparisons if item.metric_name == "latency")
    calc = calculate("difference", [latency.values[0].value, latency.values[1].value])

    assert latency.comparable is True
    assert latency.winner == "Beta"
    assert calc.valid is True
    assert calc.result_value is not None


def test_consistency_checker_flags_unsupported_report_number():
    source_values = extract_numeric_values("Source says uptime was 99.9%.", source_id="S1")
    report_claims = extract_numeric_claims("The report says uptime was 98.0%.", origin="report")
    checks = check_report_claims(report_claims, source_values)

    assert any(check.status == "warning" for check in checks)


def test_artifact_writer_and_api_rebuild_route(client, test_runs_dir):
    tid = "quant-api-test"
    td = ensure_thread_dir(test_runs_dir, tid)
    source_dir = td / "sources"
    source_dir.mkdir()
    (source_dir / "source.txt").write_text(
        "Alpha latency is 120 ms. Beta latency is 90 ms. Uptime is 99.9%.",
        encoding="utf-8",
    )
    (td / "sources.json").write_text(
        json.dumps(
            [
                {
                    "url": "https://example.com/a",
                    "ok": True,
                    "source_id": "S1",
                    "local_path": f"runs/{tid}/sources/source.txt",
                }
            ]
        ),
        encoding="utf-8",
    )
    (td / "notes.md").write_text("Notes mention 99.9% uptime.\n", encoding="utf-8")
    (td / "report.md").write_text("Report says Alpha latency is 120 ms.\n", encoding="utf-8")

    summary = rebuild_quantitative_artifacts(td, thread_id=tid)
    assert summary.value_count >= 3
    assert (td / "quantitative_profile.json").exists()
    assert (td / "numeric_claims.md").exists()

    response = client.post(f"/runs/{tid}/quantitative/rebuild")
    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == tid
    assert any(a["path"] == "quantitative_profile.md" for a in body["artifacts"])

    claims = client.get(f"/runs/{tid}/numeric-claims")
    assert claims.status_code == 200
    assert claims.json()["claims"]
