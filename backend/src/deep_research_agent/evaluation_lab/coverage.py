from __future__ import annotations

from collections import Counter
from pathlib import Path

from .case_loader import default_cases_root, load_cases
from .contracts import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkCoverageSummary,
    CheckType,
    TrapType,
)
from .governance_report import render_coverage, write_json


def analyze_coverage(cases: list[BenchmarkCase]) -> BenchmarkCoverageSummary:
    categories = sorted({case.category.value for case in cases})
    traps = sorted({trap.trap_type.value for case in cases for trap in case.traps})
    check_types = sorted(item.value for item in CheckType)
    missing_categories = sorted(set(item.value for item in BenchmarkCategory) - set(categories))
    missing_traps = sorted(set(item.value for item in TrapType) - set(traps))
    difficulty_counts = Counter(case.difficulty.value for case in cases)
    source_type_counts = Counter(
        source.source_type or "unknown" for case in cases for source in case.local_sources
    )
    category_score = len(categories) / max(1, len(BenchmarkCategory))
    trap_score = len(traps) / max(1, len(TrapType))
    difficulty_score = len(difficulty_counts) / 4
    source_score = min(1.0, len(source_type_counts) / 4)
    score = round(
        (category_score * 0.40)
        + (trap_score * 0.35)
        + (difficulty_score * 0.15)
        + (source_score * 0.10),
        4,
    )
    recommendations = []
    if missing_categories:
        recommendations.append(
            "Add benchmark cases for missing categories: " + ", ".join(missing_categories[:8])
        )
    if missing_traps:
        recommendations.append("Add traps for: " + ", ".join(missing_traps[:8]))
    if len(difficulty_counts) < 4:
        recommendations.append("Add cases across easy, moderate, hard, and adversarial difficulty.")
    return BenchmarkCoverageSummary(
        total_cases=len(cases),
        categories_covered=categories,
        categories_missing=missing_categories,
        trap_types_covered=traps,
        trap_types_missing=missing_traps,
        check_types_covered=check_types,
        check_types_missing=[],
        difficulty_distribution=dict(sorted(difficulty_counts.items())),
        source_type_distribution=dict(sorted(source_type_counts.items())),
        coverage_score=score,
        recommendations=recommendations,
    )


def coverage_for_cases_root(cases_root: Path | None = None) -> BenchmarkCoverageSummary:
    return analyze_coverage(load_cases(cases_root or default_cases_root()))


def write_coverage_report(output_dir: Path, summary: BenchmarkCoverageSummary) -> list[str]:
    write_json(output_dir / "benchmark_coverage.json", summary)
    (output_dir / "benchmark_coverage.md").write_text(render_coverage(summary), encoding="utf-8")
    return ["benchmark_coverage.json", "benchmark_coverage.md"]
