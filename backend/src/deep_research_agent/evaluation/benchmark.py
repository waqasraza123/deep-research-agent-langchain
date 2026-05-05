from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import BenchmarkCase


def default_benchmark_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "benchmarks"


def list_benchmark_cases(benchmark_dir: Path | None = None) -> list[BenchmarkCase]:
    cases_dir = (benchmark_dir or default_benchmark_dir()) / "cases"
    cases: list[BenchmarkCase] = []
    if not cases_dir.exists():
        return cases
    for path in sorted(cases_dir.glob("*.json")):
        cases.append(load_benchmark_case(path))
    return cases


def load_benchmark_case(path: Path) -> BenchmarkCase:
    raw = json.loads(path.read_text(encoding="utf-8"))
    validate = getattr(BenchmarkCase, "model_validate", None)
    if callable(validate):
        return validate(raw)
    return BenchmarkCase.parse_obj(raw)


def case_to_run_artifacts(case: BenchmarkCase, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.md").write_text(
        "# Benchmark Plan\n\n- Offline deterministic benchmark case.\n", encoding="utf-8"
    )
    (run_dir / "notes.md").write_text(case.notes or "# Notes\n\n", encoding="utf-8")
    (run_dir / "report.md").write_text(case.report or "# Report\n\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "thread_id": case.case_id,
                "question": case.question,
                "urls": case.urls,
                "input_snapshot": {"question": case.question, "urls": case.urls},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    sources: list[dict[str, Any]] = []
    sources_dir = run_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    for idx, document in enumerate(case.mocked_source_documents, start=1):
        source_id = str(document.get("source_id") or f"S{idx}")
        text = str(document.get("text") or "")
        txt_name = f"{source_id.lower()}.txt"
        url = document.get("url") or (case.urls[idx - 1] if idx - 1 < len(case.urls) else "")
        (sources_dir / txt_name).write_text(text, encoding="utf-8")
        sources.append(
            {
                "source_id": source_id,
                "url": url,
                "title": document.get("title") or f"Benchmark source {source_id}",
                "ok": bool(document.get("ok", True)),
                "fetched_at": document.get("fetched_at"),
                "local_path": f"runs/{case.case_id}/sources/{txt_name}",
                "word_count": len(text.split()),
                "char_count": len(text),
                "summary": document.get("summary", ""),
            }
        )
    if not sources and case.urls:
        for idx, url in enumerate(case.urls, start=1):
            sources.append({"source_id": f"S{idx}", "url": url, "ok": False})
    (run_dir / "sources.json").write_text(
        json.dumps(sources, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
