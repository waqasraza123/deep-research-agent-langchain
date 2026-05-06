from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from ..settings import REPO_ROOT
from .contracts import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkDifficulty,
    BenchmarkSource,
    ExpectedResearchOutput,
    model_to_plain,
)
from .errors import CaseLoadError, UnsafeBenchmarkPathError


def default_cases_root() -> Path:
    return REPO_ROOT / "backend" / "benchmarks" / "cases"


def _model_validate(model_cls, data):
    validate = getattr(model_cls, "model_validate", None)
    if callable(validate):
        return validate(data)
    return model_cls.parse_obj(data)


def _read_json(path: Path) -> dict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CaseLoadError(f"Invalid JSON in {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise CaseLoadError(f"{path} must contain a JSON object")
    return loaded


def _ensure_child(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise UnsafeBenchmarkPathError(f"Path escapes benchmark case directory: {path}")
    return resolved


def _validate_relative_local_path(local_path: str) -> None:
    p = Path(local_path)
    if p.is_absolute() or ".." in p.parts or "\\" in local_path:
        raise UnsafeBenchmarkPathError(f"Unsafe local source path: {local_path}")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 256), b""):
            h.update(chunk)
    return h.hexdigest()


def load_case(case_dir: Path | str) -> BenchmarkCase:
    case_dir = Path(case_dir)
    if not case_dir.exists() or not case_dir.is_dir():
        raise CaseLoadError(f"Benchmark case directory does not exist: {case_dir}")
    case_json = case_dir / "case.json"
    expected_json = case_dir / "expected.json"
    if not case_json.exists():
        raise CaseLoadError(f"Missing case.json in {case_dir}")
    if not expected_json.exists():
        raise CaseLoadError(f"Missing expected.json in {case_dir}")

    raw_case = _read_json(case_json)
    raw_expected = _read_json(expected_json)
    try:
        expected = _model_validate(ExpectedResearchOutput, raw_expected)
    except ValidationError as exc:
        raise CaseLoadError(f"Invalid expected.json in {case_dir}: {exc}") from exc

    raw_case["expected"] = model_to_plain(expected)
    local_sources: list[dict] = []
    for idx, raw_source in enumerate(raw_case.get("local_sources") or [], start=1):
        if not isinstance(raw_source, dict):
            raise CaseLoadError(f"local_sources[{idx}] must be an object")
        local_path = str(raw_source.get("local_path") or "")
        _validate_relative_local_path(local_path)
        source_path = _ensure_child(case_dir / local_path, case_dir)
        if not source_path.exists() or source_path.is_dir():
            raise CaseLoadError(f"Missing local source file: {local_path}")
        source = dict(raw_source)
        source.setdefault("source_id", f"source_{idx}")
        source.setdefault("url", f"benchmark://{raw_case.get('case_id')}/{source['source_id']}")
        source.setdefault("title", source["source_id"])
        source.setdefault("source_type", source_path.suffix.lstrip(".") or "txt")
        source["content_hash"] = _hash_file(source_path)
        local_sources.append(source)
    raw_case["local_sources"] = local_sources
    raw_case["case_dir"] = str(case_dir.resolve())

    try:
        case = _model_validate(BenchmarkCase, raw_case)
    except ValidationError as exc:
        raise CaseLoadError(f"Invalid case.json in {case_dir}: {exc}") from exc
    validate_case(case)
    return case


def load_cases(root_dir: Path | str | None = None) -> list[BenchmarkCase]:
    root = Path(root_dir) if root_dir is not None else default_cases_root()
    if not root.exists():
        return []
    cases: list[BenchmarkCase] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        cases.append(load_case(case_dir))
    return cases


def list_cases(
    root_dir: Path | str | None = None,
    *,
    case_ids: Iterable[str] | None = None,
    categories: Iterable[str | BenchmarkCategory] | None = None,
    tags: Iterable[str] | None = None,
    difficulty: str | BenchmarkDifficulty | None = None,
) -> list[BenchmarkCase]:
    cases = load_cases(root_dir)
    allowed_ids = {str(item) for item in case_ids or []}
    allowed_categories = {str(getattr(item, "value", item)) for item in categories or []}
    required_tags = {str(item) for item in tags or []}
    wanted_difficulty = str(getattr(difficulty, "value", difficulty)) if difficulty else None
    if allowed_ids:
        cases = [case for case in cases if case.case_id in allowed_ids]
    if allowed_categories:
        cases = [case for case in cases if case.category.value in allowed_categories]
    if required_tags:
        cases = [case for case in cases if required_tags.intersection(set(case.tags))]
    if wanted_difficulty:
        cases = [case for case in cases if case.difficulty.value == wanted_difficulty]
    return cases


def validate_case(case: BenchmarkCase) -> list[str]:
    warnings: list[str] = []
    if not case.question.strip():
        raise CaseLoadError(f"{case.case_id}: question is required")
    if len({source.source_id for source in case.local_sources}) != len(case.local_sources):
        raise CaseLoadError(f"{case.case_id}: duplicate source_id in local_sources")
    required = set(case.expected.required_artifacts)
    for artifact in ("plan.md", "notes.md", "sources.json", "report.md"):
        if artifact not in required:
            raise CaseLoadError(f"{case.case_id}: required_artifacts must include {artifact}")
    if not case.urls:
        warnings.append(f"{case.case_id}: no benchmark URLs declared")
    url_map = build_source_url_map(case)
    missing_urls = [url for url in case.urls if url not in url_map]
    if missing_urls:
        warnings.append(f"{case.case_id}: urls without local source mapping: {missing_urls}")
    return warnings


def build_source_url_map(case: BenchmarkCase) -> dict[str, BenchmarkSource]:
    out: dict[str, BenchmarkSource] = {}
    for source in case.local_sources:
        out[source.url] = source
        out[f"benchmark://{case.case_id}/{source.source_id}"] = source
    return out


def compute_case_fingerprint(case: BenchmarkCase) -> str:
    payload = model_to_plain(case)
    payload.pop("created_at", None)
    payload.pop("updated_at", None)
    payload.pop("case_dir", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
