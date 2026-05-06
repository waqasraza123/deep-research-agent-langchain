from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from ..artifacts import ensure_required_artifacts
from ..settings import REPO_ROOT, Settings
from .adversarial_checks import run_adversarial_checks
from .artifact_checks import run_artifact_checks
from .case_loader import (
    compute_case_fingerprint,
    default_cases_root,
    list_cases,
    load_cases,
    validate_case,
)
from .citation_checks import run_citation_checks
from .contracts import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkRunRequest,
    BenchmarkRunResult,
    CheckSeverity,
    EvaluationLabSummary,
    RegressionComparison,
    RegressionSuite,
    RegressionSuiteResult,
    ScoringProfile,
    now_iso_utc,
)
from .errors import BenchmarkRunError, UnsafeBenchmarkPathError
from .expected_outputs import run_expected_output_checks
from .fixture_corpus import FixtureCorpus
from .hallucination_checks import run_hallucination_checks
from .numeric_checks import run_numeric_checks
from .offline_fetcher import OfflineFetcher
from .report_writer import write_case_result, write_comparison, write_run_reports
from .scoring import calculate_score, write_score_report
from .temporal_checks import run_temporal_checks


class EvaluationLabRunner:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()
        self.cases_dir = Path(
            getattr(self.settings, "evaluation_lab_cases_dir", None) or default_cases_root()
        )
        self.runs_dir = Path(
            getattr(self.settings, "evaluation_lab_runs_dir", None)
            or (REPO_ROOT / "backend" / "benchmark_runs")
        )

    def list_cases(
        self,
        *,
        case_ids: list[str] | None = None,
        categories: list[str] | None = None,
        tags: list[str] | None = None,
        difficulty: str | None = None,
    ) -> list[BenchmarkCase]:
        return list_cases(
            self.cases_dir,
            case_ids=case_ids,
            categories=categories,
            tags=tags,
            difficulty=difficulty,
        )

    def get_case(self, case_id: str) -> BenchmarkCase:
        for case in load_cases(self.cases_dir):
            if case.case_id == case_id:
                return case
        raise BenchmarkRunError(f"Unknown benchmark case: {case_id}")

    def validate(self, request: BenchmarkRunRequest | None = None) -> dict[str, Any]:
        cases = self._select_cases(request or BenchmarkRunRequest(run_all=True, dry_run=True))
        return {
            "ok": True,
            "total_cases": len(cases),
            "cases": [
                {
                    "case_id": case.case_id,
                    "title": case.title,
                    "fingerprint": compute_case_fingerprint(case),
                    "warnings": validate_case(case),
                }
                for case in cases
            ],
        }

    def run_case(
        self,
        case: BenchmarkCase,
        *,
        parent_run_dir: Path,
        request: BenchmarkRunRequest | None = None,
    ) -> BenchmarkCaseResult:
        request = request or BenchmarkRunRequest(case_ids=[case.case_id])
        started = time.monotonic()
        case_run_dir = self._safe_child(parent_run_dir, Path("cases") / case.case_id)
        case_run_dir.mkdir(parents=True, exist_ok=True)
        thread_id = f"benchmark-{case.case_id}-{uuid.uuid4().hex[:8]}"
        status = "passed"
        warnings: list[str] = []
        try:
            if request.dry_run:
                validate_case(case)
                result = BenchmarkCaseResult(
                    case_id=case.case_id,
                    title=case.title,
                    status="skipped",
                    score=1.0,
                    passed=True,
                    duration_seconds=round(time.monotonic() - started, 4),
                    thread_id=thread_id,
                    run_dir=str(case_run_dir),
                    warnings=["dry_run: validated case without executing mock agent"],
                )
                write_case_result(case_run_dir, result)
                return result

            if not request.use_offline_fetcher:
                raise BenchmarkRunError(
                    "Evaluation lab requires offline fetcher for benchmark:// URLs"
                )
            self._write_mock_run_artifacts(case, case_run_dir, thread_id, request=request)
            ensure_required_artifacts(case_run_dir.parent, case_run_dir.name)

            checks = []
            checks.extend(
                run_artifact_checks(
                    case,
                    case_run_dir,
                    use_mock_agent=request.use_mock_agent,
                )
            )
            checks.extend(run_expected_output_checks(case, case_run_dir))
            checks.extend(run_hallucination_checks(case, case_run_dir))
            checks.extend(
                run_citation_checks(
                    case,
                    case_run_dir,
                    strict=case.scoring_profile.strict_citations,
                )
            )
            checks.extend(
                run_temporal_checks(
                    case,
                    case_run_dir,
                    strict=case.scoring_profile.strict_temporal,
                )
            )
            checks.extend(
                run_numeric_checks(
                    case,
                    case_run_dir,
                    strict=case.scoring_profile.strict_numeric,
                )
            )
            checks.extend(
                run_adversarial_checks(
                    case,
                    case_run_dir,
                    strict=getattr(self.settings, "evaluation_lab_strict_adversarial_checks", True),
                )
            )
            score = calculate_score(case, checks, profile=self._profile_for_case(case, request))
            write_score_report(case_run_dir, score)
            artifact_paths = sorted(
                str(path.relative_to(case_run_dir)).replace("\\", "/")
                for path in case_run_dir.rglob("*")
                if path.is_file()
            )
            missed_traps, detected_traps = self._trap_status(case, case_run_dir)
            passed = bool(score.passed and not missed_traps)
            status = "passed" if passed else "failed"
            result = BenchmarkCaseResult(
                case_id=case.case_id,
                title=case.title,
                status=status,
                score=score.overall_score,
                passed=passed,
                duration_seconds=round(time.monotonic() - started, 4),
                thread_id=thread_id,
                run_dir=str(case_run_dir),
                check_results=checks,
                missed_traps=missed_traps,
                detected_traps=detected_traps,
                artifact_paths=artifact_paths,
                failure_reasons=score.failure_reasons,
                warnings=warnings,
            )
            write_case_result(case_run_dir, result)
            return result
        except Exception as exc:
            result = BenchmarkCaseResult(
                case_id=case.case_id,
                title=case.title,
                status="errored",
                score=0.0,
                passed=False,
                duration_seconds=round(time.monotonic() - started, 4),
                thread_id=thread_id,
                run_dir=str(case_run_dir),
                failure_reasons=[f"{type(exc).__name__}: {exc}"],
                warnings=warnings,
            )
            write_case_result(case_run_dir, result)
            return result

    def run_cases(
        self,
        request: BenchmarkRunRequest | None = None,
    ) -> BenchmarkRunResult:
        request = request or BenchmarkRunRequest(run_all=True)
        run_id = f"eval-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        output_root = self._resolve_output_dir(request.output_dir)
        run_dir = self._safe_child(output_root, Path(run_id))
        run_dir.mkdir(parents=True, exist_ok=True)
        started_at = now_iso_utc()
        cases = self._select_cases(request)
        max_cases = request.max_cases or getattr(
            self.settings, "evaluation_lab_max_cases_per_run", 25
        )
        cases = cases[: int(max_cases)]
        results = [self.run_case(case, parent_run_dir=run_dir, request=request) for case in cases]
        passed = len([result for result in results if result.passed])
        errored = len([result for result in results if result.status == "errored"])
        skipped = len([result for result in results if result.status == "skipped"])
        failed = len(results) - passed - errored - skipped
        average = sum(result.score for result in results) / len(results) if results else 0.0
        status = "passed" if failed == 0 and errored == 0 else "failed"
        if request.dry_run:
            status = "validated"
        run_result = BenchmarkRunResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=now_iso_utc(),
            status=status,
            total_cases=len(results),
            passed_cases=passed,
            failed_cases=failed,
            errored_cases=errored,
            skipped_cases=skipped,
            average_score=average,
            case_results=results,
            report_artifacts=[
                "benchmark_run.json",
                "benchmark_run.md",
                "evaluation_lab_summary.json",
                "evaluation_lab_summary.md",
            ],
        )
        write_run_reports(run_dir, run_result)
        return run_result

    def run_suite(self, suite: RegressionSuite) -> RegressionSuiteResult:
        request = BenchmarkRunRequest(
            case_ids=suite.case_ids,
            categories=suite.categories,
            tags=suite.tags,
            run_all=not bool(suite.case_ids or suite.categories or suite.tags),
        )
        run = self.run_cases(request)
        total = run.total_cases or 1
        return RegressionSuiteResult(
            suite_id=suite.suite_id,
            run_id=run.run_id,
            status=run.status,
            started_at=run.started_at,
            completed_at=run.completed_at,
            case_results=run.case_results,
            pass_rate=run.passed_cases / total,
            average_score=run.average_score,
        )

    def read_run(self, run_id: str) -> BenchmarkRunResult:
        path = self._safe_child(self.runs_dir, Path(run_id) / "benchmark_run.json")
        if not path.exists():
            raise BenchmarkRunError(f"Unknown evaluation lab run: {run_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        validate = getattr(BenchmarkRunResult, "model_validate", None)
        return validate(data) if callable(validate) else BenchmarkRunResult.parse_obj(data)

    def read_summary(self, run_id: str, *, markdown: bool = False) -> str | EvaluationLabSummary:
        path = self._safe_child(
            self.runs_dir,
            Path(run_id)
            / ("evaluation_lab_summary.md" if markdown else "evaluation_lab_summary.json"),
        )
        if not path.exists():
            raise BenchmarkRunError(f"Unknown evaluation lab summary: {run_id}")
        if markdown:
            return path.read_text(encoding="utf-8")
        data = json.loads(path.read_text(encoding="utf-8"))
        validate = getattr(EvaluationLabSummary, "model_validate", None)
        return validate(data) if callable(validate) else EvaluationLabSummary.parse_obj(data)

    def compare_runs(self, baseline_run_id: str, current_run_id: str) -> RegressionComparison:
        baseline = self.read_run(baseline_run_id)
        current = self.read_run(current_run_id)
        by_base = {case.case_id: case for case in baseline.case_results}
        by_current = {case.case_id: case for case in current.case_results}
        newly_failed = sorted(
            case_id
            for case_id, case in by_current.items()
            if not case.passed and by_base.get(case_id) and by_base[case_id].passed
        )
        newly_passed = sorted(
            case_id
            for case_id, case in by_current.items()
            if case.passed and by_base.get(case_id) and not by_base[case_id].passed
        )
        changed = sorted(
            case_id
            for case_id, case in by_current.items()
            if by_base.get(case_id) and abs(case.score - by_base[case_id].score) >= 0.01
        )
        comparison = RegressionComparison(
            baseline_run_id=baseline.run_id,
            current_run_id=current.run_id,
            score_delta=current.average_score - baseline.average_score,
            pass_rate_delta=self._pass_rate(current) - self._pass_rate(baseline),
            newly_failed_cases=newly_failed,
            newly_passed_cases=newly_passed,
            changed_cases=changed,
            artifact_diffs={
                case_id: {"score_delta": by_current[case_id].score - by_base[case_id].score}
                for case_id in changed
            },
            summary=f"{len(newly_failed)} newly failed, {len(newly_passed)} newly passed.",
        )
        write_comparison(self._safe_child(self.runs_dir, Path(current.run_id)), comparison)
        return comparison

    def _select_cases(self, request: BenchmarkRunRequest) -> list[BenchmarkCase]:
        categories = [
            category.value if hasattr(category, "value") else str(category)
            for category in request.categories
        ]
        cases = self.list_cases(
            case_ids=request.case_ids,
            categories=categories,
            tags=request.tags,
        )
        if not (request.run_all or request.case_ids or request.categories or request.tags):
            cases = cases[:1]
        if request.case_ids and len(cases) != len(set(request.case_ids)):
            found = {case.case_id for case in cases}
            missing = sorted(set(request.case_ids) - found)
            raise BenchmarkRunError(f"Unknown benchmark case IDs: {missing}")
        return cases

    def _write_mock_run_artifacts(
        self,
        case: BenchmarkCase,
        run_dir: Path,
        thread_id: str,
        *,
        request: BenchmarkRunRequest,
    ) -> None:
        corpus = FixtureCorpus([case])
        fetcher = OfflineFetcher(corpus, allow_benchmark_scheme=request.use_offline_fetcher)
        sources = []
        source_texts = []
        sources_dir = run_dir / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        for url in case.urls:
            fetched = fetcher.fetch_document(url, max_chars=250_000)
            source_meta = fetcher.fetch_metadata(url)
            source_id = str(source_meta["benchmark_source_id"])
            filename = f"{source_id}.txt"
            (sources_dir / filename).write_text(fetched.extracted_text, encoding="utf-8")
            sources.append(
                {
                    "source_id": source_id,
                    "url": url,
                    "title": fetched.title or source_id,
                    "ok": True,
                    "content_hash": source_meta.get("content_hash"),
                    "local_path": f"sources/{filename}",
                    "benchmark_case_id": case.case_id,
                    "benchmark_source_id": source_id,
                    "published_at": source_meta.get("published_at"),
                    "updated_at": source_meta.get("updated_at"),
                    "trust_level": source_meta.get("trust_level") or "untrusted",
                    "mock": True,
                }
            )
            source_texts.append((source_id, fetched.title or source_id, fetched.extracted_text))

        report = self._render_mock_report(case, source_texts)
        notes = self._render_mock_notes(case, source_texts)
        (run_dir / "plan.md").write_text(
            "# Benchmark Mock Plan\n\n"
            "> MOCK OUTPUT: deterministic Research Evaluation Lab run.\n\n"
            f"- Question: {case.question}\n"
            "- Load local fixture sources via benchmark:// URLs.\n"
            "- Summarize only supplied source evidence.\n"
            "- Flag source traps, uncertainty, stale sources, numeric conflicts, "
            "and citation limits.\n",
            encoding="utf-8",
        )
        (run_dir / "notes.md").write_text(notes, encoding="utf-8")
        (run_dir / "sources.json").write_text(
            json.dumps(sources, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_dir / "report.md").write_text(report, encoding="utf-8")
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "mock": True,
                    "thread_id": thread_id,
                    "case_id": case.case_id,
                    "offline_fetcher": True,
                    "benchmark_scheme": True,
                    "settings_overrides_redacted": self._redact_settings(
                        request.settings_overrides
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "thread_id": thread_id,
                    "question": case.question,
                    "urls": case.urls,
                    "mock_mode": True,
                    "evaluation_lab": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _render_mock_notes(
        self, case: BenchmarkCase, source_texts: list[tuple[str, str, str]]
    ) -> str:
        lines = [
            "# Benchmark Mock Notes",
            "",
            "> MOCK OUTPUT: notes are deterministic and derived from fixture documents.",
            "",
        ]
        for source_id, title, text in source_texts:
            summary = " ".join(text.split())[:450]
            lines.append(f"- {source_id} ({title}): {summary}")
        if case.traps:
            lines.extend(["", "## Trap Review", ""])
            for trap in case.traps:
                lines.append(
                    f"- {trap.trap_type.value}: {trap.expected_detection or trap.description}"
                )
        return "\n".join(lines).rstrip() + "\n"

    def _render_mock_report(
        self, case: BenchmarkCase, source_texts: list[tuple[str, str, str]]
    ) -> str:
        combined = "\n".join(text for _, _, text in source_texts)
        lines = [
            "# Mock Research Report",
            "",
            "> MOCK OUTPUT: generated offline for the Research Evaluation Lab; use as "
            "infrastructure validation, not production research.",
            "",
            f"Question: {case.question}",
            "",
            "## Findings",
            "",
        ]
        for phrase in case.expected.must_mention[:8]:
            lines.append(f"- Based on provided sources, {phrase}.")
        for claim in case.expected.expected_claims[:8]:
            if claim.lower() not in " ".join(lines).lower():
                lines.append(f"- Source-supported claim: {claim}.")
        if not case.expected.must_mention:
            for source_id, title, text in source_texts:
                lines.append(f"- [{source_id}] {title}: {' '.join(text.split())[:260]}.")
        numbers = sorted(
            set(
                re.findall(
                    r"\b\d+(?:\.\d+)?\s*(?:requests per minute|ms|MB|GB|%)?\b", combined, re.I
                )
            )
        )
        if numbers:
            lines.append(
                f"- Numeric values found in the provided sources include: {', '.join(numbers)}."
            )
        if len(source_texts) > 1:
            lines.append(
                "- The conclusion depends on requirements and source quality; no source "
                "establishes a universal best choice."
            )
        trap_text = " ".join(trap.trap_type.value for trap in case.traps)
        if "prompt_injection" in trap_text:
            lines.append(
                "- Warning: the source contains prompt injection/source instructions; "
                "those instructions are untrusted and should not be followed."
            )
        if "stale_date" in trap_text or case.category.value == "stale_source":
            lines.append(
                "- Warning: source is outdated or archived, so current claims cannot be "
                "confirmed from the provided sources."
            )
        if "conflicting_number" in trap_text or case.category.value == "contradictory_sources":
            lines.append(
                "- Warning: conflicting sources report different values; treat the maximum "
                "upload size as unresolved without additional primary evidence."
            )
        if "missing_primary_source" in trap_text or case.category.value == "missing_primary_source":
            lines.append(
                "- Warning: primary source is missing; official documentation, compliance "
                "review, or legal review is required before adoption."
            )
        if "marketing_bias" in trap_text:
            lines.append(
                "- Warning: the available source is marketing or vendor-controlled and may "
                "be biased."
            )
        for phrase in case.expected.required_uncertainty_phrases[:6]:
            if phrase.lower() not in " ".join(lines).lower():
                lines.append(f"- Uncertainty: {phrase}.")
        lines.extend(["", "## Sources", ""])
        for source_id, title, _ in source_texts:
            lines.append(f"- [{source_id}] {title}")
        lines.extend(["", "## Limitations", ""])
        lines.append(
            "- Based on provided sources only; no external network, live model, OpenAI, "
            "or Ollama was used."
        )
        return "\n".join(lines).rstrip() + "\n"

    def _trap_status(self, case: BenchmarkCase, run_dir: Path) -> tuple[list[str], list[str]]:
        text = ""
        for rel in ("report.md", "notes.md"):
            path = run_dir / rel
            if path.exists():
                text += "\n" + path.read_text(encoding="utf-8", errors="ignore").lower()
        missed = []
        detected = []
        for trap in case.traps:
            markers = [trap.trap_type.value.replace("_", " "), trap.expected_detection.lower()]
            if any(marker and marker in text for marker in markers):
                detected.append(trap.trap_id)
            elif trap.severity in {CheckSeverity.high, CheckSeverity.critical}:
                missed.append(trap.trap_id)
        return missed, detected

    def _profile_for_case(
        self, case: BenchmarkCase, request: BenchmarkRunRequest
    ) -> ScoringProfile:
        profile = case.scoring_profile
        minimum = request.settings_overrides.get(
            "minimum_passing_score",
            getattr(
                self.settings, "evaluation_lab_minimum_passing_score", profile.minimum_passing_score
            ),
        )
        model_copy = getattr(profile, "model_copy", None)
        if callable(model_copy):
            return model_copy(update={"minimum_passing_score": float(minimum)})
        return profile.copy(update={"minimum_passing_score": float(minimum)})

    def _resolve_output_dir(self, output_dir: str | None) -> Path:
        root = self.runs_dir.resolve()
        if not output_dir:
            root.mkdir(parents=True, exist_ok=True)
            return root
        candidate = Path(output_dir)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if candidate != root and root not in candidate.parents:
            raise UnsafeBenchmarkPathError(
                "Benchmark output_dir must stay under evaluation_lab_runs_dir"
            )
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _safe_child(self, root: Path, rel: Path) -> Path:
        root = root.resolve()
        path = (root / rel).resolve()
        if path != root and root not in path.parents:
            raise UnsafeBenchmarkPathError(f"Path escapes configured root: {rel}")
        return path

    @staticmethod
    def _redact_settings(settings_overrides: dict[str, Any]) -> dict[str, Any]:
        redacted = {}
        for key, value in settings_overrides.items():
            lowered = key.lower()
            if any(secret in lowered for secret in ("key", "secret", "token", "password")):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        return redacted

    @staticmethod
    def _pass_rate(result: BenchmarkRunResult) -> float:
        return result.passed_cases / result.total_cases if result.total_cases else 0.0
