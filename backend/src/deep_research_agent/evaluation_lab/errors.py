from __future__ import annotations


class EvaluationLabError(Exception):
    """Base error for deterministic benchmark evaluation."""


class CaseLoadError(EvaluationLabError):
    pass


class FixtureCorpusError(EvaluationLabError):
    pass


class UnsafeBenchmarkPathError(EvaluationLabError):
    pass


class BenchmarkRunError(EvaluationLabError):
    pass
