from __future__ import annotations

from .contracts import QualityGateRunResult, QualityGateStatus


def gate_exit_code(result: QualityGateRunResult, *, nonzero_on_failure: bool = True) -> int:
    if result.status == QualityGateStatus.passed:
        return 0
    if result.status in {QualityGateStatus.errored, QualityGateStatus.skipped}:
        return 2
    if result.status == QualityGateStatus.failed and nonzero_on_failure:
        return 1
    return 0
