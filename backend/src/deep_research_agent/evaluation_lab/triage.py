from __future__ import annotations

from .contracts import (
    BenchmarkRunResult,
    CheckSeverity,
    CheckType,
    GateTriageSummary,
    RegressionFinding,
    RegressionFindingType,
    WarningAudit,
)


def build_triage_summary(
    run: BenchmarkRunResult,
    regressions: list[RegressionFinding],
    warning_audit: WarningAudit | None = None,
) -> GateTriageSummary:
    priorities: list[tuple[int, str, str]] = []
    root_causes: list[str] = []
    tasks: list[str] = []
    inspect: list[str] = []
    risky: list[str] = []

    for case in run.case_results:
        if case.passed:
            continue
        for check in case.check_results:
            if check.passed:
                continue
            text = f"{case.case_id}: {check.name} - {check.message}"
            priority = 7
            if "path" in check.check_id.lower() or "path" in check.message.lower():
                priority = 1
                root_causes.append("Path safety/security guard failed.")
                tasks.append("Inspect path validation and artifact write containment.")
            elif check.check_type == CheckType.prompt_injection_resistance:
                priority = 2
                root_causes.append("Prompt-injection/source instruction isolation regressed.")
                tasks.append("Harden source safety handling and injected-instruction filtering.")
            elif check.check_type in {
                CheckType.artifact_exists,
                CheckType.artifact_valid_json,
                CheckType.artifact_nonempty,
            }:
                priority = 3
                root_causes.append("Required benchmark artifact integrity failed.")
                tasks.append("Check artifact generation and required file guarantees.")
            elif check.severity == CheckSeverity.critical:
                priority = 4
                root_causes.append("Critical benchmark trap or check failed.")
            elif check.check_type == CheckType.numeric_support:
                priority = 6
                root_causes.append("Numeric extraction or entity-number association regressed.")
                tasks.append("Review numeric checks and source-grounded number rendering.")
            elif check.check_type in {CheckType.date_support, CheckType.stale_source_warning}:
                priority = 6
                root_causes.append("Temporal/staleness handling regressed.")
                tasks.append("Review freshness warnings and currentness guardrails.")
            elif check.check_type in {CheckType.citation_support, CheckType.source_traceability}:
                priority = 6
                root_causes.append("Citation/source traceability regressed.")
                tasks.append("Review citation extraction and unknown-source handling.")
            priorities.append((priority, case.case_id, text))
            inspect.append(case.case_id)
        for trap_id in case.missed_traps:
            priorities.append(
                (4, case.case_id, f"{case.case_id}: missed critical trap `{trap_id}`")
            )
            tasks.append("Inspect trap detection markers in report and notes artifacts.")

    for finding in regressions:
        priority = 5
        if finding.type in {
            RegressionFindingType.new_prompt_injection_failure,
            RegressionFindingType.new_artifact_failure,
        }:
            priority = 2
        elif finding.type in {
            RegressionFindingType.new_numeric_failure,
            RegressionFindingType.new_temporal_failure,
            RegressionFindingType.new_citation_failure,
        }:
            priority = 6
        priorities.append((priority, finding.case_id, finding.message))
        risky.append(finding.message)
        inspect.append(finding.case_id)

    if warning_audit and warning_audit.budget_exceeded:
        priorities.append((8, "", "Warning budget exceeded."))
        root_causes.append("Warnings grew beyond the configured budget.")
        tasks.append("Run warnings audit and fix serious project warnings before adding filters.")

    priorities_sorted = sorted(priorities, key=lambda item: (item[0], item[1], item[2]))
    quick_wins = []
    if warning_audit and warning_audit.serious_warning_count:
        quick_wins.append(
            "Fix serious warnings; they are usually localized lifecycle/resource issues."
        )
    if any("artifact" in item[2].lower() for item in priorities_sorted):
        quick_wins.append("Regenerate missing required artifacts before tuning scoring thresholds.")
    return GateTriageSummary(
        highest_priority_failures=[item[2] for item in priorities_sorted[:12]],
        likely_root_causes=sorted(set(root_causes))[:10],
        suggested_engineering_tasks=sorted(set(tasks))[:12],
        cases_to_inspect_first=[case for case in dict.fromkeys(inspect) if case][:10],
        quick_wins=quick_wins,
        risky_regressions=risky[:10],
    )
