from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ci import gate_exit_code
from .contracts import BenchmarkRunRequest, QualityGateRunRequest, model_to_plain
from .coverage import coverage_for_cases_root
from .gate_runner import QualityGateRunner
from .regression_runner import EvaluationLabRunner
from .warning_audit import summarize_warnings


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m deep_research_agent.evaluation_lab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("cases")
    sub.add_parser("list")
    sub.add_parser("validate")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--all", action="store_true", dest="run_all")
    run_parser.add_argument("--case", action="append", default=[])
    gate_parser = sub.add_parser("gate")
    gate_parser.add_argument("--profile", default="smoke")
    gate_parser.add_argument("--case", action="append", default=[])
    gate_parser.add_argument("--update-baseline", action="store_true")
    baselines_parser = sub.add_parser("baselines")
    baselines_sub = baselines_parser.add_subparsers(dest="baselines_command", required=True)
    baselines_sub.add_parser("list")
    promote_parser = baselines_sub.add_parser("promote")
    promote_parser.add_argument("--run-id", required=True)
    promote_parser.add_argument("--gate-id", required=True)
    promote_parser.add_argument("--name", default="Promoted baseline")
    sub.add_parser("coverage")
    warning_parser = sub.add_parser("warnings-audit")
    warning_parser.add_argument("--file")
    args = parser.parse_args()

    runner = EvaluationLabRunner()
    if args.command in {"list", "cases"}:
        print(json.dumps(model_to_plain(runner.list_cases()), indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        print(json.dumps(runner.validate(BenchmarkRunRequest(run_all=True)), indent=2))
        return 0
    if args.command == "run":
        request = BenchmarkRunRequest(
            case_ids=list(args.case),
            run_all=bool(args.run_all or not args.case),
        )
        print(json.dumps(model_to_plain(runner.run_cases(request)), indent=2, sort_keys=True))
        return 0
    if args.command == "gate":
        gate_runner = QualityGateRunner(lab_runner=runner)
        result = gate_runner.run_gate(
            QualityGateRunRequest(
                gate_id=args.profile,
                case_ids=list(args.case),
                update_baseline=bool(args.update_baseline),
            )
        )
        print(json.dumps(model_to_plain(result), indent=2, sort_keys=True))
        print(
            "governance_summary.md: "
            f"{gate_runner.gate_runs_dir / result.gate_run_id / 'governance_summary.md'}"
        )
        return gate_exit_code(result)
    if args.command == "baselines":
        gate_runner = QualityGateRunner(lab_runner=runner)
        if args.baselines_command == "list":
            print(
                json.dumps(
                    model_to_plain(gate_runner.baseline_store.list_baselines()),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.baselines_command == "promote":
            run = runner.read_run(args.run_id)
            baseline = gate_runner.baseline_store.promote_baseline(
                run, gate_id=args.gate_id, name=args.name
            )
            print(json.dumps(model_to_plain(baseline), indent=2, sort_keys=True))
            return 0
    if args.command == "coverage":
        print(json.dumps(model_to_plain(coverage_for_cases_root()), indent=2, sort_keys=True))
        return 0
    if args.command == "warnings-audit":
        text = Path(args.file).read_text(encoding="utf-8") if args.file else ""
        print(json.dumps(model_to_plain(summarize_warnings(text)), indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
