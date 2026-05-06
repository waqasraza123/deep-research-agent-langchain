from __future__ import annotations

import argparse
import json

from .contracts import BenchmarkRunRequest, model_to_plain
from .regression_runner import EvaluationLabRunner


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m deep_research_agent.evaluation_lab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("validate")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--all", action="store_true", dest="run_all")
    run_parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    runner = EvaluationLabRunner()
    if args.command == "list":
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
