# Research Evaluation Lab Benchmarks

This directory contains offline, versionable benchmark cases for the backend-only Research Evaluation Lab.

Each case lives under `cases/<case_id>/` and contains:

- `case.json`: metadata, question, `benchmark://` URLs, local source mapping, traps, tags, and scoring profile overrides.
- `expected.json`: deterministic expectations for artifacts, mentions, warnings, numbers, claims, citations, and confidence bounds.
- `sources/`: local fixture documents used by the offline fetcher.

The `benchmark://` scheme is only valid inside evaluation lab mode. Normal `/run` source fetching still accepts only HTTP(S) URLs.

Mock benchmark runs are deterministic and require no OpenAI, Ollama, external search, or network access. Scores are heuristic regression signals, not absolute semantic truth.

## Quality Gates

The Evaluation Lab governance layer turns benchmark scores into backend quality gates for CI and release checks.

Built-in profiles:

- `smoke`: quick offline confidence check for factual, adversarial, and numeric behavior.
- `pull_request`: default merge gate over the core benchmark suite.
- `adversarial`: source safety and source poisoning gate.
- `citation_strict`: citation/source traceability gate.
- `numeric_temporal`: numeric and stale-source currentness gate.
- `full_regression`: full offline regression gate, with baseline comparison when available.
- `warning_budget`: warning classification and growth control.

Run locally:

```bash
python -m deep_research_agent.evaluation_lab gate --profile smoke
python -m deep_research_agent.evaluation_lab gate --profile pull_request
python -m deep_research_agent.evaluation_lab gate --profile full_regression
```

The `gate` command exits `0` when the gate passes, `1` when the gate fails, and `2` when the gate configuration or runner errors. It prints the path to `governance_summary.md`.

## Baselines

Baselines are compact JSON snapshots under `benchmarks/baselines/`. They store case scores, pass/fail state, missed traps, check summaries, warning fingerprints, artifact fingerprints, and the git commit when available. They do not store full run artifacts or secrets.

Promote intentionally after reviewing the governance report:

```bash
python -m deep_research_agent.evaluation_lab baselines promote --run-id <run_id> --gate-id smoke --name "Reviewed smoke baseline"
python -m deep_research_agent.evaluation_lab baselines list
```

Baseline comparison detects newly failed cases, score drops, newly missed traps, new warning growth, artifact failures, prompt-injection failures, and numeric/temporal/citation regressions.

## Reports

Gate runs write:

- `quality_gate_profile.json/md`
- `quality_gate_run.json/md`
- `quality_gate_thresholds.json/md`
- `baseline_comparison.json/md`
- `regression_findings.json/md`
- `improvement_findings.json/md`
- `warning_audit.json/md`
- `benchmark_coverage.json/md`
- `flakiness_report.json/md`
- `quality_gate_triage.json/md`
- `quality_gate_recommendations.md`
- `governance_summary.json/md`

The triage summary ranks path safety/security failures first, then prompt injection, artifact integrity, missed critical traps, new failed cases, numeric/temporal/citation regressions, low score, warning growth, and coverage gaps.

## Coverage

Coverage is deterministic and based on category coverage, trap coverage, check type coverage, difficulty spread, and source type spread.

```bash
python -m deep_research_agent.evaluation_lab coverage
```

Add new cases by creating `case.json`, `expected.json`, and local `sources/` files. Include tags, difficulty, traps, and expected warnings where relevant.

## Warning Audit

Warnings are classified instead of being broadly hidden:

```bash
python -m deep_research_agent.evaluation_lab warnings-audit --file warnings.txt
```

The audit groups warnings into deprecation, pydantic, fastapi, langchain, deepagents, pytest, resource, user warning, and unknown. Serious warnings include resource/runtime issues such as unclosed files or never-awaited coroutines.

No blanket ignore policy should be added for project warnings. Narrow third-party filters are acceptable only when documented and stable.

## CI

CI can run the smoke gate without external credentials:

```bash
python -m deep_research_agent.evaluation_lab gate --profile smoke
```

The gate uses mock mode and the offline fetcher. `benchmark://` remains Evaluation Lab only and is not enabled for normal `/run`.

## Limitations

- Deterministic checks are heuristic regression signals, not full semantic judges.
- Mock mode validates infrastructure and benchmark governance; it does not prove live model quality.
- Passing a gate does not prove truth or completeness.
- Baseline promotion should be reviewed intentionally.
- Benchmark coverage is useful for regression control but not exhaustive.
