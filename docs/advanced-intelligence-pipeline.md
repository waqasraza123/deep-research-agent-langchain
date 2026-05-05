# Advanced Backend Intelligence Pipeline

This repository is backend-only. It has no UI and does not add frontend code for the
advanced intelligence artifacts.

## Run Lifecycle

The `/run` pipeline is artifact driven:

1. Capture the run input snapshot.
2. Fetch or load source content.
3. Run source safety before source text is used as agent context.
4. Write sanitized source copies and annotate `sources.json`.
5. Run temporal extraction and currentness checks.
6. Run quantitative extraction over numeric claims, tables, and CSV-like content.
7. Invoke the configured agent/model with bounded sanitized source context plus temporal and
   quantitative warning blocks.
8. Preserve the required `plan.md`, `notes.md`, `sources.json`, and `report.md`.
9. Build evidence, synthesis, hypothesis tests, confidence updates, evaluation, and summaries.
10. Refresh provenance, dependency DAG, reproducibility report, and replay plan.
11. Write `advanced_intelligence_summary.json` and `.md` with major warnings and confidence.

## Source Safety

Fetched source text is treated as untrusted evidence, not instructions. The source safety subsystem
detects prompt-injection patterns and source-poisoning signals with deterministic heuristics. It
wraps source content in an explicit trust boundary and writes sanitized copies under
`sanitized_sources/`.

High-risk source content is not passed raw into the agent. Critical-risk content is excluded from
agent context by default. The original source metadata remains auditable through `sources.json`,
`source_safety.json`, `prompt_injection_findings.json`, and `source_poisoning_findings.json`.

## Temporal Intelligence

Temporal intelligence extracts publication, update, event, effective, deadline, and version-like
dates from sources, notes, and reports. It builds `timeline.json`, `temporal_profile.json`, and
`currentness_assessment.json`. Freshness-sensitive questions are penalized when sources are stale,
undated, or only weakly dated.

## Quantitative Intelligence

Quantitative intelligence extracts numeric values, numeric claims, metric definitions, table
profiles, CSV profiles, comparisons, and consistency checks. It flags unsupported numeric claims and
comparisons that are not apples-to-apples. These warnings reduce hypothesis confidence when numeric
claims are central to a conclusion.

## Hypotheses And Confidence

Hypotheses are generated offline from the question, subquestions, notes, report, sources, source
audit, retrieval context, evidence ledger, and synthesis artifacts. Evidence testing uses phrase,
keyword, entity, numeric/date overlap, source quality, citation readiness, primary-source signals,
source diversity, freshness, contradictions, source safety, and quantitative consistency.

Confidence updates are conservative. Temporal staleness, high-risk source safety findings,
unsupported numeric claims, contradictions, missing primary sources, and sensitive domains reduce
confidence. The system should not be treated as proof or professional advice.

## Provenance And Replay

The provenance subsystem scans the final run directory, redacts secrets, fingerprints sources and
model invocations, infers artifact dependencies, and writes:

- `artifact_manifest.json` / `.md`
- `artifact_dependency_dag.json` / `.md`
- `reproducibility_report.json` / `.md`
- `replay_plan.json` / `.md`

Replay metadata is best effort. Runs that depend on live URLs or nondeterministic models are marked
partially replayable or not fully replayable.

## Settings

Advanced subsystems are controlled by safe, offline-testable settings:

- `SOURCE_SAFETY_ENABLED`
- `TEMPORAL_INTELLIGENCE_ENABLED`
- `QUANTITATIVE_INTELLIGENCE_ENABLED`
- `HYPOTHESIS_ENGINE_ENABLED`
- `PROVENANCE_ENABLED`
- `HIGH_RISK_SOURCE_POLICY`
- `MAX_HYPOTHESES`
- `MAX_HYPOTHESIS_EVIDENCE_ITEMS`
- `FRESHNESS_WARNING_THRESHOLD_DAYS`
- `QUANTITATIVE_EXTRACTION_ENABLED`
- `PROVENANCE_MANIFEST_ENABLED`
- `REPLAY_PLAN_ENABLED`

## Offline Tests

Run tests without OpenAI, Ollama, external search, or network dependencies:

```bash
cd backend
pytest
ruff check .
```

## Known Limitations

- Source safety, temporal, quantitative, and hypothesis testing are deterministic heuristics.
- Currentness validation does not perform live web refreshes unless a future fetch/search provider is
  explicitly enabled.
- Quantitative extraction is not a full spreadsheet engine.
- Provenance dependency inference is best effort when subsystem producer metadata is unavailable.
- Human review remains required for legal, medical, financial, security, and other high-stakes use.
