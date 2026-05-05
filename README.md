# Deep Research Agent Backend

Backend-only FastAPI prototype for an artifact-driven research-agent service built around
LangChain Deep Agents/LangGraph. There is no frontend in this repository; generated image
files are historical demo assets, not an active UI.

The service accepts a research question plus optional source URLs, creates a deterministic
research strategy, fetches and optionally expands sources, runs an agent or deterministic
mock backend, then writes traceable artifacts under `runs/<thread_id>/`.

## What `/run` Does

The integrated backend lifecycle is:

1. Create a run registry record and public `run.json` snapshot.
2. Persist the run input snapshot, settings, review preference, and initial status.
3. Generate `strategy.json`, `strategy.md`, `subquestions.json`, and `verification_plan.md`.
4. Fetch root sources and, when enabled, perform bounded one-hop source expansion.
5. Build and execute an adaptive typed research task graph before agent execution.
6. Load prior local research memory and write `memory_context.json` / `memory_context.md`.
7. Write `sources.json`, `source_graph.json`, `source_graph.md`, and source quality metadata.
8. Audit fetched sources for credibility, freshness, authority, bias risk, primary-source
   likelihood, and citation readiness.
9. Run the model/agent layer, or deterministic mock mode, with orchestration and source-audit
   instructions.
10. Guarantee `plan.md`, `notes.md`, `sources.json`, and `report.md` exist.
11. Index successful fetched sources into local SQLite memory and write memory graph artifacts.
12. Build evidence artifacts from notes, report, and captured sources.
13. Build deterministic synthesis artifacts and, when safe, replace `report.md` with an
    assembled report while preserving the original as `report.raw.md`.
14. Build deterministic research quality evaluation artifacts from the report, evidence ledger,
    source audit, synthesis outputs, and available source text.
15. Persist `events.jsonl`, `events.md`, and `budget.json`.
16. Mark the run completed, failed, cancelled, or waiting for review.

## Generated Artifacts

Required core artifacts:

- `plan.md`
- `notes.md`
- `sources.json`
- `report.md`

Intelligence and traceability artifacts:

- `run.json`
- `strategy.json`
- `strategy.md`
- `subquestions.json`
- `verification_plan.md`
- `task_graph.json`
- `task_graph.md`
- `stage_outputs.json`
- `specialist_findings.md`
- `orchestration_summary.json`
- `orchestration_summary.md`
- `source_graph.json`
- `source_graph.md`
- `source_audit.json`
- `source_audit.md`
- `source_rankings.json`
- `source_warnings.md`
- `citation_readiness.json`
- `memory_context.json`
- `memory_context.md`
- `memory_graph.json`
- `memory_graph.md`
- `evidence_ledger.json`
- `evidence_ledger.md`
- `unsupported_claims.md`
- `contradictions.md`
- `citation_map.json`
- `evidence_coverage.json`
- `synthesis_input.json`
- `synthesis_output.json`
- `findings.json`
- `finding_clusters.json`
- `argument_map.json`
- `argument_map.md`
- `comparison_matrix.json`
- `comparison_matrix.md`
- `decision_memo.json`
- `decision_memo.md`
- `uncertainty_boundaries.json`
- `uncertainty_boundaries.md`
- `report_assembly_plan.json`
- `report_assembly_plan.md`
- `report.raw.md` when synthesis replaces the raw report
- `evaluation.json`
- `evaluation.md`
- `coverage_gaps.json`
- `coverage_gaps.md`
- `hallucination_risk.json`
- `hallucination_risk.md`
- `quality_score.json`
- `quality_score.md`
- `intelligence_summary.json`
- `intelligence_summary.md`
- `events.jsonl`
- `events.md`
- `budget.json`

Fetched source manifests use a stable `source_identity` object where possible. It includes
`source_id`, `url`, `canonical_url`, `normalized_url`, `domain`, `title`, `content_hash`,
`source_kind`, `parent_url`, and `fetched_at`, so memory, source audit, synthesis, and evaluation
can refer to the same source consistently.

Fetched source text and metadata live under `runs/<thread_id>/sources/`.

## Research Memory

The backend maintains an offline SQLite memory store at `runs/_memory/memory.sqlite` by default.
Set `MEMORY_DATA_DIR` to move it elsewhere and `MEMORY_STALE_AFTER_DAYS` to tune reuse warnings.

Memory records include normalized questions, source URLs, canonical URLs, title/domain data,
content hashes, quality scores, deterministic entities/topics, summaries, warnings, and artifact
references. Before each run, the backend retrieves similar previous questions, prior useful
sources, known entities/topics, stale warnings, and source reuse candidates. These are written as
`memory_context.json` and `memory_context.md` and are explicitly treated as prior context, not new
evidence. Fresh fetched source content always remains authoritative.

The deterministic extractor is offline and heuristic. It recognizes people, organizations,
products, frameworks/libraries, locations, dates, money, percentages, numeric values, technical
terms, legal/policy terms, and research topics using regexes, token windows, capitalization, URL
hints, and stopword lists. This is useful for operator triage and cross-run reuse, but it is not a
model-quality entity linker and can miss ambiguous entities or over-label capitalized phrases.

Memory endpoints:

- `GET /memory/search?q=...`
- `GET /memory/sources?domain=...`
- `GET /memory/entities/{entity_name}`
- `POST /runs/{thread_id}/memory/rebuild`

## Adaptive Orchestration

The backend includes a deterministic orchestration layer under
`deep_research_agent.orchestration`. It classifies each request, creates a typed task graph,
routes needed stages to specialist contracts, records skipped stages with reasons, and produces
an instruction block for the LangGraph/Deep Agents run.

Task types include question normalization, source triage, extraction, evidence collection,
subquestion work, contradiction and risk scans, synthesis, citation review, and final report
review. Specialist roles include planner, source triager, evidence collector, skeptical reviewer,
domain analyst, synthesis writer, citation auditor, and risk reviewer.

The first implementation is heuristic and offline. It does not call hidden models and should be
read as auditable planning context, not expert truth. High-stakes, date-sensitive, legal,
medical, financial, policy, comparative, and technical due diligence questions receive stricter
routing and conservative confidence instructions.

## Mock Mode

Mock mode is deterministic, offline, and suitable for tests or orchestration checks:

```bash
curl http://localhost:8000/run \
  -H 'content-type: application/json' \
  -d '{"question":"Validate the backend flow.", "mock_mode": true}'
```

You can also set:

```bash
MODEL_PROVIDER=mock
```

Mock output is clearly marked and is not factual research. Mock fallback is only used when
`ALLOW_MOCK_FALLBACK=true` or request field `allow_mock_fallback=true` is set.

## Source Expansion

Source expansion is safe by default:

- `follow_links` defaults to `false`.
- `max_links_per_source` defaults to `0` and is clamped to `0-10`.
- Expansion is one hop only.
- Global crawl expansion is budgeted.
- Unsafe hosts, local/private addresses, unsupported schemes, noisy links, and duplicates are
  skipped or blocked before fetch where possible.

`source_graph.json` and `source_graph.md` explain fetched, skipped, duplicate, and discovered
links, including quality scores and parent-child relationships.

## Source Audit

The source audit subsystem is deterministic and offline. It runs after source fetching and
before report generation, then passes a concise instruction block to the agent describing:

- sources to prioritize for direct citation
- sources that need primary-source verification
- stale or undated sources when freshness matters
- bias, promotion, affiliate, or vendor-comparison risks
- missing source types, authority gaps, and citation blockers

Each source receives typed scores for credibility, freshness, authority, bias risk,
primary-source likelihood, and citation readiness. Recommended usage is one of:
`cite_directly`, `use_as_background`, `use_with_caution`, `verify_with_primary_source`, or
`exclude_from_report`.

The audit is intentionally conservative. It uses URL, domain, title, source metadata, content
length, visible dates, citation/reference language, promotional wording, and duplicate/fetch
warnings. It does not call external credibility APIs and does not prove factual truth.
High-scoring sources can still be wrong, and low-scoring sources can still be useful as leads.

## Evidence Ledger

The evidence layer is deterministic and offline. It extracts candidate claims from `notes.md`,
`report.md`, and source text, maps citations using lexical/value overlap, scores confidence,
and flags unsupported claims or possible contradictions.

Limitations:

- It is a review aid, not proof of truth.
- Weak citation matches can be false positives.
- Broad or stale claims still need human review.
- Contradiction detection is heuristic.

## Synthesis Engine

The synthesis layer is deterministic, offline, and backend-only. It consumes `notes.md`,
`report.md`, `sources.json`, strategy artifacts, source audit artifacts, and
`evidence_ledger.json` when present. It extracts traceable findings, clusters them, and writes:

- Argument maps separating main answer, supporting claims, counterclaims, assumptions, weak
  evidence, unresolved questions, implications, and risks.
- Comparison matrices when the question is comparative, such as `A vs B` or vendor/framework
  tradeoff questions.
- Decision memos when the question implies adoption, recommendation, architecture choice, or
  tradeoff.
- Uncertainty boundaries covering known, likely, uncertain, unverified, freshness-dependent,
  human-review, and primary-source requirements.
- A stronger assembled `report.md` from structured pieces when there are enough extracted
  findings. The raw agent report is preserved as `report.raw.md`.

Supported report profiles are `concise_answer`, `deep_research_report`,
`technical_due_diligence`, `comparative_report`, `decision_memo`, `risk_review`, and
`literature_style_review`.

Limitations:

- Synthesis does not call a model and does not introduce new facts.
- Comparison dimensions and decision options are inferred with heuristics.
- Empty cells and missing evidence are marked as gaps rather than filled.
- Confidence labels come from available evidence or artifact provenance; no fake confidence is
  assigned.
- Operators should review weak, unsupported, contradictory, stale, or high-stakes findings before
  relying on the output.

## Research Quality Evaluation

The evaluation subsystem is deterministic, offline, and backend-only. It runs after evidence and
synthesis when available, and can be rebuilt with `POST /runs/{thread_id}/evaluation/rebuild`.
It does not call hidden models and does not require external credentials.

Evaluation consumes whichever artifacts exist: `evidence_ledger.json`, `source_audit.json`,
synthesis artifacts, `report.md`, `notes.md`, `sources.json`, source text, strategy, and
subquestion artifacts. If source text or the evidence ledger is missing, confidence is lowered and
low-confidence output is capped rather than treated as verified.

The evaluator writes:

- `evaluation.json` / `evaluation.md`: overall score, confidence, criterion scores, reasons, and
  recommended fixes
- `coverage_gaps.json` / `coverage_gaps.md`: unanswered subquestions, missing entities, unused
  sources, unsupported claims, missing primary sources, missing opposing views, and freshness gaps
- `hallucination_risk.json` / `hallucination_risk.md`: unsupported values, dates, entities,
  strong claims, recommendations, contradictions, and absolute wording
- `quality_score.json` / `quality_score.md`: compact score summary for operators and automation

## Benchmarks

Offline benchmark cases live under `backend/benchmarks/cases/`. Each case defines the question,
mocked source documents, expected artifacts, expected entities, minimum scores, known traps, and
required warnings. The regression runner builds temporary run artifacts, rebuilds evidence and
evaluation, then reports pass/fail without network or model calls.

API usage:

```bash
curl http://localhost:8000/benchmarks/cases
curl -X POST http://localhost:8000/benchmarks/run \
  -H 'content-type: application/json' \
  -d '{"case_ids":["current-ai-search-quality"]}'
```

Local test usage:

```bash
cd backend
pytest tests/test_evaluation_subsystem.py
```

## Models

Supported providers:

- `mock`: deterministic offline model.
- `openai`: OpenAI-compatible remote API, requires `OPENAI_API_KEY`.
- `llamacpp`: OpenAI-compatible local llama.cpp endpoint.
- `ollama`: local Ollama model.

Example OpenAI-compatible configuration:

```bash
MODEL_PROVIDER=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TIMEOUT_S=60
OPENAI_MAX_RETRIES=1
OPENAI_MAX_TOKENS=350
```

Example Ollama configuration:

```bash
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
OLLAMA_NUM_PREDICT=220
```

Example llama.cpp configuration:

```bash
MODEL_PROVIDER=llamacpp
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_MODEL=local-model
```

Backend intelligence feature flags default to enabled and remain offline/deterministic unless the
model provider itself is remote:

```bash
MEMORY_ENABLED=true
SOURCE_REUSE_ENABLED=true
SOURCE_AUDIT_ENABLED=true
ORCHESTRATION_ENABLED=true
SYNTHESIS_ENABLED=true
EVALUATION_ENABLED=true
MAX_MEMORY_RESULTS=20
SOURCE_SCORING_THRESHOLD=0.45
EVALUATION_THRESHOLD=0.65
BENCHMARK_PATH=backend/benchmarks
```

`intelligence_summary.json` and `intelligence_summary.md` are generated at the end of a run as the
operator-facing rollup across memory, orchestration, source audit, synthesis, and evaluation. They
summarize top sources, warnings, coverage gaps, hallucination risk, generated artifacts, and
recommended follow-up actions.

## API

Useful endpoints:

- `GET /health`
- `GET /models`
- `GET /runtime/diagnostics`
- `POST /orchestration/preview`
- `POST /research-plan`
- `POST /plan`
- `POST /run`
- `GET /runs`
- `GET /runs/{thread_id}`
- `GET /runs/{thread_id}/task-graph`
- `GET /runs/{thread_id}/stage-outputs`
- `GET /runs/{thread_id}/artifacts`
- `GET /runs/{thread_id}/artifacts/{artifact_name}`
- `GET /runs/{thread_id}/events`
- `GET /runs/{thread_id}/budget`
- `GET /runs/{thread_id}/memory`
- `POST /source-audit`
- `GET /runs/{thread_id}/source-audit`
- `GET /runs/{thread_id}/citation-readiness`
- `POST /runs/{thread_id}/evidence/rebuild`
- `POST /runs/{thread_id}/synthesis/rebuild`
- `GET /runs/{thread_id}/synthesis`
- `POST /runs/{thread_id}/evaluation/rebuild`
- `GET /runs/{thread_id}/evaluation`
- `GET /runs/{thread_id}/quality-score`
- `GET /runs/{thread_id}/intelligence-summary`
- `POST /benchmarks/run`
- `GET /benchmarks/cases`
- `GET /runs/{thread_id}/argument-map`
- `GET /runs/{thread_id}/comparison-matrix`
- `GET /runs/{thread_id}/decision-memo`
- `GET /runs/{thread_id}/uncertainty`
- `POST /runs/{thread_id}/cancel`
- `GET /runs/{thread_id}/review`
- `POST /runs/{thread_id}/review/approve`
- `POST /runs/{thread_id}/review/request-changes`
- `POST /runs/{thread_id}/review/reject`

Legacy artifact aliases remain available under `/threads/{thread_id}/artifacts`.

## Review Gates And Cancellation

Pass `"require_review": true` to `/run` to finish in `waiting_for_review`. Operators can then
approve, request changes, or reject through the review endpoints.

Cancellation is cooperative. The backend checks cancellation markers between major stages, but
it cannot interrupt an in-flight synchronous `agent.invoke` call.

## Local Development

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt
pip install -e backend
uvicorn deep_research_agent.api:app --reload --port 8000
```

Run tests:

```bash
cd backend
pytest
ruff check .
```

CI installs runtime and development requirements, installs the backend package in editable
mode, then runs lint, type checks, and offline tests.

## Safety Defaults

- No frontend or browser automation is required.
- Tests run without OpenAI credentials, Ollama, or external network access.
- `runs/` is ignored except for `runs/.keep`.
- Root source count and link expansion are capped.
- Source fetches enforce URL validation and host blocking.
- Budgets track model calls, source fetches, generated characters, runtime, artifact size, and
  crawl expansion.
