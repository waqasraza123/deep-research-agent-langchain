# Deep Research Agent Backend

> **deploy on cloud**
>
> - **Render:** deploy as a Python Web Service from the repo root. Build with
>   `pip install -r backend/requirements.txt && pip install -e backend`; start with
>   `uvicorn deep_research_agent.api:app --host 0.0.0.0 --port $PORT`.
> - **Custom cloud/container:** run any Python 3.11+ ASGI host that installs the backend package
>   and starts `uvicorn deep_research_agent.api:app --host 0.0.0.0 --port ${PORT:-8000}`. Mount or
>   persist `runs/` if you need generated artifacts and SQLite memory to survive restarts.

Backend-only FastAPI prototype for an artifact-driven research-agent service built with
LangChain Deep Agents/LangGraph patterns. There is no active frontend in this repository.

The service accepts a research question plus optional source URLs, builds a strategy, fetches and
audits sources, runs either a configured model provider or deterministic mock mode, then writes
traceable artifacts under `runs/<thread_id>/`.

## Features

- FastAPI API with `/run`, run inspection, artifact download, cancellation, and review endpoints.
- Deterministic research protocols and intelligence profiles for general, technical, legal,
  market, academic, financial, medical, and current-events research.
- Offline-safe source discovery planning, source audit, document parsing, lexical retrieval,
  temporal intelligence, quantitative intelligence, evidence ledger, hypothesis testing,
  synthesis, verification, quality evaluation, and local SQLite memory.
- Backend provenance artifacts for manifests, dependency DAGs, reproducibility reports, replay
  plans, and run-to-run artifact diffs. See `docs/provenance.md`.
- Source safety, temporal, quantitative, hypothesis, and provenance signals are merged into
  `advanced_intelligence_summary.json` / `.md`. See `docs/advanced-intelligence-pipeline.md`.
- Model provider support for `openai`, `ollama`, `llamacpp`, and deterministic `mock`.
- Safety defaults for bounded source fetching, one-hop optional link expansion, URL validation,
  budget tracking, and high-stakes review recommendations.
- SQLite-backed autonomous runtime control for long-running research jobs: idempotent submission,
  durable job/stage/event records, local worker leasing, pause/resume/cancel, retry/dead-letter,
  stale lease recovery, diagnostics, and runtime artifacts. This is backend-only.
- Research Workflow Compiler and gated execution runtime for typed backend-only workflow modes,
  dry-run previews, rebuilds from existing artifacts, artifact contracts, readiness reports, and
  optional Evaluation Lab quality gates.

## Local Development

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt
pip install -e backend
uvicorn deep_research_agent.api:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Run a deterministic offline request:

```bash
curl http://localhost:8000/run \
  -H 'content-type: application/json' \
  -d '{"question":"Validate the backend flow.", "mock_mode": true}'
```

Run tests and lint:

```bash
cd backend
pytest
ruff check .
```

## Configuration

Common model settings:

```bash
# deterministic offline mode
MODEL_PROVIDER=mock

# OpenAI-compatible remote API
MODEL_PROVIDER=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini

# local Ollama
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# local llama.cpp OpenAI-compatible server
MODEL_PROVIDER=llamacpp
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_MODEL=local-model
```

Useful runtime settings:

```bash
PORT=8000
MEMORY_ENABLED=true
SOURCE_DISCOVERY_ENABLED=false
SOURCE_DISCOVERY_PROVIDER=disabled
SOURCE_SAFETY_ENABLED=true
TEMPORAL_INTELLIGENCE_ENABLED=true
QUANTITATIVE_INTELLIGENCE_ENABLED=true
HYPOTHESIS_ENGINE_ENABLED=true
PROVENANCE_ENABLED=true
DOCUMENT_INTELLIGENCE_ENABLED=true
RETRIEVAL_ENABLED=true
EMBEDDING_PROVIDER=disabled
VERIFICATION_ENABLED=true
SYNTHESIS_ENABLED=true
EVALUATION_ENABLED=true
HIGH_RISK_SOURCE_POLICY=quote_high_exclude_critical
MAX_HYPOTHESES=12
MAX_HYPOTHESIS_EVIDENCE_ITEMS=6

# autonomous runtime control
RUNTIME_CONTROL_ENABLED=true
RUNTIME_ASYNC_ENABLED=false
RUNTIME_SQLITE_PATH=
RUNTIME_LEASE_SECONDS=120
RUNTIME_HEARTBEAT_SECONDS=30
RUNTIME_MAX_ATTEMPTS=3
RUNTIME_MAX_RUNTIME_SECONDS=900
RUNTIME_MAX_STAGE_SECONDS=300
RUNTIME_MAX_EVENTS=5000
RUNTIME_MAX_ARTIFACT_BYTES=50000000
RUNTIME_ALLOW_FORCE_CANCEL=true
RUNTIME_RESUME_ENABLED=true
RUNTIME_DEAD_LETTER_ENABLED=true
RUNTIME_RUN_POSTPROCESSING=true
RUNTIME_MOCK_AGENT_EXECUTION_ENABLED=false
```

Source discovery is disabled by default and does not perform hidden live search. `mock` and
`static` provider modes are available for tests and deterministic development.

## API

Most-used endpoints:

- `GET /health`
- `GET /models`
- `GET /runtime/diagnostics`
- `POST /run`
- `GET /runs`
- `GET /runs/{thread_id}`
- `GET /runs/{thread_id}/artifacts`
- `GET /runs/{thread_id}/artifacts/{artifact_name}`
- `GET /runs/{thread_id}/manifest`
- `GET /runs/{thread_id}/provenance`
- `GET /runs/{thread_id}/reproducibility`
- `GET /runs/{thread_id}/replay-plan`
- `POST /runs/{thread_id}/replay`
- `GET /runs/{thread_id}/advanced-intelligence-summary`
- `POST /runs/diff`
- `GET /runs/{thread_id}/events`
- `GET /runs/{thread_id}/budget`
- `POST /runs/{thread_id}/cancel`
- `GET /runs/{thread_id}/review`
- `POST /runs/{thread_id}/review/approve`
- `POST /runs/{thread_id}/review/request-changes`
- `POST /runs/{thread_id}/review/reject`
- `POST /runtime/jobs`
- `GET /runtime/jobs`
- `GET /runtime/jobs/{job_id}`
- `POST /runtime/jobs/{job_id}/run`
- `POST /runtime/worker/process-next`
- `POST /runtime/jobs/{job_id}/cancel`
- `POST /runtime/jobs/{job_id}/pause`
- `POST /runtime/jobs/{job_id}/resume`
- `GET /runtime/jobs/{job_id}/events`
- `GET /runtime/jobs/{job_id}/stages`
- `GET /runtime/jobs/{job_id}/budget`
- `GET /runtime/jobs/{job_id}/recovery-plan`
- `POST /runtime/recover-stale`
- `GET /runtime/dead-letter`
- `POST /runtime/jobs/{job_id}/restore`
- `GET /runs/{thread_id}/runtime`
- `GET /workflows/templates`
- `GET /workflows/templates/{template_id}`
- `POST /workflows/preview`
- `POST /workflows/compile`
- `POST /workflows/run`
- `POST /runs/{thread_id}/workflows/rebuild`
- `GET /runs/{thread_id}/workflow`
- `GET /runs/{thread_id}/workflow/manifest`
- `GET /runs/{thread_id}/workflow/readiness`
- `GET /runs/{thread_id}/workflow/stages`
- `GET /runs/{thread_id}/workflow/plan`
- `POST /runs/{thread_id}/workflow/quality-gate`

Specialized endpoints also exist for protocols, source discovery, document intelligence,
retrieval, memory, temporal intelligence, quantitative intelligence, evidence, hypotheses, verification, synthesis,
evaluation, benchmarks, and quality scores.

## Research Workflow Compiler

The workflow runtime is backend-only. It wraps the existing research service in a typed compiler
and executor without changing the legacy `/run` behavior unless workflow fields are supplied.

Built-in workflow modes include `quick_brief`, `deep_research`, `technical_due_diligence`,
`framework_comparison`, `vendor_evaluation`, `legal_policy_review`, `implementation_planning`,
`source_audit_only`, `evidence_extraction_only`, `verification_only`, `evaluation_only`,
`adversarial_source_review`, `offline_benchmark`, and `rebuild_from_artifacts`.

Each template defines stages, dependencies, artifact contracts, policies, success criteria, and
failure behavior. The compiler resolves a template by `template_id`, explicit `mode`, or
deterministic question inference, then writes:

- `workflow_template.json` / `.md`
- `workflow_compiled.json` / `.md`
- `workflow_execution_plan.json` / `.md`
- `workflow_dependency_graph.json` / `.md`
- `workflow_artifact_contracts.json` / `.md`
- `workflow_warnings.json` / `.md`

Execution records every stage in `workflow_stage_results.json` / `.md`. Finalization validates
required artifacts, then writes `workflow_artifact_validation.json` / `.md`,
`workflow_manifest.json` / `.md`, `workflow_readiness.json` / `.md`, and
`workflow_execution_summary.json` / `.md`. Existing guaranteed artifacts remain `plan.md`,
`notes.md`, `sources.json`, and `report.md`.

Preview is deterministic and does not execute stages:

```bash
curl http://localhost:8000/workflows/preview \
  -H 'content-type: application/json' \
  -d '{"question":"Compare LangGraph vs CrewAI", "mode":"framework_comparison"}'
```

Run a workflow offline with mock artifacts:

```bash
curl http://localhost:8000/workflows/run \
  -H 'content-type: application/json' \
  -d '{"question":"Validate workflow runtime", "mode":"quick_brief",
       "settings_overrides":{"model_provider":"mock","mock_mode":true}}'
```

Rebuild downstream artifacts from an existing run without refetching or model calls by default:

```bash
curl http://localhost:8000/runs/<thread_id>/workflows/rebuild \
  -H 'content-type: application/json' \
  -d '{"thread_id":"<thread_id>", "mode":"verification_only"}'
```

Quality gates are optional. When `run_quality_gate` is true, the workflow adds a quality-gate
stage and runs the selected Evaluation Lab gate in offline/mock mode, defaulting to `smoke`.
Gate failures degrade readiness unless `WORKFLOWS_FAIL_ON_QUALITY_GATE_FAILURE=true`.

Workflow settings:

```bash
WORKFLOWS_ENABLED=true
WORKFLOWS_DEFAULT_MODE=quick_brief
WORKFLOWS_MAX_STAGES=32
WORKFLOWS_ALLOW_MOCK_AGENT=false
WORKFLOWS_ALLOW_EXTERNAL_NETWORK=true
WORKFLOWS_REBUILD_ALLOW_REFETCH=false
WORKFLOWS_REBUILD_ALLOW_MODEL_CALLS=false
WORKFLOWS_QUALITY_GATE_ENABLED=true
WORKFLOWS_DEFAULT_QUALITY_GATE=smoke
WORKFLOWS_FAIL_ON_QUALITY_GATE_FAILURE=false
```

Known limitations:

- Optional stages run only when the matching backend subsystem is available; otherwise the stage is
  skipped or the workflow is degraded according to the template policy.
- Mock mode is for local development, CI, and artifact plumbing. It does not prove research quality.
- Workflow readiness is a backend artifact and quality signal, not a guarantee of truth.
- Passing the Evaluation Lab smoke gate does not prove live model quality.
- Rebuild workflows use available artifacts and cannot recover missing source evidence unless
  refetch or model calls are explicitly allowed.

## Provenance Replay

Each completed run can now be replayed into a separate run directory with:

```bash
curl http://localhost:8000/runs/<thread_id>/replay \
  -H 'content-type: application/json' \
  -d '{"offline_only":true}'
```

Replay copies only seed artifacts needed for deterministic reconstruction, rewrites
`runs/<source_thread_id>/...` source references to the replay thread, rebuilds offline intelligence
layers, writes `replay_execution.json` / `.md`, refreshes provenance, and returns a manifest diff
plus expected-hash matches and mismatches. It does not refetch live URLs or rerun model-dependent
agent generation in offline mode, so report text and source payloads are treated as captured inputs.

## Agentic Research Control Plane

This is still a backend-only service. The Agentic Research Control Plane adds governance and
auditability around the existing Deep Agents/LangGraph runner; it is not a frontend operator
console.

The control plane defines typed specialist roles, deterministic skill selection, source-context
quarantine, tool and filesystem policies, handoff plans, compiled instructions, trace analysis,
and artifact validation. It preserves the existing required deliverables:

- `plan.md`
- `notes.md`
- `sources.json`
- `report.md`

Built-in roles include supervisor, planner, source triager, source reader, evidence extractor,
skeptical reviewer, technical analyst, comparison analyst, risk reviewer, citation auditor,
synthesis writer, and final editor. Role selection is deterministic: comparative questions add
the comparison analyst, technical questions add the technical analyst, sensitive/current/legal/
financial/security questions add risk and skeptical review, and strict citation mode adds the
citation auditor.

Built-in skills cover question decomposition, source quality triage, untrusted source reading,
evidence tables, comparative matrices, technical due diligence, contradiction scans,
overclaiming review, citation readiness, risk registers, synthesis outline, final polishing,
temporal currentness, quantitative claim review, and source safety review.

Source text is quarantined as untrusted evidence. Compiled system instructions include the trust
boundary and policy summaries, but do not dump raw source content into trusted supervisor
instructions. Source-looking commands such as “ignore previous instructions,” “reveal secrets,”
or “execute this” are flagged for analysis rather than followed.

Control-plane policies restrict which roles can use source fetch, artifact read/write, model, and
subagent categories. Filesystem policy only allows run-directory artifacts, blocks traversal, and
limits role write permissions. Some enforcement is direct, such as the governed source-fetch tool
wrapper; other enforcement is advisory/post-run when the installed Deep Agents API does not expose
role-specific runtime hooks.

Preview without running the agent:

```bash
curl http://localhost:8000/agent-control/preview \
  -H 'content-type: application/json' \
  -d '{"question":"Compare FastAPI and LangGraph for a backend research agent.","urls":[]}'
```

Inspect control-plane metadata:

- `GET /agent-control/roles`
- `GET /agent-control/skills`
- `POST /agent-control/preview`
- `GET /runs/{thread_id}/agent-control`
- `GET /runs/{thread_id}/agent-control/plan`
- `GET /runs/{thread_id}/agent-control/policies`
- `GET /runs/{thread_id}/agent-control/instructions`
- `GET /runs/{thread_id}/agent-control/handoffs`
- `GET /runs/{thread_id}/agent-control/trace`
- `GET /runs/{thread_id}/agent-control/validation`
- `POST /runs/{thread_id}/agent-control/rebuild`

Generated control artifacts include `agent_control_plan.json/.md`, `role_selection.json/.md`,
`skill_selection.json/.md`, `agent_policies.json/.md`, `tool_policies.json`,
`filesystem_policies.json`, `context_bundles.json/.md`, `trust_boundary.md`,
`source_context_warnings.json/.md`, `compiled_instructions.json`,
`supervisor_instructions.md`, `subagent_instructions.md`, `subagent_specs.json/.md`,
`artifact_contracts.json/.md`, `agent_handoffs.json/.md`, `agent_trace.jsonl/.md`,
`trace_analysis.json/.md`, `policy_violations.json/.md`, `handoff_validation.json/.md`,
`agent_output_validation.json/.md`, `agent_control_summary.json/.md`,
`control_plane_warnings.md`, and `agent_control_error.json` when a control-plane step fails.

Relevant settings default to enabled:

```bash
AGENT_CONTROL_ENABLED=true
AGENT_CONTROL_STRICT_ROLE_ISOLATION=true
AGENT_CONTROL_SOURCE_CONTEXT_QUARANTINE_ENABLED=true
AGENT_CONTROL_TOOL_GOVERNANCE_ENABLED=true
AGENT_CONTROL_FILESYSTEM_GOVERNANCE_ENABLED=true
AGENT_CONTROL_SKILL_SELECTION_ENABLED=true
AGENT_CONTROL_SUBAGENT_PLANNING_ENABLED=true
AGENT_CONTROL_TRACE_ANALYSIS_ENABLED=true
AGENT_CONTROL_ARTIFACT_VALIDATION_ENABLED=true
AGENT_CONTROL_MAX_COMPILED_INSTRUCTION_CHARS=12000
AGENT_CONTROL_MAX_SUBAGENTS=8
AGENT_CONTROL_MAX_HANDOFFS=32
AGENT_CONTROL_MAX_CONTEXT_CHARS_PER_ROLE=16000
AGENT_CONTROL_FAIL_ON_POLICY_VIOLATION=false
AGENT_CONTROL_FAIL_ON_MISSING_REQUIRED_ARTIFACT=false
```

Offline tests do not require OpenAI, Ollama, external web search, or external services:

```bash
cd backend
pytest tests/test_agent_control.py
pytest
```

Known limitations: direct subagent/tool/filesystem enforcement depends on the installed Deep
Agents API surface, so some policies are recorded as advisory and validated after the run. Model
behavior still determines specialist quality unless deterministic mock mode is used. Post-run
trace analysis can audit artifacts and known events, but it cannot prove every internal model
decision without deeper callback instrumentation.

## Autonomous Runtime Control

The original `POST /run` path remains synchronous by default. Set `RUNTIME_ASYNC_ENABLED=true` or
send `"runtime_async": true` to submit through the autonomous runtime instead. The async response
returns a `job_id`, `thread_id`, current status, current stage, warnings, and artifact links without
requiring the request lifecycle to stay attached to model execution. For local development you can
send `"run_now": true` or call `POST /runtime/jobs/{job_id}/run` / `POST /runtime/worker/process-next`
to process work once in-process.

Runtime jobs are stored in SQLite, defaulting to `runs/runtime.sqlite3`. The repository stores jobs,
stage records, events, leases, queue entries, control requests, and dead-letter records. Job
submission computes an idempotency key from the normalized question, URL set, and redacted settings;
active duplicates return the existing job instead of creating duplicate work.

Runtime execution is stage-level resumable:

- `input_snapshot`
- `planning`
- `source_fetching`
- `source_processing`
- `agent_execution`
- `artifact_backfill`
- `intelligence_postprocessing`
- `verification`
- `finalization`

The runtime guarantees the existing deliverables where possible: `plan.md`, `notes.md`,
`sources.json`, and `report.md`. It also writes runtime artifacts such as
`runtime_input_snapshot.json`, `runtime_job.json`, `runtime_stages.json`, `runtime_stages.md`,
`runtime_events.jsonl`, `runtime_events.md`, `runtime_budget.json`, `runtime_budget.md`,
`runtime_source_fetch_summary.json`, `runtime_source_processing_summary.md`,
`runtime_postprocessing_summary.md`, `runtime_verification_skipped.md`,
`runtime_recovery_plan.json`, `runtime_recovery_plan.md`, `runtime_final_summary.json`,
`runtime_final_summary.md`, `runtime_dead_letter.json`, and `runtime_error.json` when applicable.

Cancellation and pause are cooperative. A queued job can be cancelled or paused immediately. A
running job transitions to `cancelling` or `pausing` and the worker stops between stages. Force
cancellation marks runtime state immediately, but it may not interrupt an already-blocking model
call unless the underlying provider supports cancellation.

Retry uses a typed policy with exponential backoff metadata. Transient timeout/provider-style
errors are retryable; validation, security/path traversal, and missing configuration-style errors
are not. Exhausted jobs move to the dead-letter queue when enabled and can be restored with
`POST /runtime/jobs/{job_id}/restore`.

Stale lease recovery is local and SQLite-backed. `POST /runtime/recover-stale` expires leases that
missed heartbeat, inspects stage/artifact state, and requeues jobs when stage-level recovery is
safe. Resume is artifact/stage driven; it is not token-level continuation inside a model call.

Budgets are checked before and after stage execution and persisted to `runtime_budget.json`. The
runtime tracks elapsed runtime, per-stage time, source counts, model calls where detectable,
artifact bytes, retry count, event count, source count, and extracted character count. Soft budget
exceedance records warnings; hard exceedance fails the job when
`RUNTIME_FAIL_ON_BUDGET_EXCEEDED=true`.

`RUNTIME_MOCK_AGENT_EXECUTION_ENABLED=true` enables deterministic offline runtime worker execution
for tests and local control-plane validation. It writes the required artifacts and marks output as
mock-generated. Production-like settings do not silently mock unless explicitly configured or the
job request asks for mock execution.

Known limits: the SQLite runtime is local/dev/single-node friendly, not a distributed queue
replacement; cancellation is cooperative; force cancellation does not kill in-flight blocking calls;
resume is stage-level/artifact-based; runtime mock execution intentionally skips network fetching
for offline control-plane tests.

Temporal endpoints:

- `POST /runs/{thread_id}/temporal/rebuild`
- `GET /runs/{thread_id}/timeline`
- `GET /runs/{thread_id}/currentness`
- `GET /runs/{thread_id}/temporal-warnings`

Quantitative endpoints:

- `POST /runs/{thread_id}/quantitative/rebuild`
- `GET /runs/{thread_id}/quantitative`
- `GET /runs/{thread_id}/numeric-claims`
- `GET /runs/{thread_id}/quantitative-warnings`

See `docs/quantitative-intelligence.md` for artifact details and limitations.

Hypothesis endpoints:

- `POST /runs/{thread_id}/hypotheses/rebuild`
- `GET /runs/{thread_id}/hypotheses`
- `GET /runs/{thread_id}/hypothesis-graph`
- `GET /runs/{thread_id}/confidence-updates`

Source safety endpoints:

- `POST /source-safety/assess`
- `GET /runs/{thread_id}/source-safety`

Advanced summary endpoint:

- `GET /runs/{thread_id}/advanced-intelligence-summary`

See `docs/advanced-intelligence-pipeline.md` for the integrated backend-only lifecycle,
source-safety isolation behavior, confidence penalties, provenance/replay metadata, and limitations.

Research Intelligence Kernel endpoints:

- `POST /intelligence/analyze`
- `POST /runs/{thread_id}/intelligence/rebuild`
- `GET /runs/{thread_id}/intelligence`
- `GET /runs/{thread_id}/blueprint`
- `GET /runs/{thread_id}/critique`
- `GET /runs/{thread_id}/verification`
- `GET /runs/{thread_id}/confidence`
- `GET /runs/{thread_id}/readiness`

The rebuild endpoint uses existing run artifacts only. It does not refetch sources, perform live
search, call OpenAI, call Ollama, or rerun the agent.

## Research Intelligence Kernel

The backend includes a deterministic, backend-only Research Intelligence Kernel under
`deep_research_agent.intelligence_kernel`. It runs after `/run` creates or backfills the guaranteed
artifacts and can also be rebuilt for an existing `runs/<thread_id>/` folder. The kernel is an audit
and quality layer around source content and generated artifacts; it is not a replacement for the
agent, for live source discovery, or for human review.

The kernel does the following:

- analyzes the request intent and complexity using offline rules
- creates a typed execution blueprint with source, evidence, verification, citation, freshness,
  safety, and synthesis policies
- normalizes `sources.json` into governed source units with URL normalization, trust, role,
  duplicate, weak-extraction, and prompt-injection warnings
- extracts deterministic evidence units from source text and existing markdown artifacts
- extracts claims from `report.md` and `notes.md`
- critiques unsupported, stale, numeric, one-sided, sensitive-domain, and weak-artifact risks
- generates and runs local verification tasks against existing evidence only
- calibrates confidence for claims and the whole report
- writes an artifact registry and operator-facing readiness summary

The main kernel artifacts are:

- `kernel_blueprint.json` / `.md`
- `kernel_passes.json` / `.md`
- `source_inventory.json` / `.md`
- `source_units.json` / `.md`
- `evidence_units.json` / `.md`
- `evidence_coverage.json` / `.md`
- `claims.json` / `.md`
- `critique_findings.json` / `.md`
- `operator_warnings.json` / `.md`
- `verification_tasks.json` / `.md`
- `verification_results.json` / `.md`
- `confidence_calibration.json` / `.md`
- `kernel_artifact_registry.json` / `.md`
- `kernel_summary.json` / `.md`
- `research_readiness.md`

`research_readiness.md` is the fastest operator artifact to read. It answers whether the report is
usable, the final confidence, the biggest risks, manual checks required, weak claims, strongest
sources, sources not to trust blindly, and next actions.

Kernel settings default to offline-safe behavior:

```bash
INTELLIGENCE_KERNEL_ENABLED=true
INTELLIGENCE_OFFLINE_MODE=true
INTELLIGENCE_MAX_SOURCE_UNITS=100
INTELLIGENCE_MAX_EVIDENCE_UNITS=500
INTELLIGENCE_MAX_CLAIMS=200
INTELLIGENCE_MAX_CLAIMS_TO_VERIFY=50
INTELLIGENCE_FAIL_ON_CRITICAL_WARNINGS=false
INTELLIGENCE_SENSITIVE_DOMAIN_REVIEW_REQUIRED=true
```

Run offline kernel tests with:

```bash
cd backend
.venv/bin/python -m pytest tests/test_intelligence_kernel.py -q
```

Known limitations: deterministic verification can find overlap, missing support, numeric/date
mismatches, weak source coverage, and sensitive-domain risk, but it does not prove claims true. It
uses local artifacts as evidence and is deliberately conservative when sources are missing,
undated, weak, or not primary. Medical, legal, and financial outputs remain informational and
require qualified human review.

## Temporal Intelligence

The backend includes an offline-only temporal subsystem under
`deep_research_agent.temporal`. It extracts dates from source metadata, URLs, titles, source text,
`report.md`, `notes.md`, and `sources.json`; detects current/latest questions; identifies release,
changelog, deprecated, archived, legacy, beta, preview, stable, and semantic-version signals; builds
timelines; classifies source currentness; and checks date-sensitive report claims for dated source
support.

Temporal checks are deterministic heuristics, not web lookups and not model calls. They deliberately
prefer uncertainty over over-normalizing ambiguous dates. Access/fetch dates are recorded, but source
currentness is based on publication, update, effective, release, deadline, event, or mentioned dates
when available. If a freshness-sensitive question relies on stale or undated sources, the agent
receives a concise warning block before report writing and the run stores temporal warning artifacts.

## Hypothesis-Driven Research

The backend includes an offline-only hypothesis subsystem under
`deep_research_agent.hypotheses`. It proposes possible answers from the research question,
subquestions, `plan.md`, `notes.md`, `report.md`, `sources.json`, source audit artifacts,
retrieval context packs, evidence ledger, and synthesis output when available. It then tests each
hypothesis against deterministic evidence signals instead of asking a model for a judgment.

Hypothesis testing tracks supporting and opposing evidence, source diversity, source quality,
citation readiness, primary-source signals, freshness signals, contradictions, unresolved gaps, and
confidence updates. Confidence is intentionally conservative: weak evidence, sensitive domains,
missing primary sources, unresolved contradictions, and strong unsupported wording all reduce the
posterior score. Temporal staleness, high-risk source safety findings, unsupported numeric claims,
and non-comparable quantitative comparisons also reduce confidence.

Generated artifacts include `hypotheses.json`, `hypotheses.md`, `hypothesis_tests.json`,
`hypothesis_tests.md`, `hypothesis_graph.json`, `hypothesis_graph.md`,
`confidence_updates.json`, and `confidence_updates.md`. These artifacts are decision support for
later synthesis and evaluation. They are not proof, and they do not replace human review for legal,
medical, financial, security, or other high-stakes conclusions.

## Research Evaluation Lab

The backend includes a backend-only Research Evaluation Lab under
`deep_research_agent.evaluation_lab`. It provides offline, deterministic regression benchmarks for
research-agent behavior: local cases, fixture source loading, `benchmark://` URL simulation,
mock-agent execution, artifact validation, expected-output checks, hallucination-risk checks,
citation checks, temporal checks, numeric checks, adversarial checks, scoring, reports, and API
endpoints.

Benchmark cases are versioned under `backend/benchmarks/cases/<case_id>/`:

- `case.json` defines the case id, title, category, difficulty, question, `benchmark://` URLs,
  local source fixtures, tags, traps, and scoring profile overrides.
- `expected.json` defines must-mention and must-not-mention phrases, expected entities, numbers,
  dates, claims, forbidden claims, required warnings, required artifacts, expected citation sources,
  and optional confidence bounds.
- `sources/` contains local Markdown, HTML, text, CSV, PDF, or DOCX fixtures. The evaluation lab
  computes content hashes and treats fixture text as untrusted evidence.

The initial corpus covers:

- `simple_factual`
- `framework_comparison`
- `prompt_injection_source`
- `stale_source_current_question`
- `contradictory_sources`
- `numeric_claims`
- `missing_primary_source`

The `benchmark://case_id/source_id` scheme is accepted only by the evaluation lab offline fetcher.
Normal `/run` behavior still uses the existing source fetcher and does not enable benchmark fixture
URLs unless explicitly configured for evaluation. Mock benchmark runs are deterministic, write the
guaranteed `plan.md`, `notes.md`, `sources.json`, and `report.md`, mark mock mode clearly, and do not
call OpenAI, Ollama, external search, or the network.

Scoring dimensions are deterministic heuristics: artifact integrity, answer relevance, expected
content coverage, source traceability, citation support, numeric accuracy, temporal handling,
contradiction handling, adversarial resistance, uncertainty handling, overclaiming control, and
required warning coverage. Scores are regression signals for engineering work, not semantic proof.
The checks are intentionally conservative around stale sources, missing primary sources, numeric
swaps, unsupported numbers or dates, unknown citations, prompt-injection leakage, and strong
recommendations from weak evidence.

API endpoints:

- `GET /evaluation-lab/cases?category=&tag=&difficulty=`
- `GET /evaluation-lab/cases/{case_id}`
- `POST /evaluation-lab/validate`
- `POST /evaluation-lab/run`
- `GET /evaluation-lab/runs/{run_id}`
- `GET /evaluation-lab/runs/{run_id}/summary?format=json|md`
- `POST /evaluation-lab/compare`
- `GET /evaluation-lab/profiles`

Example run request:

```json
{
  "case_ids": ["simple_factual", "numeric_claims"],
  "use_mock_agent": true,
  "use_offline_fetcher": true
}
```

Outputs are written under `backend/benchmark_runs/<run_id>/`, including `benchmark_run.json`,
`benchmark_run.md`, `evaluation_lab_summary.json`, `evaluation_lab_summary.md`, and per-case
`case_result.*`, `benchmark_*_checks.*`, and `benchmark_score.*` artifacts.

Known limitations: deterministic checks cannot fully judge semantic correctness, paraphrases may be
missed, and mock mode is for infrastructure/offline testing rather than production-quality research.
Live model behavior can differ from mock behavior, so these benchmarks should guide regression work
instead of being treated as absolute truth.

## Artifacts

Each run writes artifacts under `runs/<thread_id>/`. Core artifacts include:

- `plan.md`
- `notes.md`
- `sources.json`
- `report.md`
- `events.jsonl`
- `budget.json`

Depending on enabled features, runs can also include strategy, protocol, source discovery,
document profile, retrieval, context pack, memory, temporal profile, timeline, currentness,
temporal claim, quantitative profile, numeric claim, table/CSV profile, quantitative comparison,
source safety, sanitized source, evidence, hypothesis, verification, synthesis, evaluation,
advanced intelligence summary, provenance, replay, quality, and review artifacts.
Fetched source text and metadata live under
`runs/<thread_id>/sources/`.

## Review And Safety

Pass `"require_review": true` to `/run` to finish in `waiting_for_review`. Operators can approve,
request changes, or reject through review endpoints.

The backend is a research and traceability prototype, not professional advice. Legal, medical, and
financial outputs are informational only and should be reviewed by qualified humans before use.

## Repository Notes

- Python package: `backend/src/deep_research_agent`
- Runtime requirements: `backend/requirements.txt`
- Development requirements: `backend/requirements-dev.txt`
- Generated run data: `runs/` (ignored except `runs/.keep`)
