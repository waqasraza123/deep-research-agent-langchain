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
5. Write `sources.json`, `source_graph.json`, `source_graph.md`, and source quality metadata.
6. Run the model/agent layer, or deterministic mock mode.
7. Guarantee `plan.md`, `notes.md`, `sources.json`, and `report.md` exist.
8. Build evidence artifacts from notes, report, and captured sources.
9. Persist `events.jsonl`, `events.md`, and `budget.json`.
10. Mark the run completed, failed, cancelled, or waiting for review.

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
- `source_graph.json`
- `source_graph.md`
- `evidence_ledger.json`
- `evidence_ledger.md`
- `unsupported_claims.md`
- `contradictions.md`
- `citation_map.json`
- `evidence_coverage.json`
- `events.jsonl`
- `events.md`
- `budget.json`

Fetched source text and metadata live under `runs/<thread_id>/sources/`.

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

## Evidence Ledger

The evidence layer is deterministic and offline. It extracts candidate claims from `notes.md`,
`report.md`, and source text, maps citations using lexical/value overlap, scores confidence,
and flags unsupported claims or possible contradictions.

Limitations:

- It is a review aid, not proof of truth.
- Weak citation matches can be false positives.
- Broad or stale claims still need human review.
- Contradiction detection is heuristic.

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

## API

Useful endpoints:

- `GET /health`
- `GET /models`
- `GET /runtime/diagnostics`
- `POST /research-plan`
- `POST /plan`
- `POST /run`
- `GET /runs`
- `GET /runs/{thread_id}`
- `GET /runs/{thread_id}/artifacts`
- `GET /runs/{thread_id}/artifacts/{artifact_name}`
- `GET /runs/{thread_id}/events`
- `GET /runs/{thread_id}/budget`
- `POST /runs/{thread_id}/evidence/rebuild`
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
