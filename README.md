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
- Model provider support for `openai`, `ollama`, `llamacpp`, and deterministic `mock`.
- Safety defaults for bounded source fetching, one-hop optional link expansion, URL validation,
  budget tracking, and high-stakes review recommendations.

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
DOCUMENT_INTELLIGENCE_ENABLED=true
RETRIEVAL_ENABLED=true
EMBEDDING_PROVIDER=disabled
VERIFICATION_ENABLED=true
SYNTHESIS_ENABLED=true
EVALUATION_ENABLED=true
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
- `POST /runs/diff`
- `GET /runs/{thread_id}/events`
- `GET /runs/{thread_id}/budget`
- `POST /runs/{thread_id}/cancel`
- `GET /runs/{thread_id}/review`
- `POST /runs/{thread_id}/review/approve`
- `POST /runs/{thread_id}/review/request-changes`
- `POST /runs/{thread_id}/review/reject`

Specialized endpoints also exist for protocols, source discovery, document intelligence,
retrieval, memory, temporal intelligence, quantitative intelligence, evidence, hypotheses, verification, synthesis,
evaluation, benchmarks, and quality scores.

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
posterior score.

Generated artifacts include `hypotheses.json`, `hypotheses.md`, `hypothesis_tests.json`,
`hypothesis_tests.md`, `hypothesis_graph.json`, `hypothesis_graph.md`,
`confidence_updates.json`, and `confidence_updates.md`. These artifacts are decision support for
later synthesis and evaluation. They are not proof, and they do not replace human review for legal,
medical, financial, security, or other high-stakes conclusions.

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
evidence, hypothesis, verification, synthesis, evaluation, quality, and review artifacts.
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
