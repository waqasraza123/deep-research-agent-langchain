````md
# Deep Research Agent (Deep Agents + LangGraph)

A production-grade research agent that:

- Creates a plan
- Reads sources you provide
- Produces artifacts (plan, notes, sources.json, report.md)
- Stores fetched sources under runs/<thread_id>/sources
- Keeps everything traceable and easy to review

## Outputs

Each run produces:

- runs/<thread_id>/plan.md
- runs/<thread_id>/notes.md
- runs/<thread_id>/sources.json
- runs/<thread_id>/report.md
- runs/<thread_id>/sources/\*.txt
- runs/<thread_id>/sources/\*.json

## Requirements

- Python 3.11+

## Quickstart (OpenAI, recommended)

1. Create `.env` in the repo root:

```bash
MODEL_PROVIDER=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TIMEOUT_S=60
OPENAI_MAX_RETRIES=1
OPENAI_MAX_TOKENS=350

MAX_PAGE_CHARS=15000
HTTP_TIMEOUT_S=20
```
````

2. Install and run backend:

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
pip install -e backend
uvicorn deep_research_agent.api:app --reload --port 8000
```

3. Test:

```bash
curl http://localhost:8000/run \
  -H 'content-type: application/json' \
  -d '{
    "question":"Summarize Deep Agents in 6 bullets and 1-line conclusion.",
    "urls":["https://docs.langchain.com/oss/python/deepagents/overview"],
    "max_sources": 1,
    "max_links_per_source": 0,
    "follow_links": false
  }'
```

Then open:

- runs/<thread_id>/report.md
- runs/<thread_id>/sources.json

## Quickstart (optional local: Ollama)

1. Run Ollama:

```bash
brew install ollama
ollama serve
ollama pull llama3.1
```

2. Set `.env`:

```bash
MODEL_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
OLLAMA_NUM_PREDICT=220

MAX_PAGE_CHARS=15000
HTTP_TIMEOUT_S=20
```

3. Run backend:

```bash
source backend/.venv/bin/activate
uvicorn deep_research_agent.api:app --reload --port 8000
```

## API

- GET /health
- POST /run
- GET /threads/{thread_id}/artifacts
- GET /threads/{thread_id}/artifacts/{path}

## Notes

- runs/ is not committed (except runs/.keep)
- For quick validation runs: max_sources=1 and follow_links=false
- Model availability depends on your account and verification status
