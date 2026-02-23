# Deep Research Agent (Deep Agents + LangGraph)

A prodution-grade research agent that:

- Creates a plan
- Reads sources you provide
- Produces artifacts (plan, notes, sources.json, report.md)
- Keeps everything traceable and easy to review

## Outputs

Each run produces:

- runs/<thread_id>/plan.md
- runs/<thread_id>/notes.md
- runs/<thread_id>/sources.json
- runs/<thread_id>/report.md

## Quickstart (free, local)

1. Install and run Ollama:

- brew install ollama
- ollama serve
- ollama pull llama3.1

2. Run backend:

- python3 -m venv backend/.venv
- source backend/.venv/bin/activate
- pip install -r backend/requirements.txt
- uvicorn backend.app.main:app --reload --port 8000

3. Test:
   POST /run with a question + URLs

## API

- GET /health
- POST /run
- GET /threads/{thread_id}/artifacts
- GET /threads/{thread_id}/artifacts/{path}

## Notes

- runs/ is not committed
- later swap model provider to Anthropic in one place
