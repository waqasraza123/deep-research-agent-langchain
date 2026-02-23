# Deep Research Agent

A production-minded deep research agent scaffold:
- Reproducible runs saved under runs/
- Clear docs in docs/
- Backend code in backend/
- CI for lint, typecheck, tests

## Quickstart

    cp .env.example .env
    python -m venv .venv
    source .venv/bin/activate
    pip install -r backend/requirements.txt
    pytest

## Repo layout
- backend/ core code
- docs/ documentation
- runs/ artifacts, logs, outputs (keep minimal in git)

## Notes
- Never commit secrets. Use .env locally.
