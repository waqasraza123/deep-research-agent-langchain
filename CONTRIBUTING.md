# Contributing

## Dev setup

    cp .env.example .env
    python -m venv .venv
    source .venv/bin/activate
    pip install -r backend/requirements.txt
    pytest

## Guidelines
- Keep PRs small and focused
- Add tests for logic changes
- No secrets in commits
