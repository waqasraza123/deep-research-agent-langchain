# Research Evaluation Lab Benchmarks

This directory contains offline, versionable benchmark cases for the backend-only Research Evaluation Lab.

Each case lives under `cases/<case_id>/` and contains:

- `case.json`: metadata, question, `benchmark://` URLs, local source mapping, traps, tags, and scoring profile overrides.
- `expected.json`: deterministic expectations for artifacts, mentions, warnings, numbers, claims, citations, and confidence bounds.
- `sources/`: local fixture documents used by the offline fetcher.

The `benchmark://` scheme is only valid inside evaluation lab mode. Normal `/run` source fetching still accepts only HTTP(S) URLs.

Mock benchmark runs are deterministic and require no OpenAI, Ollama, external search, or network access. Scores are heuristic regression signals, not absolute semantic truth.
