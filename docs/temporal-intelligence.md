# Temporal Intelligence

The backend temporal subsystem lives under `deep_research_agent.temporal`. It is deterministic,
offline-only, and does not call a model or the web.

It writes these artifacts for a run:

- `temporal_profile.json` / `temporal_profile.md`
- `timeline.json` / `timeline.md`
- `currentness_assessment.json` / `currentness_assessment.md`
- `temporal_claims.json`
- `temporal_warnings.md`

The subsystem extracts dates from source metadata, titles, URL paths, source text, `report.md`,
`notes.md`, and `sources.json`. Supported signals include ISO dates, month/day/year,
day/month/year, month/year, year-only references, release/version wording, published/updated/as-of
phrases, effective dates, deadlines, expiry dates, and access dates.

It also detects time-sensitive questions using terms such as latest, current, today, recent,
pricing, API docs, regulation, law, policy, market, benchmarks, versions, releases, model
capabilities, company facts, and security advisories.

Currentness is conservative. Access or fetch dates are recorded, but source currentness is based on
content dates when available: published, updated, effective, expiry, release, deadline, event, or
mentioned dates. If a source has no content date and the question needs freshness, the system warns
instead of assuming the source is current.

Backend endpoints:

- `POST /runs/{thread_id}/temporal/rebuild`
- `GET /runs/{thread_id}/timeline`
- `GET /runs/{thread_id}/currentness`
- `GET /runs/{thread_id}/temporal-warnings`

Limitations:

- Date parsing is heuristic and intentionally avoids aggressive interpretation of ambiguous dates.
- Version detection finds textual signals such as `v2`, semantic versions, changelogs, archived
  docs, legacy docs, beta/preview docs, and latest/current docs, but it does not compare against a
  live upstream release registry.
- Claim checks are deterministic support checks, not semantic proof. They flag stale-source risk,
  missing date support, and obvious date contradictions for human or agent caution.
