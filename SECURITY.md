# Security Policy

## Reporting a Vulnerability
Please do not open public issues for security reports.

Email: security@yourdomain.com (replace this)
Include:
- What you found
- Steps to reproduce
- Impact
- Any suggested fix

We will acknowledge receipt within 72 hours.

## Untrusted Source Handling

Fetched webpages and documents are treated as untrusted evidence, never as instructions. The
backend assesses every captured source for prompt-injection patterns, source-poisoning signals,
metadata mismatches, hidden-instruction hints, suspicious citation behavior, and extraction
anomalies before source text is placed into agent or retrieval context.

Source safety artifacts are written per run:

- `source_safety.json` / `source_safety.md`
- `prompt_injection_findings.json` / `prompt_injection_findings.md`
- `source_poisoning_findings.json` / `source_poisoning_findings.md`
- `sanitized_sources.json`
- `trust_boundary_policy.md`

Raw source text remains traceable on disk, but downstream agent context prefers sanitized source
artifacts. Critical-risk sources are excluded from agent context by default. High-risk sources are
quoted or withheld according to the deterministic risk score, and the manifest records what was
excluded and why.
