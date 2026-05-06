# Run Export Bundles

The run export subsystem creates backend-only zip bundles for audit handoff, archival, and
reproducibility review. It does not execute research, refetch sources, or call models. It packages
existing artifacts from `runs/<thread_id>/` and writes export metadata back under
`runs/<thread_id>/exports/`.

Generated artifacts:

- `exports/run_export.zip`: zip archive containing selected run artifacts under `artifacts/` plus
  an embedded export manifest.
- `exports/export_manifest.json`: sidecar manifest with final archive size and archive SHA-256.
- `exports/export_manifest.md`: human-readable export summary.

API endpoints:

- `POST /runs/{thread_id}/export`
- `GET /runs/{thread_id}/export`
- `GET /runs/{thread_id}/export/download`

## Request

Default request:

```json
{
  "profile": "audit",
  "include_raw_sources": false,
  "include_internal": false,
  "redact": true,
  "max_total_bytes": 100000000,
  "max_file_bytes": 25000000
}
```

Fields:

- `profile`: `public`, `audit`, or `full`.
- `include_raw_sources`: include files below `sources/` and `sanitized_sources/`.
- `include_internal`: include internal registry/control files such as `.run.json`.
- `redact`: redact likely secrets in text and JSON payloads before writing them into the zip.
- `include_patterns`: explicit glob patterns. When supplied, these replace profile defaults.
- `exclude_patterns`: extra glob patterns applied after built-in export exclusions.
- `max_total_bytes`: source-byte budget considered for the export.
- `max_file_bytes`: per-file source-byte limit.
- `notes`: operator note persisted into the export manifest.

Example:

```bash
curl http://localhost:8000/runs/<thread_id>/export \
  -H 'content-type: application/json' \
  -d '{"profile":"audit","include_raw_sources":false,"redact":true}'
```

Then download:

```bash
curl -o run-export.zip http://localhost:8000/runs/<thread_id>/export/download
```

## Profiles

`public` includes the user-facing report, core run metadata, provenance, reproducibility, replay,
quality, evidence, verification, source-audit, source-safety, temporal, quantitative, hypothesis,
synthesis, and summary artifacts. It is meant for a compact operator handoff and excludes raw source
payloads by default.

`audit` includes all non-internal run artifacts except export artifacts and raw source payloads,
unless `include_raw_sources=true`. This is the default because it captures the decision trail while
reducing accidental source-text exposure.

`full` includes every non-internal artifact except built-in excluded files. Use it with
`include_raw_sources=true` only when the receiving system is allowed to store captured source text
and sanitized-source evidence.

## Redaction

Redaction applies only to bytes written into the zip. It does not mutate the original run artifacts.
JSON files are parsed and passed through the same recursive secret redactor used by provenance.
Text-like files also redact common token forms such as OpenAI-style keys, GitHub tokens, bearer
tokens, and obvious `api_key`, `token`, `secret`, `password`, or `authorization` assignments.

Binary files are copied unchanged when selected. Keep binary or raw-source exports behind explicit
operator review when the run may contain proprietary, licensed, or sensitive input data.

## Manifest Semantics

Every exported artifact records:

- source path and archive path,
- original size and exported size,
- original SHA-256,
- exported SHA-256 after redaction,
- whether the payload was redacted.

Skipped artifacts record a reason, such as raw source exclusion, profile mismatch, explicit exclude
pattern, per-file byte limit, or total byte budget. The sidecar `exports/export_manifest.json`
contains the final zip archive SHA-256. The manifest embedded inside the zip is written before the
archive hash exists, so use the sidecar manifest as the final archive-integrity source.

## Operational Guidance

Use `profile=public` for external review packages, `profile=audit` for internal traceability, and
`profile=full` for incident response or offline reproduction when the recipient is trusted to
receive source text. Keep `redact=true` unless you are deliberately exporting exact bytes for a
controlled forensic workflow.
