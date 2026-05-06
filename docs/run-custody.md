# Run Custody Certificates

The run custody subsystem creates deterministic handoff certificates from existing run artifacts.
It does not execute research, call models, refetch sources, create export bundles, or run tests. Its
purpose is to give operators one file pair that states whether a run is ready for custody transfer,
archive, external review, or incident follow-up.

Generated artifacts:

- `custody_certificate.json`: machine-readable checks, readiness, artifact hashes, and metadata.
- `custody_certificate.md`: operator-facing custody summary.

API endpoints:

- `POST /runs/{thread_id}/custody`
- `GET /runs/{thread_id}/custody`
- `GET /runs/{thread_id}/custody/markdown`

Generating a certificate also records `custody.certificate_generated` in the operator audit trail.
The certificate excludes `custody_certificate.*` and `operator_audit.*` from its artifact hash
inventory because those files are dynamic control artifacts that change when custody generation
itself is recorded.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "require_review_approval": false,
  "require_export_bundle": false,
  "require_retention_policy": false,
  "require_operator_audit": false,
  "require_provenance": true,
  "include_artifact_hashes": true,
  "max_artifacts": 1000,
  "notes": ""
}
```

Example strict handoff:

```bash
curl http://localhost:8000/runs/<thread_id>/custody \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator",
       "require_review_approval":true,
       "require_export_bundle":true,
       "require_retention_policy":true,
       "require_operator_audit":true,
       "require_provenance":true}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the certificate and audit event.
- `require_review_approval`: block readiness unless the run review status is approved.
- `require_export_bundle`: block readiness unless `exports/run_export.zip` exists and matches
  `exports/export_manifest.json`.
- `require_retention_policy`: block readiness unless `retention_policy.json` exists.
- `require_operator_audit`: block readiness unless per-run audit events exist and verify.
- `require_provenance`: block readiness unless provenance artifacts are present.
- `include_artifact_hashes`: include SHA-256 inventory for stable run artifacts.
- `max_artifacts`: cap the artifact hash inventory for very large run directories.
- `notes`: operator notes persisted into the certificate.

## Readiness

`readiness` is one of:

- `ready`: all checks passed.
- `needs_attention`: no blocking failures, but at least one warning remains.
- `blocked`: at least one blocking check failed.

Warnings are intended for optional-but-recommended controls, such as missing review approval when
approval is not required by the request. Blockers represent failed required controls, failed export
hash validation, invalid audit chains, rejected reviews, failed runs, cancelled runs, or missing
required core artifacts.

## Checks

The certificate evaluates:

- Required artifacts: `plan.md`, `notes.md`, `sources.json`, and `report.md`.
- Run state: lifecycle status, terminal errors, and review-waiting state.
- Review approval: current review status and reviewer identity when present.
- Provenance: `artifact_manifest.json`, `artifact_dependency_dag.json`,
  `reproducibility_report.json`, and `replay_plan.json`.
- Retention policy: retention class, hold state, retain-until, and delete-after fields.
- Export bundle: archive existence and archive SHA-256 match against export manifest.
- Operator audit: global and per-run hash-chain verification.
- Artifact inventory: stable artifact count, total bytes, and SHA-256 records.

## Operational Guidance

Generate custody after the run has been reviewed, exported, and protected by retention policy when
those controls are required. Use the Markdown file for quick operator review and the JSON file for
automation. For final external handoff, create or refresh the export bundle after custody generation
if the recipient must receive `custody_certificate.*` inside the archive.

The certificate is a local deterministic readiness artifact. It does not replace centralized audit
logging, durable storage, access controls, or qualified human review for legal, medical, financial,
security, or other high-stakes conclusions.

After custody generation, use `POST /runs/{thread_id}/integrity` when you need to check whether
the current artifact directory still matches the provenance manifest, custody hash inventory, export
archive manifest, and operator-audit chains.

Before external disclosure, use `POST /runs/{thread_id}/disclosure` to scan for likely secrets,
raw source exposure, raw-source export settings, and oversized text artifacts that were not scanned.
