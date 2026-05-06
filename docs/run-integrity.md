# Run Integrity Reports

The run integrity subsystem creates deterministic artifact-drift reports for existing run
directories. It does not execute research, call models, refetch sources, create exports, run tests,
or build the project. It verifies local files against the audit and custody controls already written
under `runs/<thread_id>/`.

Generated artifacts:

- `integrity_report.json`: machine-readable findings, readiness, artifact hashes, and metadata.
- `integrity_report.md`: operator-facing integrity summary.

API endpoints:

- `POST /runs/{thread_id}/integrity`
- `GET /runs/{thread_id}/integrity`
- `GET /runs/{thread_id}/integrity/markdown`

Generating a report records `integrity.report_generated` in the operator audit trail. The report
excludes `operator_audit.*`, `custody_certificate.*`, and `integrity_report.*` from its stable
artifact inventory because those files are control artifacts that can change as verification is
performed.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "require_provenance_manifest": false,
  "require_custody_certificate": false,
  "require_export_bundle": false,
  "include_artifact_hashes": true,
  "max_artifacts": 1000,
  "notes": ""
}
```

Example strict verification:

```bash
curl http://localhost:8000/runs/<thread_id>/integrity \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator",
       "require_provenance_manifest":true,
       "require_custody_certificate":true,
       "require_export_bundle":true}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the report and audit event.
- `require_provenance_manifest`: fail when `artifact_manifest.json` is missing.
- `require_custody_certificate`: fail when `custody_certificate.json` is missing.
- `require_export_bundle`: fail when `exports/run_export.zip` or its manifest is missing.
- `include_artifact_hashes`: include a current SHA-256 inventory for stable run artifacts.
- `max_artifacts`: cap the current artifact hash inventory for large run directories.
- `notes`: operator notes persisted into the report.

## Readiness

`readiness` is one of:

- `valid`: all checks passed.
- `warnings`: no failed checks, but one or more warnings remain.
- `failed`: at least one integrity check failed.

Warnings are used for optional controls that are missing or incomplete. Failures are used for
required missing controls, missing required artifacts, invalid audit chains, export archive hash
mismatches, provenance manifest drift, or custody inventory drift.

## Checks

The report evaluates:

- Required artifacts: `plan.md`, `notes.md`, `sources.json`, and `report.md`.
- Provenance manifest: every stable `artifact_manifest.json` entry still exists and has the same
  SHA-256 content hash.
- Custody certificate: every artifact recorded in `custody_certificate.json` still exists and has
  the same SHA-256 content hash.
- Export bundle: `exports/run_export.zip` exists and matches `exports/export_manifest.json`.
- Operator audit: global and per-run operator audit JSONL hash chains verify.
- Current artifact inventory: stable artifact count, total bytes, and current SHA-256 records.

## Operational Guidance

Generate an integrity report after custody generation, before long-term archive, after restoring a
run from storage, or before relying on an old export. If the report fails, inspect the failed
finding metadata in `integrity_report.json` before regenerating provenance, custody, or export
artifacts. Regeneration can be appropriate after intentional changes, but it should not be used to
hide unexplained drift.

The report is a local deterministic verification artifact. It does not replace centralized logging,
immutable storage, external checksums, access controls, or qualified human review for high-stakes
research outputs.
