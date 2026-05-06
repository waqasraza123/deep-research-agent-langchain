# Run Handoff Manifests

The run handoff subsystem creates a final deterministic go/no-go manifest for a run package. It
does not execute research, call models, refetch sources, create exports, run tests, or build the
project. It reads existing lifecycle state and control artifacts, then writes one manifest that
summarizes whether the run is ready for handoff.

Generated artifacts:

- `handoff_manifest.json`: machine-readable gate statuses, blockers, warnings, and handoff metadata.
- `handoff_manifest.md`: operator-facing final handoff summary.

API endpoints:

- `POST /runs/{thread_id}/handoff`
- `GET /runs/{thread_id}/handoff`
- `GET /runs/{thread_id}/handoff/markdown`

Generating a manifest records `handoff.manifest_generated` in the operator audit trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "recipient": "",
  "purpose": "external_handoff",
  "require_review_approval": false,
  "require_export_bundle": true,
  "require_retention_policy": true,
  "require_custody_ready": true,
  "require_integrity_valid": true,
  "require_disclosure_clear": true,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/<thread_id>/handoff \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator",
       "recipient":"external-review",
       "purpose":"audit_handoff",
       "require_review_approval":true}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the manifest and audit event.
- `recipient`: free-form handoff recipient label.
- `purpose`: free-form handoff purpose label.
- `require_review_approval`: block handoff unless review status is approved.
- `require_export_bundle`: block handoff unless an export bundle exists and matches its manifest.
- `require_retention_policy`: block handoff unless retention policy exists.
- `require_custody_ready`: block handoff unless custody readiness is `ready`.
- `require_integrity_valid`: block handoff unless integrity readiness is `valid`.
- `require_disclosure_clear`: block handoff unless disclosure readiness is `clear`.
- `notes`: operator notes persisted into the manifest.

## Readiness

`readiness` is one of:

- `ready_for_handoff`: all required gates passed.
- `needs_attention`: no blocking gate failed, but one or more warnings remain.
- `blocked`: at least one required gate failed.

## Gates

The manifest evaluates:

- Run state: failed and cancelled runs block handoff; waiting-for-review warns.
- Review approval: approval is required only when `require_review_approval=true`; rejected runs
  always block handoff.
- Retention policy: checks that `retention_policy.*` exists when required.
- Export bundle: verifies `exports/run_export.zip` against `exports/export_manifest.json`.
- Custody certificate: reads `custody_certificate.*` and enforces readiness.
- Integrity report: reads `integrity_report.*` and enforces readiness.
- Disclosure report: reads `disclosure_report.*` and enforces readiness.
- Operator audit: verifies global and per-run audit hash chains.

## Operational Guidance

Generate the handoff manifest only after the upstream controls have been refreshed in the intended
order: review dossier, review decision when required, retention policy, export bundle, custody
certificate, integrity report, and disclosure report. If the manifest is blocked, inspect the gate
metadata in `handoff_manifest.json`, fix the upstream control, regenerate that control, then
regenerate the handoff manifest.

For external packages that must include the final handoff manifest, regenerate the export bundle
after handoff generation. The manifest is a deterministic local operator artifact and does not
replace legal review, data classification, immutable storage, or centralized audit logging.
