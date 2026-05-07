# Handoff Release Ledger Verification

The handoff release ledger verification subsystem creates deterministic verification reports for
repository-level release custody ledgers. It reads the saved ledger, recomputes a current in-memory
ledger snapshot with the saved required controls, compares summaries and per-release custody
signals, verifies referenced artifacts still exist, checks JSON/Markdown sidecar consistency, and
verifies the global operator audit hash chain. It does not execute research, call models, refetch
sources, create exports, create bundles, record receipts, regenerate the ledger, run tests, or
build the project.

Generated artifacts:

- `runs/_handoff/handoff_release_ledger_verification.json`
- `runs/_handoff/handoff_release_ledger_verification.md`

API endpoints:

- `POST /runs/handoff-release-ledger/verification`
- `GET /runs/handoff-release-ledger/verification`
- `GET /runs/handoff-release-ledger/verification/markdown`

Generating a report records `handoff.release_ledger_verified` in the global operator audit trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "require_ledger_artifacts": true,
  "require_snapshot_match": true,
  "require_artifact_presence": true,
  "require_global_operator_audit": true,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-release-ledger/verification \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator"}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the report and audit event.
- `require_ledger_artifacts`: fail when ledger JSON/Markdown sidecars are missing or inconsistent.
- `require_snapshot_match`: fail when current release custody state differs from the saved ledger.
- `require_artifact_presence`: fail when artifacts referenced by ledger items are missing.
- `require_global_operator_audit`: fail when the global operator audit hash chain is invalid.
- `notes`: operator notes persisted into the verification report.

## Readiness

`readiness` is one of:

- `valid`: every required verification control passed.
- `warnings`: no required control failed, but at least one warning remains.
- `failed`: at least one required verification control failed.

## Verification Scope

The report verifies:

- `handoff_release_ledger.json` and `.md` exist.
- The Markdown sidecar matches the saved JSON ledger content.
- Current in-memory ledger summary matches the saved ledger summary.
- Each saved release item still exists in the current release custody snapshot.
- Current release readiness, verification, bundle, receipt, checksum, blockers, and warnings match
  the saved ledger item.
- New current releases are not missing from the saved ledger.
- Artifacts referenced by saved ledger items still exist under `runs/`.
- Global operator audit JSONL hash chain verifies.

## Operational Guidance

Run ledger verification after generating the ledger and again before using it as a portfolio-level
custody record. A snapshot mismatch usually means a release manifest, bundle, verification report,
receipt, or audit state changed after the ledger was generated. Review the drift, regenerate the
underlying control when needed, regenerate the ledger, then rerun verification.

The verification report is a local deterministic sidecar. It does not replace immutable storage,
recipient-side checksum validation, legal review, or centralized audit logging.
