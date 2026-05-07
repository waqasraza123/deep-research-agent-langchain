# Handoff Release Portfolio Closeout Verification

The handoff release portfolio closeout verification subsystem creates deterministic verification
reports for saved final closeout manifests. It reads the saved closeout, checks JSON/Markdown
sidecar consistency, confirms closeout readiness, re-hashes every final artifact recorded by the
closeout, verifies upstream custody artifact timestamps still match, and verifies the global
operator audit hash chain. It does not execute research, call models, refetch sources, copy files,
send data to recipients, regenerate closeout manifests, run tests, or build the project.

Generated artifacts:

- `runs/_handoff/handoff_release_portfolio_closeout_verification.json`
- `runs/_handoff/handoff_release_portfolio_closeout_verification.md`

API endpoints:

- `POST /runs/handoff-release-portfolio-closeout/verification`
- `GET /runs/handoff-release-portfolio-closeout/verification`
- `GET /runs/handoff-release-portfolio-closeout/verification/markdown`

Generating a report records `handoff.release_portfolio_closeout_verified` in the global operator
audit trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "require_closeout_artifacts": true,
  "require_closeout_closed": true,
  "require_final_artifact_hashes": true,
  "require_upstream_timestamps_match": true,
  "require_global_operator_audit": true,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-release-portfolio-closeout/verification \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator"}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the report and audit event.
- `require_closeout_artifacts`: fail when closeout sidecars are missing or inconsistent.
- `require_closeout_closed`: fail unless the saved closeout readiness is `closed`.
- `require_final_artifact_hashes`: fail when any artifact hash recorded by closeout has drifted.
- `require_upstream_timestamps_match`: fail when upstream custody artifacts changed after closeout.
- `require_global_operator_audit`: fail when the global operator audit hash chain is invalid.
- `notes`: operator notes persisted into the verification report.

## Readiness

`readiness` is one of:

- `valid`: every required verification control passed.
- `warnings`: no required control failed, but at least one warning remains.
- `failed`: at least one required verification control failed.

## Verification Scope

The report verifies:

- `handoff_release_portfolio_closeout.json` and `.md` exist.
- The Markdown sidecar matches the saved JSON closeout content.
- The saved closeout readiness is `closed`.
- Every final artifact hash recorded by the closeout still matches current disk state.
- Missing, newly unsafe, changed, and newly present-but-unhashed artifacts are reported.
- Upstream ledger, attestation, receipt, and verification timestamps match the saved closeout.
- Global operator audit JSONL hash chain verifies.

## Operational Guidance

Run closeout verification after generating the closeout and again before archiving the final
portfolio custody package. A final artifact hash mismatch means a closure artifact changed after
closeout. An upstream timestamp mismatch means a custody control was regenerated after closeout and
the closure manifest should be regenerated. Treat `failed` as a stop condition for relying on the
closeout package.

The verification report is a local deterministic sidecar. It does not replace immutable artifact
storage, recipient-side checksum validation, legal review, centralized audit logging, or external
signature infrastructure.
