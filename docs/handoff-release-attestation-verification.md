# Handoff Release Attestation Verification

The handoff release attestation verification subsystem creates deterministic verification reports
for saved portfolio attestations. It reads the saved attestation, checks JSON/Markdown sidecar
consistency, verifies the attestation is ready for reliance, compares current ledger and ledger
verification hashes with the saved attestation, re-hashes every attested artifact, and verifies the
global operator audit hash chain. It does not execute research, call models, refetch sources,
create exports, create bundles, record receipts, regenerate ledgers, regenerate attestations, run
tests, or build the project.

Generated artifacts:

- `runs/_handoff/handoff_release_attestation_verification.json`
- `runs/_handoff/handoff_release_attestation_verification.md`

API endpoints:

- `POST /runs/handoff-release-attestation/verification`
- `GET /runs/handoff-release-attestation/verification`
- `GET /runs/handoff-release-attestation/verification/markdown`
- `POST /runs/handoff-release-portfolio-receipt`
- `GET /runs/handoff-release-portfolio-receipt`
- `GET /runs/handoff-release-portfolio-receipt/markdown`
- `POST /runs/handoff-release-portfolio-receipt/verification`
- `GET /runs/handoff-release-portfolio-receipt/verification`
- `GET /runs/handoff-release-portfolio-receipt/verification/markdown`

Generating a report records `handoff.release_attestation_verified` in the global operator audit
trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "require_attestation_artifacts": true,
  "require_attestation_ready": true,
  "require_control_hashes": true,
  "require_artifact_hashes": true,
  "require_global_operator_audit": true,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-release-attestation/verification \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator"}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the report and audit event.
- `require_attestation_artifacts`: fail when attestation sidecars are missing or inconsistent.
- `require_attestation_ready`: fail unless the saved attestation readiness is `attested`.
- `require_control_hashes`: fail when current ledger controls differ from saved attestation hashes.
- `require_artifact_hashes`: fail when any attested artifact is missing, unsafe, or changed.
- `require_global_operator_audit`: fail when the global operator audit hash chain is invalid.
- `notes`: operator notes persisted into the verification report.

## Readiness

`readiness` is one of:

- `valid`: every required verification control passed.
- `warnings`: no required control failed, but at least one warning remains.
- `failed`: at least one required verification control failed.

## Verification Scope

The report verifies:

- `handoff_release_attestation.json` and `.md` exist.
- The Markdown sidecar matches the saved JSON attestation content.
- The saved attestation readiness is `attested`.
- Current `handoff_release_ledger.json` hash matches the saved attestation.
- Current `handoff_release_ledger_verification.json` hash matches the saved attestation.
- Current ledger and ledger verification generation timestamps match the saved attestation.
- Every artifact hash recorded by the attestation still matches current disk state.
- Missing, newly present, changed, and path-unsafe artifacts are reported explicitly.
- Global operator audit JSONL hash chain verifies.

## Operational Guidance

Run attestation verification after generating the attestation and again immediately before relying
on the portfolio for transfer, archive, or recipient acceptance. A control hash mismatch means the
ledger or ledger verification changed after attestation generation. An artifact hash mismatch means
the release portfolio on disk no longer matches the certified custody package. Review the drift,
restore the expected artifact or regenerate the upstream control, regenerate the attestation, then
rerun verification.

The verification report is a local deterministic sidecar. It does not replace immutable artifact
storage, recipient-side checksum validation, legal review, centralized audit logging, or external
signature infrastructure. After verification is valid and the recipient confirms the portfolio
checksum, record the portfolio receipt documented in
`docs/handoff-release-portfolio-receipt.md`, then verify the receipt with
`docs/handoff-release-portfolio-receipt-verification.md`.
