# Handoff Release Portfolio Receipt Verification

The handoff release portfolio receipt verification subsystem creates deterministic verification
reports for saved final portfolio receipts. It reads the saved portfolio receipt, checks
JSON/Markdown sidecar consistency, confirms receipt readiness, compares the current attestation
with the receipt hashes, checks attestation verification state, confirms the recipient-observed
attestation checksum, and verifies the global operator audit hash chain. It does not execute
research, call models, refetch sources, copy files, send data to recipients, regenerate receipts,
run tests, or build the project.

Generated artifacts:

- `runs/_handoff/handoff_release_portfolio_receipt_verification.json`
- `runs/_handoff/handoff_release_portfolio_receipt_verification.md`

API endpoints:

- `POST /runs/handoff-release-portfolio-receipt/verification`
- `GET /runs/handoff-release-portfolio-receipt/verification`
- `GET /runs/handoff-release-portfolio-receipt/verification/markdown`
- `POST /runs/handoff-release-portfolio-closeout`
- `GET /runs/handoff-release-portfolio-closeout`
- `GET /runs/handoff-release-portfolio-closeout/markdown`

Generating a report records `handoff.release_portfolio_receipt_verified` in the global operator
audit trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "require_receipt_artifacts": true,
  "require_receipt_recorded": true,
  "require_attestation_hash_match": true,
  "require_attestation_verification_valid": true,
  "require_recipient_checksum_match": true,
  "require_global_operator_audit": true,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-release-portfolio-receipt/verification \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator"}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the report and audit event.
- `require_receipt_artifacts`: fail when receipt sidecars are missing or inconsistent.
- `require_receipt_recorded`: fail unless the saved receipt readiness is `recorded`.
- `require_attestation_hash_match`: fail when current attestation differs from the receipt.
- `require_attestation_verification_valid`: fail when attestation verification is missing, drifted,
  or no longer `valid`.
- `require_recipient_checksum_match`: fail when the recipient checksum is missing or mismatched.
- `require_global_operator_audit`: fail when the global operator audit hash chain is invalid.
- `notes`: operator notes persisted into the verification report.

## Readiness

`readiness` is one of:

- `valid`: every required verification control passed.
- `warnings`: no required control failed, but at least one warning remains.
- `failed`: at least one required verification control failed.

## Verification Scope

The report verifies:

- `handoff_release_portfolio_receipt.json` and `.md` exist.
- The Markdown sidecar matches the saved JSON receipt content.
- The saved receipt readiness is `recorded`.
- Current `handoff_release_attestation.json` hash matches the saved receipt.
- Current attestation generation time matches the saved receipt.
- Current attestation verification exists, is `valid`, and matches the saved receipt timestamps.
- Attestation verification still references the same attestation hash as the receipt.
- Recipient-observed attestation SHA-256 matches the saved receipt hash.
- Global operator audit JSONL hash chain verifies.

## Operational Guidance

Run portfolio receipt verification after recording the portfolio receipt and again before closing
handoff custody or archiving the final handoff package. An attestation hash mismatch means the
certified portfolio changed after recipient acknowledgement. An attestation verification mismatch
means the proof used to accept the receipt changed after receipt generation. A recipient checksum
mismatch means the recipient did not acknowledge the same attestation JSON currently stored in the
portfolio.

The verification report is a local deterministic sidecar. It does not replace immutable artifact
storage, recipient-side checksum validation, legal review, centralized audit logging, or external
signature infrastructure. After verification is valid, generate the final portfolio closeout
documented in `docs/handoff-release-portfolio-closeout.md`.
