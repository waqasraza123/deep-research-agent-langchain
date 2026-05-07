# Handoff Release Portfolio Receipt

The handoff release portfolio receipt subsystem records a deterministic repository-level recipient
acknowledgement for the final handoff release portfolio. It reads the saved portfolio attestation,
the saved attestation verification report, the current attestation JSON hash, optional
recipient-observed attestation checksum, and global operator audit state, then writes receipt
artifacts under `runs/_handoff/`. It does not execute research, call models, refetch sources, copy
files, send data to recipients, regenerate attestations, run tests, or build the project.

Generated artifacts:

- `runs/_handoff/handoff_release_portfolio_receipt.json`
- `runs/_handoff/handoff_release_portfolio_receipt.md`

API endpoints:

- `POST /runs/handoff-release-portfolio-receipt`
- `GET /runs/handoff-release-portfolio-receipt`
- `GET /runs/handoff-release-portfolio-receipt/markdown`
- `POST /runs/handoff-release-portfolio-receipt/verification`
- `GET /runs/handoff-release-portfolio-receipt/verification`
- `GET /runs/handoff-release-portfolio-receipt/verification/markdown`
- `POST /runs/handoff-release-portfolio-closeout`
- `GET /runs/handoff-release-portfolio-closeout`
- `GET /runs/handoff-release-portfolio-closeout/markdown`

Generating a receipt records `handoff.release_portfolio_receipt_recorded` in the global operator
audit trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "recipient": "",
  "recipient_contact": "",
  "transfer_method": "",
  "transfer_reference": "",
  "transferred_at": null,
  "received_by": "",
  "received_at": null,
  "recipient_attestation_sha256": null,
  "outcome": "accepted",
  "require_attestation_ready": true,
  "require_attestation_verification_valid": true,
  "require_attestation_hash_match": true,
  "require_recipient_checksum_match": true,
  "require_global_operator_audit": true,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-release-portfolio-receipt \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator",
       "recipient":"external-review",
       "recipient_contact":"review-team@example.com",
       "transfer_method":"secure_object_storage",
       "transfer_reference":"s3://handoffs/final-portfolio/",
       "received_by":"review-team",
       "received_at":"2026-05-07T15:00:00Z",
       "recipient_attestation_sha256":"<sha256-from-recipient>"}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the receipt and audit event.
- `recipient`: recipient organization, team, system, or case label.
- `recipient_contact`: optional person, team address, ticket, or system identifier.
- `transfer_method`: transfer channel such as secure object storage, encrypted drive, internal
  ticket, or vault reference.
- `transfer_reference`: durable transfer locator, ticket, storage key, evidence ID, or case number.
- `transferred_at`: transfer time. When omitted, receipt generation time is used.
- `received_by`: recipient-side identity that acknowledged receipt.
- `received_at`: recipient-side receipt timestamp when available.
- `recipient_attestation_sha256`: SHA-256 of `handoff_release_attestation.json` observed by the
  recipient.
- `outcome`: `accepted`, `accepted_with_exceptions`, `rejected`, or `pending`.
- `require_attestation_ready`: block when the saved attestation readiness is not `attested`.
- `require_attestation_verification_valid`: block when attestation verification is not `valid`.
- `require_attestation_hash_match`: block when the current attestation hash differs from the
  attestation verification report.
- `require_recipient_checksum_match`: block when recipient checksum is missing or mismatched.
- `require_global_operator_audit`: block when the global operator audit hash chain is invalid.
- `notes`: operator notes persisted into the receipt.

## Readiness

`readiness` is one of:

- `recorded`: all required receipt controls passed.
- `warnings`: no required control failed, but metadata or recipient outcome needs attention.
- `blocked`: at least one required attestation, checksum, acceptance, or audit control failed.

## Receipt Checks

The receipt evaluates:

- Attestation readiness from `handoff_release_attestation.json`.
- Attestation verification readiness from `handoff_release_attestation_verification.json`.
- Current attestation JSON SHA-256 against the attestation verification report.
- Recipient-observed attestation SHA-256 against the local attestation.
- Transfer metadata completeness for recipient, method, and reference.
- Recipient outcome, with rejected outcomes blocking receipt reliance.
- Global operator audit JSONL hash-chain verification.

## Operational Guidance

Generate this receipt after the attestation is generated, attestation verification is valid, the
portfolio has been transferred or made available to the recipient, and the recipient has confirmed
the attestation JSON checksum. For strict external transfer, keep all required controls enabled and
include `recipient_attestation_sha256`. If the recipient checksum differs, do not close custody:
restore or recreate the portfolio, rerun attestation verification, and record a new portfolio
receipt only after the recipient confirms the corrected hash.

Portfolio receipts are local deterministic custody artifacts. They do not move files, prove legal
acceptance, replace recipient-side validation, or provide immutable storage. Store the receipt with
the attestation, attestation verification report, release ledger, ledger verification report, and
global operator audit export for the final portfolio handoff record. After recording the receipt,
create the verification report documented in
`docs/handoff-release-portfolio-receipt-verification.md`, then generate the closeout manifest in
`docs/handoff-release-portfolio-closeout.md`.
