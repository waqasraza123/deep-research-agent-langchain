# Handoff Release Receipts

The handoff release receipt subsystem records a deterministic transfer receipt for a portable
handoff release bundle. It reads the release manifest, bundle manifest, bundle verification report,
current bundle archive hash, optional recipient checksum, and global operator audit state, then
writes receipt artifacts beside the release package. It does not execute research, call models,
copy files, send data to recipients, create bundles, run tests, or build the project.

Generated artifacts:

- `runs/_handoff/releases/<release_id>/handoff_release_receipt.json`
- `runs/_handoff/releases/<release_id>/handoff_release_receipt.md`

API endpoints:

- `POST /runs/handoff-releases/{release_id}/receipt`
- `GET /runs/handoff-releases/{release_id}/receipt`
- `GET /runs/handoff-releases/{release_id}/receipt/markdown`

Generating a receipt records `handoff.release_receipt_recorded` in the global operator audit trail.

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
  "recipient_bundle_sha256": null,
  "outcome": "accepted",
  "require_bundle_ready": true,
  "require_bundle_verification_valid": true,
  "require_bundle_hash_match": true,
  "require_recipient_checksum_match": true,
  "require_global_operator_audit": true,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-releases/<release_id>/receipt \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator",
       "recipient":"external-review",
       "recipient_contact":"review-team@example.com",
       "transfer_method":"secure_object_storage",
       "transfer_reference":"s3://handoffs/release-a.zip",
       "received_by":"review-team",
       "received_at":"2026-05-07T15:00:00Z",
       "recipient_bundle_sha256":"<sha256-from-recipient>"}'
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
- `recipient_bundle_sha256`: bundle SHA-256 observed by the recipient.
- `outcome`: `accepted`, `accepted_with_exceptions`, `rejected`, or `pending`.
- `require_bundle_ready`: block when the bundle manifest readiness is not `ready`.
- `require_bundle_verification_valid`: block when bundle verification is missing or not `valid`.
- `require_bundle_hash_match`: block when the current ZIP hash differs from the bundle manifest.
- `require_recipient_checksum_match`: block when recipient checksum is missing or mismatched.
- `require_global_operator_audit`: block when the global operator audit hash chain is invalid.
- `notes`: operator notes persisted into the receipt.

## Readiness

`readiness` is one of:

- `recorded`: all required receipt controls passed.
- `warnings`: no required control failed, but metadata, release readiness, or recipient outcome
  needs attention.
- `blocked`: at least one required transfer, checksum, bundle, verification, or audit control
  failed.

## Receipt Checks

The receipt evaluates:

- Release readiness from `handoff_release.json`.
- Bundle readiness from `handoff_release_bundle_manifest.json`.
- Bundle verification readiness from `handoff_release_bundle_verification.json`.
- Current bundle ZIP SHA-256 against the bundle sidecar manifest.
- Recipient-observed bundle SHA-256 against the bundle sidecar manifest.
- Transfer metadata completeness for recipient, method, and reference.
- Recipient outcome, with rejected outcomes blocking receipt reliance.
- Global operator audit JSONL hash-chain verification.

## Operational Guidance

Generate a receipt after the transfer package is created, verified, copied, and acknowledged by the
recipient. For strict external transfer, keep all required controls enabled and include the
recipient-observed SHA-256. If the recipient checksum differs, recreate or re-copy the transfer
package and record a new receipt after the recipient confirms the corrected hash.

Receipts are local deterministic custody artifacts. They do not move files, prove legal acceptance,
replace recipient-side validation, or provide immutable storage. Store the receipt with the bundle,
sidecar manifest, bundle verification report, and global operator audit export for the final
handoff record. After recording receipts, generate the repository-level ledger documented in
`docs/handoff-release-ledger.md` to summarize final transfer custody across release packages.
