# Handoff Release Ledger

The handoff release ledger is a repository-level custody index for release transfer packages. It
reads existing release manifests, release verification reports, bundle manifests, bundle
verification reports, transfer receipts, bundle archive hashes, and global operator audit state,
then writes a deterministic ledger under `runs/_handoff/`. It does not execute research, call
models, refetch sources, create exports, create bundles, transfer packages, record receipts, run
tests, or build the project.

Generated artifacts:

- `runs/_handoff/handoff_release_ledger.json`
- `runs/_handoff/handoff_release_ledger.md`

API endpoints:

- `POST /runs/handoff-release-ledger`
- `GET /runs/handoff-release-ledger`
- `GET /runs/handoff-release-ledger/markdown`
- `POST /runs/handoff-release-ledger/verification`
- `GET /runs/handoff-release-ledger/verification`
- `GET /runs/handoff-release-ledger/verification/markdown`
- `POST /runs/handoff-release-attestation`
- `GET /runs/handoff-release-attestation`
- `GET /runs/handoff-release-attestation/markdown`
- `POST /runs/handoff-release-attestation/verification`
- `GET /runs/handoff-release-attestation/verification`
- `GET /runs/handoff-release-attestation/verification/markdown`
- `POST /runs/handoff-release-portfolio-receipt`
- `GET /runs/handoff-release-portfolio-receipt`
- `GET /runs/handoff-release-portfolio-receipt/markdown`
- `POST /runs/handoff-release-portfolio-receipt/verification`
- `GET /runs/handoff-release-portfolio-receipt/verification`
- `GET /runs/handoff-release-portfolio-receipt/verification/markdown`
- `POST /runs/handoff-release-portfolio-closeout`
- `GET /runs/handoff-release-portfolio-closeout`
- `GET /runs/handoff-release-portfolio-closeout/markdown`
- `POST /runs/handoff-release-portfolio-closeout/verification`
- `GET /runs/handoff-release-portfolio-closeout/verification`
- `GET /runs/handoff-release-portfolio-closeout/verification/markdown`

Generating a ledger records `handoff.release_ledger_generated` in the global operator audit trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "include_releases_without_receipt": true,
  "require_release_ready": true,
  "require_release_verification_valid": true,
  "require_bundle_ready": true,
  "require_bundle_verification_valid": true,
  "require_receipt_recorded": true,
  "require_recipient_checksum_match": true,
  "require_global_operator_audit": true,
  "max_releases": 1000,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-release-ledger \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator"}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the ledger and audit event.
- `include_releases_without_receipt`: include releases that have no transfer receipt yet.
- `require_release_ready`: block a ledger item when release readiness is not `ready_for_release`.
- `require_release_verification_valid`: block when release verification is missing or not `valid`.
- `require_bundle_ready`: block when the release bundle is missing or not `ready`.
- `require_bundle_verification_valid`: block when bundle verification is missing or not `valid`.
- `require_receipt_recorded`: block when a transfer receipt is missing or not `recorded`.
- `require_recipient_checksum_match`: block when the receipt lacks a matching recipient checksum.
- `require_global_operator_audit`: block when the global operator audit hash chain is invalid.
- `max_releases`: maximum number of most-recent release manifests to index.
- `notes`: operator notes persisted into the ledger.

## Readiness

Each indexed release has `readiness`:

- `complete`: every required custody control passed and no warnings remain.
- `needs_attention`: no required control failed, but one or more warnings remain.
- `blocked`: at least one required release, bundle, receipt, checksum, or audit control failed.

## Indexed Signals

Each ledger item records:

- Release readiness, generation time, and selected run count.
- Release verification readiness and generation time.
- Bundle readiness, generation time, archive SHA-256, and archive hash validity.
- Bundle verification readiness and generation time.
- Receipt readiness, outcome, generation time, recipient, transfer method, and reference.
- Recipient-observed bundle checksum and whether it matches the bundle manifest.
- Global operator audit verification status and event count.
- Missing controls, blockers, warnings, and evidence artifacts.

## Operational Guidance

Generate the ledger after release receipts are recorded to produce a repository-level view of final
transfer custody. Regenerate it whenever release verification, bundle creation, bundle
verification, receipt data, or operator audit state changes. A complete ledger item means the local
release package has a ready release, valid release verification, ready bundle, valid bundle
verification, recorded receipt, matching recipient checksum, and valid global audit chain under the
request's required controls.

The ledger is a snapshot, not immutable storage. Store it with release manifests, bundle manifests,
bundle verification reports, receipts, and operator audit exports for portfolio-level custody
review. After generation, create the verification report documented in
`docs/handoff-release-ledger-verification.md` to confirm the saved ledger still matches current
release custody state. After verification is valid, generate the release attestation documented in
`docs/handoff-release-attestation.md` to record final portfolio hashes, then verify that
attestation with `docs/handoff-release-attestation-verification.md`. After recipient checksum
confirmation, record the portfolio receipt in `docs/handoff-release-portfolio-receipt.md` and
verify it with `docs/handoff-release-portfolio-receipt-verification.md`. Generate the final
closeout manifest in `docs/handoff-release-portfolio-closeout.md` when every custody control is
ready for closure, then verify it with
`docs/handoff-release-portfolio-closeout-verification.md`.
