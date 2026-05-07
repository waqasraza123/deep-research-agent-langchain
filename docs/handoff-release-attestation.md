# Handoff Release Attestation

The handoff release attestation subsystem creates a portfolio-level custody certificate over the
saved release ledger and saved ledger verification report. It reads the latest ledger and ledger
verification sidecars, checks the ledger completion state, requires a valid ledger verification by
default, verifies the global operator audit hash chain, and records SHA-256 hashes for the final
portfolio controls plus release artifacts referenced by the ledger. It does not execute research,
call models, refetch sources, create exports, create bundles, record receipts, regenerate ledgers,
run tests, or build the project.

Generated artifacts:

- `runs/_handoff/handoff_release_attestation.json`
- `runs/_handoff/handoff_release_attestation.md`

API endpoints:

- `POST /runs/handoff-release-attestation`
- `GET /runs/handoff-release-attestation`
- `GET /runs/handoff-release-attestation/markdown`

Generating an attestation records `handoff.release_attestation_generated` in the global operator
audit trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "attestation_scope": "release_transfer_portfolio",
  "require_ledger_complete": true,
  "require_ledger_verification_valid": true,
  "require_global_operator_audit": true,
  "require_artifact_presence": true,
  "include_release_artifact_hashes": true,
  "max_release_artifacts": 5000,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-release-attestation \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator"}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the attestation and audit event.
- `attestation_scope`: human-readable scope label for the certified transfer portfolio.
- `require_ledger_complete`: block when no releases are indexed or any release is incomplete.
- `require_ledger_verification_valid`: block unless the saved ledger verification is `valid`.
- `require_global_operator_audit`: block when the global operator audit hash chain is invalid.
- `require_artifact_presence`: block when any attested artifact is missing or path-unsafe.
- `include_release_artifact_hashes`: hash artifacts referenced by each ledger item.
- `max_release_artifacts`: upper bound for release artifact hashes included in one attestation.
- `notes`: operator notes persisted into the attestation.

## Readiness

`readiness` is one of:

- `attested`: every required attestation control passed.
- `warnings`: no required control failed, but at least one warning remains.
- `blocked`: at least one required attestation control failed.

## Attestation Scope

The attestation hashes these portfolio controls every time:

- `runs/_handoff/handoff_release_ledger.json`
- `runs/_handoff/handoff_release_ledger.md`
- `runs/_handoff/handoff_release_ledger_verification.json`
- `runs/_handoff/handoff_release_ledger_verification.md`

When `include_release_artifact_hashes` is true, it also hashes artifacts listed by each saved
ledger item until `max_release_artifacts` is reached. Missing artifacts and path traversal
references are reported as inventory findings. The attestation sidecars themselves are listed as
generated artifacts, but they are not self-hashed because that would make the certificate
nondeterministic.

## Operational Guidance

Generate an attestation only after release manifests, release verifications, bundles, bundle
verifications, receipts, the release ledger, and ledger verification have been generated. Treat
`blocked` as a stop condition for transfer reliance. Treat `warnings` as an operator review state:
the certificate exists, but the portfolio is not clean enough for unattended custody handoff.

The attestation is a local deterministic certificate. It does not replace immutable artifact
storage, recipient-side checksum validation, legal review, centralized audit logging, or external
signature infrastructure.
