# Handoff Release Portfolio Closeout

The handoff release portfolio closeout subsystem creates the final deterministic go/no-go manifest
for a repository-level release transfer portfolio. It reads the saved release ledger, ledger
verification, attestation, attestation verification, portfolio receipt, portfolio receipt
verification, and global operator audit state, then writes closeout artifacts under
`runs/_handoff/`. It does not execute research, call models, refetch sources, copy files, send data
to recipients, regenerate upstream custody controls, run tests, or build the project.

Generated artifacts:

- `runs/_handoff/handoff_release_portfolio_closeout.json`
- `runs/_handoff/handoff_release_portfolio_closeout.md`

API endpoints:

- `POST /runs/handoff-release-portfolio-closeout`
- `GET /runs/handoff-release-portfolio-closeout`
- `GET /runs/handoff-release-portfolio-closeout/markdown`
- `POST /runs/handoff-release-portfolio-closeout/verification`
- `GET /runs/handoff-release-portfolio-closeout/verification`
- `GET /runs/handoff-release-portfolio-closeout/verification/markdown`

Generating closeout records `handoff.release_portfolio_closed` in the global operator audit trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "closeout_scope": "release_transfer_portfolio",
  "require_ledger_complete": true,
  "require_ledger_verification_valid": true,
  "require_attestation_ready": true,
  "require_attestation_verification_valid": true,
  "require_portfolio_receipt_recorded": true,
  "require_portfolio_receipt_verification_valid": true,
  "require_final_artifact_presence": true,
  "require_global_operator_audit": true,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-release-portfolio-closeout \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator"}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the closeout and audit event.
- `closeout_scope`: human-readable scope label for the closed portfolio.
- `require_ledger_complete`: block when the saved ledger has blocked or needs-attention releases.
- `require_ledger_verification_valid`: block when saved ledger verification is not `valid`.
- `require_attestation_ready`: block when saved attestation readiness is not `attested`.
- `require_attestation_verification_valid`: block when attestation verification is not `valid`.
- `require_portfolio_receipt_recorded`: block when portfolio receipt readiness is not `recorded`.
- `require_portfolio_receipt_verification_valid`: block when receipt verification is not `valid`.
- `require_final_artifact_presence`: block when any final closeout artifact is missing or unsafe.
- `require_global_operator_audit`: block when the global operator audit hash chain is invalid.
- `notes`: operator notes persisted into the closeout manifest.

## Readiness

`readiness` is one of:

- `closed`: every required closeout control passed.
- `warnings`: no required control failed, but at least one warning remains.
- `blocked`: at least one required closeout control failed.

## Closeout Scope

The closeout hashes these final artifacts:

- `runs/_handoff/handoff_release_ledger.json`
- `runs/_handoff/handoff_release_ledger.md`
- `runs/_handoff/handoff_release_ledger_verification.json`
- `runs/_handoff/handoff_release_ledger_verification.md`
- `runs/_handoff/handoff_release_attestation.json`
- `runs/_handoff/handoff_release_attestation.md`
- `runs/_handoff/handoff_release_attestation_verification.json`
- `runs/_handoff/handoff_release_attestation_verification.md`
- `runs/_handoff/handoff_release_portfolio_receipt.json`
- `runs/_handoff/handoff_release_portfolio_receipt.md`
- `runs/_handoff/handoff_release_portfolio_receipt_verification.json`
- `runs/_handoff/handoff_release_portfolio_receipt_verification.md`
- `runs/_audit/operator_audit.jsonl`
- `runs/_audit/operator_audit.md`

The closeout sidecars themselves are listed as generated artifacts, but they are not self-hashed
because that would make the manifest nondeterministic.

## Closeout Checks

The closeout evaluates:

- Release ledger completeness.
- Ledger verification readiness.
- Portfolio attestation readiness.
- Attestation verification readiness.
- Portfolio receipt readiness.
- Portfolio receipt verification readiness.
- Cross-artifact agreement on attestation identity and generation timestamps.
- Final artifact presence, path safety, size, and SHA-256 hashes.
- Global operator audit JSONL hash-chain verification.

## Operational Guidance

Generate closeout only after portfolio receipt verification is `valid`. Treat `blocked` as a stop
condition for custody closure. Treat `warnings` as an operator review state: the manifest exists,
but the portfolio is not clean enough for unattended closure. If any upstream custody artifact is
regenerated after closeout, generate the dependent verification reports again and record a new
closeout.

The closeout is a local deterministic manifest. It does not replace immutable artifact storage,
recipient-side checksum validation, legal review, centralized audit logging, or external signature
infrastructure. After generation, create the verification report documented in
`docs/handoff-release-portfolio-closeout-verification.md`.
