# Handoff Release Verification

The handoff release verification subsystem creates deterministic verification reports for existing
handoff release manifests. It reads a release record, compares it with current registry, export,
release-artifact, and operator-audit state, then writes sidecar verification artifacts in the
release directory. It does not execute research, call models, refetch sources, create exports, run
tests, or build the project.

Generated artifacts:

- `runs/_handoff/releases/<release_id>/handoff_release_verification.json`
- `runs/_handoff/releases/<release_id>/handoff_release_verification.md`

API endpoints:

- `POST /runs/handoff-releases/{release_id}/verification`
- `GET /runs/handoff-releases/{release_id}/verification`
- `GET /runs/handoff-releases/{release_id}/verification/markdown`

Generating a verification report records `handoff.release_verified` in the global operator audit
trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "require_registry_hash_match": true,
  "require_export_hashes": true,
  "require_release_artifacts": true,
  "require_global_operator_audit": true,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-releases/<release_id>/verification \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator"}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the report and audit event.
- `require_registry_hash_match`: fail when the current registry hash differs from the release
  snapshot hash.
- `require_export_hashes`: fail when a selected run export is missing or its SHA-256 changed.
- `require_release_artifacts`: fail when `handoff_release.json` or `.md` is missing.
- `require_global_operator_audit`: fail when the global operator audit hash chain is invalid.
- `notes`: operator notes persisted into the report.

## Readiness

`readiness` is one of:

- `valid`: every required verification control passed.
- `warnings`: no required control failed, but at least one warning remains.
- `failed`: at least one required verification control failed.

## Verification Scope

The report verifies:

- Release JSON/Markdown artifacts exist.
- Current `runs/_handoff/handoff_registry.json` SHA-256 matches the release snapshot hash.
- Global operator audit JSONL hash chain verifies.
- Each selected run's export archive still exists and matches the SHA-256 recorded in the release.
- Artifacts referenced by each release run still exist in the run directory.

## Operational Guidance

Run verification immediately before transferring release packages and again after restoring a
release from storage. A registry hash mismatch does not automatically prove a release is invalid:
it can mean the registry was regenerated after release creation. For strict external handoff,
investigate any mismatch and regenerate the registry, release, and verification report when the
current repository state should be the source of truth.

The verification report is a sidecar audit artifact. It does not replace the release manifest or
the per-run export bundles selected for transfer. After a release verifies cleanly, create the
portable bundle documented in `docs/handoff-release-bundle.md` so the release record,
verification sidecars, registry snapshot, and selected run exports can be transferred together.
