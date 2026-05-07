# Handoff Release Bundle Verification

The handoff release bundle verification subsystem creates deterministic verification reports for
portable handoff release ZIP packages. It reads an existing bundle sidecar manifest and ZIP
archive, checks archive hash integrity, ZIP inventory, entry SHA-256 values, archive path safety,
embedded manifest consistency, and the global operator audit chain, then writes sidecar
verification artifacts beside the bundle. It does not execute research, call models, refetch
sources, create exports, create bundles, run tests, or build the project.

Generated artifacts:

- `runs/_handoff/releases/<release_id>/handoff_release_bundle_verification.json`
- `runs/_handoff/releases/<release_id>/handoff_release_bundle_verification.md`

API endpoints:

- `POST /runs/handoff-releases/{release_id}/bundle/verification`
- `GET /runs/handoff-releases/{release_id}/bundle/verification`
- `GET /runs/handoff-releases/{release_id}/bundle/verification/markdown`

Generating a report records `handoff.release_bundle_verified` in the global operator audit trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "require_archive_hash_match": true,
  "require_manifest_entries": true,
  "require_path_safety": true,
  "require_global_operator_audit": true,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-releases/<release_id>/bundle/verification \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator"}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the report and audit event.
- `require_archive_hash_match`: fail when the current ZIP hash or size differs from the sidecar
  bundle manifest.
- `require_manifest_entries`: fail when a manifest-listed archive entry is missing or its size or
  SHA-256 differs from the bundle manifest.
- `require_path_safety`: fail when the ZIP has absolute paths, parent traversal paths, backslash
  paths, or duplicate entries.
- `require_global_operator_audit`: fail when the global operator audit hash chain is invalid.
- `notes`: operator notes persisted into the verification report.

## Readiness

`readiness` is one of:

- `valid`: every required verification control passed.
- `warnings`: no required control failed, but at least one warning remains.
- `failed`: at least one required verification control failed.

## Verification Scope

The report verifies:

- Bundle ZIP, JSON sidecar, and Markdown sidecar exist.
- Current bundle ZIP SHA-256 and size match `handoff_release_bundle_manifest.json`.
- ZIP internal CRC checks pass.
- ZIP entry paths are relative, unique, and traversal-safe.
- Every `entries[]` item in the sidecar manifest exists in the ZIP.
- Every manifest-listed entry has matching byte size and SHA-256.
- Embedded `handoff_release_bundle_manifest.json` has the same release identity and generation time
  as the sidecar manifest.
- Global operator audit JSONL hash chain verifies.

The manifest copy inside the ZIP is written before the final archive SHA-256 exists, so the
sidecar manifest remains the source of truth for the ZIP archive hash.

## Operational Guidance

Run bundle verification immediately after creating a bundle and again after copying, restoring, or
receiving a bundle from storage. A failed archive hash or manifest-entry check means the ZIP and
sidecar no longer describe the same package. For strict external transfer, recreate the bundle from
fresh release inputs, rerun bundle verification, then transfer the ZIP together with the sidecar
manifest and verification report.

Use relaxed required controls only for internal exception workflows, and keep the reason in
`notes`. The verification report is a deterministic local artifact; it does not replace immutable
object storage, recipient-side checksum validation, legal review, or centralized audit logging.
After recipient checksum confirmation, record the transfer receipt documented in
`docs/handoff-release-receipt.md`.
