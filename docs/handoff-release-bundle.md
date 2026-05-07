# Handoff Release Bundles

The handoff release bundle subsystem creates a portable ZIP package for an existing handoff
release. It reads the release manifest, optional verification report, registry snapshot, and
selected run export archives, then writes a bundle archive plus sidecar JSON/Markdown manifests. It
does not execute research, call models, refetch sources, create per-run exports, run tests, or
build the project.

Generated artifacts:

- `runs/_handoff/releases/<release_id>/handoff_release_bundle.zip`
- `runs/_handoff/releases/<release_id>/handoff_release_bundle_manifest.json`
- `runs/_handoff/releases/<release_id>/handoff_release_bundle_manifest.md`

API endpoints:

- `POST /runs/handoff-releases/{release_id}/bundle`
- `GET /runs/handoff-releases/{release_id}/bundle`
- `GET /runs/handoff-releases/{release_id}/bundle/markdown`
- `GET /runs/handoff-releases/{release_id}/bundle/download`
- `POST /runs/handoff-releases/{release_id}/bundle/verification`
- `GET /runs/handoff-releases/{release_id}/bundle/verification`
- `GET /runs/handoff-releases/{release_id}/bundle/verification/markdown`

Generating a bundle records `handoff.release_bundle_created` in the global operator audit trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "require_release_ready": true,
  "require_verification_valid": true,
  "include_verification": true,
  "include_registry_snapshot": true,
  "include_run_exports": true,
  "require_run_exports": true,
  "overwrite_existing": true,
  "max_total_input_bytes": 2000000000,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-releases/<release_id>/bundle \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator"}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the bundle manifest and audit
  event.
- `require_release_ready`: mark the bundle blocked unless the release manifest is
  `ready_for_release`.
- `require_verification_valid`: mark the bundle blocked unless the release verification report is
  present and `valid`.
- `include_verification`: include `handoff_release_verification.json` / `.md` when present.
- `include_registry_snapshot`: include the current handoff registry JSON/Markdown snapshot when
  present.
- `include_run_exports`: include each selected run's `exports/run_export.zip`.
- `require_run_exports`: mark missing or hash-mismatched run export archives as blockers.
- `overwrite_existing`: replace an existing bundle archive and sidecar manifests for the release.
- `max_total_input_bytes`: maximum combined input bytes considered for the package.
- `notes`: operator notes persisted into the bundle manifest.

## Readiness

`readiness` is one of:

- `ready`: release controls passed and all requested package inputs were included.
- `warnings`: no required control failed, but optional package inputs or registry sidecars are
  missing.
- `blocked`: at least one required release, verification, or run-export control failed.

## Bundle Contents

The ZIP uses stable archive paths:

- `release/handoff_release.json`
- `release/handoff_release.md`
- `release/handoff_release_verification.json` when included
- `release/handoff_release_verification.md` when included
- `registry/handoff_registry.json` when included and present
- `registry/handoff_registry.md` when included and present
- `run_exports/<thread_id>/run_export.zip` for each included selected run
- `handoff_release_bundle_manifest.json`
- `handoff_release_bundle_manifest.md`

The sidecar manifest records archive size and SHA-256 after the ZIP is written. The manifest copy
inside the ZIP is the package inventory at write time; use the sidecar manifest for the final
bundle archive hash.

## Operational Guidance

Generate the bundle only after the handoff registry, release manifest, and release verification
have been refreshed for the intended transfer. The bundle does not regenerate missing run exports;
if a selected run export is missing or its hash differs from the release manifest, regenerate that
run's export, rebuild the registry and release, rerun verification, then create the bundle again.

For strict external transfer, keep the default required controls enabled. Relax them only for
internal exception workflows, and keep the reason in `notes`. Store the ZIP together with
`handoff_release_bundle_manifest.json` so recipients can verify the archive SHA-256 independently.
Before and after copying the package, generate the verification report documented in
`docs/handoff-release-bundle-verification.md`.
