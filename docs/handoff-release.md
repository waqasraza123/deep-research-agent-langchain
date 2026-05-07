# Handoff Releases

The handoff release subsystem creates repository-level release manifests for batches of
handoff-ready runs. It consumes the existing handoff registry and writes immutable JSON/Markdown
records under `runs/_handoff/releases/<release_id>/`. It does not execute research, call models,
refetch sources, create exports, run tests, or build the project.

Generated artifacts:

- `runs/_handoff/releases/<release_id>/handoff_release.json`: machine-readable release record.
- `runs/_handoff/releases/<release_id>/handoff_release.md`: operator-facing release summary.

API endpoints:

- `POST /runs/handoff-releases`
- `GET /runs/handoff-releases`
- `GET /runs/handoff-releases/{release_id}`
- `GET /runs/handoff-releases/{release_id}/markdown`

Generating a release records `handoff.release_generated` in the global operator audit trail. The
release manifest is a global record and does not mutate individual run packages.

## Request

Default request:

```json
{
  "release_id": null,
  "requested_by": "operator",
  "recipient": "",
  "purpose": "external_handoff_release",
  "thread_ids": [],
  "include_ready_runs_when_empty": true,
  "require_registry_ready": true,
  "require_handoff_ready": true,
  "require_export_bundle": true,
  "require_no_missing_controls": true,
  "overwrite_existing": false,
  "notes": ""
}
```

Example release for all registry-ready runs:

```bash
curl http://localhost:8000/runs/handoff-releases \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator","recipient":"external-review"}'
```

Example release for explicit runs:

```bash
curl http://localhost:8000/runs/handoff-releases \
  -H 'content-type: application/json' \
  -d '{"release_id":"audit-2026-05-06",
       "requested_by":"operator",
       "recipient":"external-review",
       "thread_ids":["<thread_id_1>","<thread_id_2>"]}'
```

Fields:

- `release_id`: optional stable release identifier. If omitted, the service generates one.
- `requested_by`: operator or automation identity persisted into the release and audit event.
- `recipient`: release recipient label.
- `purpose`: release purpose label.
- `thread_ids`: explicit run IDs to include. When empty, ready registry runs are selected by
  default.
- `include_ready_runs_when_empty`: select all registry-ready runs when `thread_ids` is empty.
- `require_registry_ready`: block release if a selected run is not `ready_for_handoff`.
- `require_handoff_ready`: block release if a selected run's handoff manifest is not ready.
- `require_export_bundle`: block release if a selected run lacks a valid export bundle.
- `require_no_missing_controls`: block release if registry control artifacts are missing.
- `overwrite_existing`: replace an existing release artifact with the same `release_id`.
- `notes`: operator notes persisted into the release manifest.

## Readiness

`readiness` is one of:

- `ready_for_release`: every selected run satisfies the required release controls.
- `needs_attention`: no blocking release findings exist, but warnings remain.
- `blocked`: at least one selected run fails a required release control or a requested run is
  missing from the registry.

## Release Contents

Each release manifest records:

- Release identity, recipient, purpose, requester, and generation timestamp.
- Source handoff registry generation timestamp and SHA-256 hash.
- Requested thread IDs and missing requested thread IDs.
- Selected run readiness, lifecycle/review state, export hash status, custody/integrity/disclosure
  status, retention state, legal-hold state, operator-audit state, missing controls, blockers, and
  warnings.
- Release-level blocker and warning summaries.

## Operational Guidance

Generate the handoff registry immediately before creating a release so the release references a
fresh registry snapshot. For external handoffs, use strict defaults: require registry readiness,
handoff readiness, export bundle availability, and no missing controls. Relax those flags only for
internal exception workflows, and keep the reason in `notes`.

The release manifest is a release decision record, not the portable data package itself. Recipients
still receive the per-run export bundles, and operators can use the release manifest to prove which
run packages were selected, what registry snapshot was used, and which readiness controls were
enforced.

Before transfer, generate the verification report documented in
`docs/handoff-release-verification.md` to re-check the release artifacts, registry snapshot hash,
selected export archive hashes, and global operator audit chain. After verification, create the
portable transfer package documented in `docs/handoff-release-bundle.md`.
