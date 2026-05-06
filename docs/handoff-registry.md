# Handoff Registry

The handoff registry is a repository-level readiness index for run handoffs. It reads existing
per-run lifecycle metadata and control artifacts, then writes a deterministic global snapshot under
`runs/_handoff/`. It does not execute research, call models, refetch sources, create exports, run
tests, or build the project.

Generated artifacts:

- `runs/_handoff/handoff_registry.json`: machine-readable readiness summary for indexed runs.
- `runs/_handoff/handoff_registry.md`: operator-facing portfolio handoff dashboard.

API endpoints:

- `POST /runs/handoff-registry`
- `GET /runs/handoff-registry`
- `GET /runs/handoff-registry/markdown`

Generating a registry records `handoff.registry_generated` in the global operator audit trail. It
does not write per-run audit entries because the registry is a global index, not a mutation of each
run package.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "include_runs_without_handoff": true,
  "require_operator_audit_valid": true,
  "max_runs": 1000,
  "notes": ""
}
```

Example:

```bash
curl http://localhost:8000/runs/handoff-registry \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator","max_runs":500}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the registry and audit event.
- `include_runs_without_handoff`: include runs that have not generated `handoff_manifest.*` yet.
- `require_operator_audit_valid`: mark invalid per-run audit chains as blocking registry findings.
- `max_runs`: maximum number of most-recent repository runs to index.
- `notes`: operator notes persisted into the registry.

## Indexed Signals

Each registry item includes:

- Run status, review status, creation time, and last update time.
- Final handoff readiness and handoff generation timestamp when `handoff_manifest.json` exists.
- Custody, integrity, disclosure, retention, export, and operator-audit status.
- Export profile, archive hash, and archive-vs-manifest hash verification.
- Retention class, legal-hold state, and active hold IDs.
- Missing controls, blockers, warnings, and evidence artifact names.

## Readiness

`readiness` is one of:

- `ready_for_handoff`: the run has no registry-level blockers or warnings and the handoff manifest
  is ready.
- `needs_attention`: the run is not blocked, but has warnings, missing non-terminal controls, or a
  non-ready handoff state.
- `blocked`: the run has blocking handoff findings, failed/cancelled lifecycle state, rejected or
  changes-requested review state, invalid required audit verification, export hash mismatch, or no
  handoff manifest.

## Operational Guidance

Use the registry after generating per-run handoff manifests to produce a repository-level queue for
operators. The registry is intentionally a snapshot: regenerate it after changing any run-level
control artifact. A registry item warns when a handoff manifest is older than upstream control
artifacts such as retention, export, custody, integrity, or disclosure reports.

The registry complements `docs/run-handoff.md`; it does not replace the per-run manifest, immutable
storage, legal review, or centralized audit logging.
