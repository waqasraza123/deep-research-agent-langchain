# Operator Audit Trail

The operator audit subsystem records backend-only actions that change review state, retention
policy, legal holds, replay outputs, export bundles, custody certificates, or cleanup state. It is
file-backed and uses the same `runs/` artifact layout as the rest of the service.

Generated artifacts:

- `runs/_audit/operator_audit.jsonl`: global append-only audit events.
- `runs/_audit/operator_audit.md`: global human-readable audit summary.
- `runs/<thread_id>/operator_audit.jsonl`: per-run audit events for actions that affect a run.
- `runs/<thread_id>/operator_audit.md`: per-run human-readable audit summary.

API endpoints:

- `GET /operator-audit`
- `GET /operator-audit?thread_id=<thread_id>`
- `GET /operator-audit/verify`
- `GET /operator-audit/verify?thread_id=<thread_id>`
- `GET /runs/{thread_id}/operator-audit`
- `GET /runs/{thread_id}/operator-audit/verify`

## Recorded Actions

The API records an audit event after these operations complete successfully:

- `POST /runs/cleanup/apply`
- `POST /runs/{thread_id}/retention`
- `POST /runs/{thread_id}/retention/hold`
- `POST /runs/{thread_id}/retention/release`
- `POST /runs/{thread_id}/review/dossier`
- `POST /runs/{thread_id}/review/approve`
- `POST /runs/{thread_id}/review/request-changes`
- `POST /runs/{thread_id}/review/reject`
- `POST /runs/{thread_id}/replay`
- `POST /runs/{thread_id}/export`
- `POST /runs/{thread_id}/custody`

Cleanup events are written to the global audit trail after deletion completes. Per-run cleanup logs
are not preserved for deleted run directories, so the global audit trail is the durable cleanup
source.

## Event Shape

Each JSONL row is one `OperatorAuditEvent`:

```json
{
  "event_id": "op-...",
  "created_at": "2026-05-06T12:00:00Z",
  "event_type": "review.approved",
  "actor": "operator",
  "thread_id": "run-id",
  "affected_thread_ids": ["run-id"],
  "summary": "Run review was approved.",
  "artifacts": ["review_dossier.json"],
  "metadata": {},
  "previous_hash": "previous event hash or null",
  "event_hash": "sha256 of canonical event payload"
}
```

`metadata` is passed through the provenance secret redactor before persistence. It is intended for
operator-visible context such as review notes, requested changes, retention reason, export profile,
archive hash, replay status, or cleanup targets.

## Hash Chain

Each audit file is its own hash chain. The event hash is computed from a canonical JSON
representation of the event with `event_hash` blanked and `previous_hash` set to the prior event in
that same file.

Global and per-run copies of the same logical event can have different `event_hash` values because
they are appended to different audit scopes with different prior hashes. This is expected. Verify
the global file as the cross-run source of truth and verify each per-run file when reviewing one run
in isolation.

Verification:

```bash
curl http://localhost:8000/operator-audit/verify
curl http://localhost:8000/runs/<thread_id>/operator-audit/verify
```

Verification checks:

- every row is valid JSON and matches the audit schema,
- `previous_hash` points to the preceding row in the same file,
- `event_hash` matches the canonical event payload.

Appending refuses to extend a malformed audit log. This avoids creating a new valid-looking tail
after a corrupted or edited row.

## Operational Guidance

Use the global audit file for custody review, cleanup review, export handoff, and incident
timelines. Use per-run audit files when reviewing a single run alongside `review_dossier.*`,
`retention_policy.*`, `replay_execution.*`, and `exports/export_manifest.*`.

Audit files are local artifacts, not a replacement for centralized security logging. Production
deployments that need stronger guarantees should persist `runs/` on durable storage, ship
`runs/_audit/operator_audit.jsonl` to an external log sink, and restrict write access to the service
identity.
