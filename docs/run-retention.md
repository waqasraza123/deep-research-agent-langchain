# Run Retention And Legal Holds

The run retention subsystem stores per-run cleanup policy under `runs/<thread_id>/` and makes the
cleanup planner honor protected runs. It is backend-only and file-backed, matching the rest of the
run artifact system.

Generated artifacts:

- `retention_policy.json`: machine-readable retention class, dates, holds, and warnings.
- `retention_policy.md`: operator-facing retention summary.

API endpoints:

- `POST /runs/{thread_id}/retention`
- `GET /runs/{thread_id}/retention`
- `POST /runs/{thread_id}/retention/hold`
- `POST /runs/{thread_id}/retention/release`
- Existing cleanup endpoints remain `GET /runs/cleanup/plan` and `POST /runs/cleanup/apply`.

## Retention Policy

Set or replace a run policy:

```bash
curl http://localhost:8000/runs/<thread_id>/retention \
  -H 'content-type: application/json' \
  -d '{"retention_class":"audit","reason":"approved review package","requested_by":"operator"}'
```

Fields:

- `retention_class`: `ephemeral`, `standard`, `audit`, or `regulated`.
- `retain_days`: optional explicit retention duration from now.
- `retain_until`: optional ISO timestamp. Overrides class/default days.
- `delete_after`: optional ISO timestamp for external lifecycle tracking.
- `legal_hold`: marks the run as held even without individual hold records.
- `reason`: operator-visible justification.
- `requested_by`: operator or automation identity.

Default class durations:

- `ephemeral`: 7 days
- `standard`: 30 days
- `audit`: 365 days
- `regulated`: 2555 days

These defaults are local policy defaults, not legal advice. Regulated retention should be configured
by the organization that owns the data.

## Legal Holds

Add a hold:

```bash
curl http://localhost:8000/runs/<thread_id>/retention/hold \
  -H 'content-type: application/json' \
  -d '{"hold_id":"case-2026-05","reason":"customer dispute","requested_by":"legal"}'
```

Release one hold:

```bash
curl http://localhost:8000/runs/<thread_id>/retention/release \
  -H 'content-type: application/json' \
  -d '{"hold_id":"case-2026-05","released_by":"legal","reason":"case closed"}'
```

If `hold_id` is omitted during release, all active holds are released. The top-level `legal_hold`
flag remains true while any hold is active and is cleared when no active holds remain.

## Cleanup Behavior

`GET /runs/cleanup/plan` now returns two sets:

- `items`: runs eligible for deletion.
- `protected_items`: runs that matched cleanup rules but are blocked by retention.

A run is blocked when:

- `legal_hold=true`,
- one or more active holds exist,
- `retain_until` is in the future.

`POST /runs/cleanup/apply` refuses to delete any requested run that appears in `protected_items`.
Operators must release holds or wait until `retain_until` has passed before cleanup can delete the
run directory.

## Operational Guidance

Use `standard` for normal local runs, `audit` after review/export/replay handoff, `regulated` for
organization-managed compliance retention, and `ephemeral` only for throwaway development runs. Add
legal holds for disputes, investigations, reproducibility reviews, or any situation where deletion
must be blocked independently of age and size cleanup rules.
