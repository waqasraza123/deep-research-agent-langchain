# Run Disclosure Reports

The run disclosure subsystem creates deterministic disclosure-risk reports for existing run
directories. It does not execute research, call models, refetch sources, create exports, run tests,
or build the project. It scans local artifact names, text payloads, and export metadata for risks
that should be reviewed before external handoff.

Generated artifacts:

- `disclosure_report.json`: machine-readable findings, risk level, artifact summary, and metadata.
- `disclosure_report.md`: operator-facing disclosure summary.

API endpoints:

- `POST /runs/{thread_id}/disclosure`
- `GET /runs/{thread_id}/disclosure`
- `GET /runs/{thread_id}/disclosure/markdown`

Generating a report records `disclosure.report_generated` in the operator audit trail.

## Request

Default request:

```json
{
  "requested_by": "operator",
  "require_no_high_risk": false,
  "scan_text": true,
  "max_files": 1000,
  "max_scan_bytes": 500000,
  "notes": ""
}
```

Example strict handoff scan:

```bash
curl http://localhost:8000/runs/<thread_id>/disclosure \
  -H 'content-type: application/json' \
  -d '{"requested_by":"operator",
       "require_no_high_risk":true,
       "scan_text":true,
       "max_scan_bytes":1000000}'
```

Fields:

- `requested_by`: operator or automation identity persisted into the report and audit event.
- `require_no_high_risk`: sets readiness to `blocked` when high or critical findings exist.
- `scan_text`: scan text-like artifacts for likely secret patterns.
- `max_files`: cap the artifact scan for very large run directories.
- `max_scan_bytes`: skip secret scanning for text artifacts larger than this byte limit.
- `notes`: operator notes persisted into the report.

## Readiness

`readiness` is one of:

- `clear`: no findings were produced.
- `review_required`: one or more findings need operator review.
- `blocked`: `require_no_high_risk=true` and at least one high or critical finding exists.

`risk_level` is the highest finding severity observed: `low`, `medium`, `high`, or `critical`.

## Findings

The report evaluates:

- Raw source artifacts under `sources/`.
- Sanitized source artifact counts under `sanitized_sources/`.
- Likely secret text patterns, including OpenAI-style keys, GitHub tokens, bearer tokens, and
  obvious credential assignments.
- Text artifacts skipped because they exceed `max_scan_bytes`.
- Export manifest posture, including missing export manifests, disabled redaction, and raw-source
  export inclusion.

Findings intentionally do not include matched secret values. They record the pattern name, path, and
match count so operators can inspect and remediate without copying credentials into the report.

## Operational Guidance

Run disclosure scans before external handoff and after creating or refreshing an export bundle. For
public or customer-facing packages, use `require_no_high_risk=true`, keep export redaction enabled,
and exclude raw source payloads unless the recipient is explicitly authorized to receive captured
source text.

Use `POST /runs/{thread_id}/handoff` after disclosure scanning when you need a single final
go/no-go manifest across review, retention, export, custody, integrity, disclosure, and audit
controls.

This report is a deterministic local review aid. It does not replace data classification, DLP,
centralized secret scanning, legal review, access controls, or qualified human review for
high-stakes outputs.
