# Review Dossiers

The review dossier subsystem creates deterministic operator review packets from existing run
artifacts. It does not approve runs automatically and it does not replace qualified expert review.
Its purpose is to make the approval decision auditable by surfacing blockers, warnings, evidence
artifacts, required actions, and a recommended disposition.

Generated artifacts:

- `review_dossier.json`: machine-readable review criteria and recommended decision.
- `review_dossier.md`: operator-facing checklist and evidence summary.

API endpoints:

- `POST /runs/{thread_id}/review/dossier`
- `GET /runs/{thread_id}/review/dossier`
- `GET /runs/{thread_id}/review/dossier/markdown`
- Existing decision endpoints remain `POST /runs/{thread_id}/review/approve`,
  `POST /runs/{thread_id}/review/request-changes`, and `POST /runs/{thread_id}/review/reject`.

## Request

Default request:

```json
{
  "confidence_threshold": 0.55,
  "require_export_bundle": false,
  "require_replay_evidence": false,
  "require_provenance": true
}
```

Fields:

- `reviewer`: optional reviewer label stored in the dossier.
- `confidence_threshold`: minimum acceptable `advanced_intelligence_summary` confidence.
- `require_export_bundle`: block approval if `exports/export_manifest.json` is missing.
- `require_replay_evidence`: block approval if `replay_execution.json` is missing.
- `require_provenance`: block approval if reproducibility/provenance artifacts are missing.
- `notes`: operator note persisted into the dossier.

Example:

```bash
curl http://localhost:8000/runs/<thread_id>/review/dossier \
  -H 'content-type: application/json' \
  -d '{"reviewer":"risk-reviewer","require_export_bundle":true}'
```

## Criteria

The dossier evaluates:

- Required deliverables: `plan.md`, `notes.md`, `sources.json`, and `report.md`.
- Run errors: any recorded run error blocks approval.
- Confidence: `advanced_intelligence_summary.json` must meet the configured threshold.
- Verification: high-priority unsupported, contradicted, or unresolved claims block approval.
- Quality score: low quality scores block approval; missing scores produce review warnings.
- Source safety: critical source-safety findings recommend rejection; high-risk findings warn.
- Currentness: stale or contradictory date-sensitive evidence blocks approval.
- Provenance: missing reproducibility evidence blocks when required; partial replayability warns.
- Export bundle: missing handoff package blocks when required.
- Replay evidence: hash mismatches and missing expected artifacts warn; missing replay blocks only
  when required.

Each criterion records status, severity, finding, required action, and evidence artifacts.

## Recommended Decisions

`ready_for_approval` means all configured blocking criteria passed. The reviewer still owns the
final judgment and should inspect the report, citations, source quality, and domain risk.

`changes_requested_recommended` means at least one blocker or warning needs operator attention. Use
`POST /runs/{thread_id}/review/request-changes` after writing concrete requested changes.

`reject_recommended` is reserved for critical blockers such as run errors or critical source-safety
findings. Use `POST /runs/{thread_id}/review/reject` when the run should not be used.

## Operational Flow

1. Generate or refresh downstream artifacts for the run.
2. Optionally run provenance replay and create an export bundle.
3. Generate the review dossier.
4. Inspect `review_dossier.md`, `report.md`, `research_readiness.md`, verification outputs, and
   source-safety/currentness warnings.
5. Approve, request changes, or reject through the existing review endpoints.

The dossier is deliberately conservative. Missing optional artifacts become warnings unless the
request marks them as required. High-stakes domains still require qualified human review even when
the dossier recommends approval.
