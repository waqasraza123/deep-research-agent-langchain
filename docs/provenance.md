# Provenance and Replay Metadata

The backend writes audit metadata for each run under `runs/<thread_id>/`.
The provenance subsystem is backend-only and is generated from captured run files, the run
registry snapshot, source manifests, runtime events, and redacted model/settings metadata.

Generated provenance artifacts:

- `artifact_manifest.json` / `artifact_manifest.md`: artifact inventory with content hashes,
  producer subsystem, dependencies, settings fingerprint, source fingerprints, and model
  fingerprints.
- `artifact_dependency_dag.json` / `artifact_dependency_dag.md`: inferred lineage edges from
  inputs, settings, sources, models, and intermediate artifacts to final artifacts.
- `reproducibility_report.json` / `reproducibility_report.md`: replayability assessment showing
  live-source dependencies, model nondeterminism, offline-regenerable artifacts, required
  providers/credentials, redacted settings, and non-reproducible parts.
- `replay_plan.json` / `replay_plan.md`: ordered steps for reproducing the run without performing
  replay automatically.
- `replay_execution.json` / `replay_execution.md`: written on replay runs to show copied seed
  artifacts, rebuilt layers, warnings, expected-hash comparisons, and baseline manifest diff
  counts.

The manifest uses SHA-256 hashes for artifact bytes and deterministic hashes for structured
fingerprints. API keys, tokens, passwords, credentials, and authorization values are redacted before
settings or model configs are persisted or hashed. Self-referential provenance files are marked with
`content_hash: "self-referential"` because their own bytes cannot be included in a stable manifest
hash.

API endpoints:

- `GET /runs/{thread_id}/manifest`
- `GET /runs/{thread_id}/provenance`
- `GET /runs/{thread_id}/reproducibility`
- `GET /runs/{thread_id}/replay-plan`
- `POST /runs/{thread_id}/replay`
- `POST /runs/diff` with `left_thread_id` and `right_thread_id`

## Replay Execution

`POST /runs/{thread_id}/replay` creates a new isolated run directory and registry entry. By default
the replay run ID is generated as `replay-<thread_id>-<suffix>`, though operators can provide
`replay_thread_id` when they need a stable target. Existing non-empty replay directories are
rejected unless `allow_overwrite=true`.

Default request:

```json
{
  "offline_only": true,
  "fail_on_layer_error": false
}
```

Optional fields:

- `replay_thread_id`: explicit target run ID. It must be a safe single path segment and must differ
  from the source run.
- `allow_overwrite`: delete and recreate an existing target run directory.
- `include_artifacts`: additional source-run relative artifacts to seed into the replay.
- `rebuild_layers`: subset of deterministic layers to rebuild.
- `question_override`: replay question to store in the new run registry.
- `fail_on_layer_error`: stop on the first layer exception. The default records the failure and
  continues so operators get as much comparison data as possible.

The default layer order is:

1. `source_safety`
2. `source_audit`
3. `document_intelligence`
4. `retrieval`
5. `temporal`
6. `quantitative`
7. `evidence`
8. `hypotheses`
9. `synthesis`
10. `verification`
11. `evaluation`
12. `summaries`
13. `intelligence_kernel`
14. `provenance`

Replay is intentionally conservative. It treats `plan.md`, `notes.md`, `report.md`, `sources.json`,
source text files, sanitized source files, and selected planning/safety artifacts as captured input.
It rewrites JSON references from `runs/<source_thread_id>/...` to
`runs/<replay_thread_id>/...` before rebuilding layers. This avoids mutating the original run and
keeps source-file path resolution local to the replay directory.

`offline_only=true` does not refetch live URLs and does not rerun model-dependent agent generation.
It validates deterministic backend layers against the captured evidence. A mismatch therefore means
one of three things:

- deterministic logic changed since the baseline run,
- the replay seed artifacts differ from the original captured inputs,
- an artifact depends on timestamps, registry metadata, generated IDs, or other intentionally
  variable replay context.

The response includes:

- `summary.status`: `completed`, `completed_with_warnings`, or `failed`.
- `summary.rebuilt_layers` and `summary.skipped_layers`.
- `summary.hash_matches`, `summary.hash_mismatches`, and
  `summary.missing_expected_artifacts` from `replay_plan.expected_output_hashes`.
- `summary.baseline_diff`, using the same added/removed/changed/unchanged artifact structure as
  `POST /runs/diff`.

The diff helper compares manifests or run directories and reports added, removed, changed, and
unchanged artifacts, including hash, size, and producer changes.
