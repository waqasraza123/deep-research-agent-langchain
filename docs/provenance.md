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
- `POST /runs/diff` with `left_thread_id` and `right_thread_id`

Replay metadata is planning-only. The backend does not replay a run unless a future explicit replay
endpoint invokes those steps. The diff helper compares manifests or run directories and reports
added, removed, changed, and unchanged artifacts, including hash, size, and producer changes.
