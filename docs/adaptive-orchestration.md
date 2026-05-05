# Adaptive Orchestration

The backend includes a deterministic orchestration subsystem under
`deep_research_agent.orchestration`. It classifies each research request, builds a typed task
graph, routes work through specialist stage contracts, persists stage outputs, and gives the
agent an instruction block before report generation.

This layer is intentionally backend-only and offline. It does not call hidden models, and its
heuristics are written as auditable findings, warnings, required next steps, and confidence
policies rather than expert truth.

## Task Graph

The graph is represented by Pydantic contracts:

- `ResearchTaskGraph`
- `ResearchTaskNode`
- `ResearchTaskEdge`
- `ResearchTaskStatus`
- `ResearchTaskType`
- `SpecialistRole`
- `StageInput`
- `StageOutput`
- `StageError`
- `ExecutionDecision`

Task types cover question normalization, source triage, source extraction, evidence collection,
subquestion answering, contradiction scan, risk scan, synthesis, citation review, and final
report review.

Specialist roles include planner, source triager, evidence collector, skeptical reviewer, domain
analyst, synthesis writer, citation auditor, and risk reviewer.

## Routing

The adaptive router uses deterministic signals from the question and run settings:

- question complexity
- URL count and available source count
- comparison language
- legal, medical, financial, and policy terms
- technical implementation language
- date-sensitive language
- broad or open-ended phrasing
- missing URLs
- link-following settings

Simple summarization skips heavyweight contradiction, risk, and final review stages. Comparative
work requires source triage, evidence collection, synthesis, and citation review. Legal,
financial, medical, policy, freshness-sensitive, and technical due-diligence questions use more
conservative confidence rules and add risk or failure-mode review where appropriate.

## Artifacts

Each run can write:

- `task_graph.json`
- `task_graph.md`
- `stage_outputs.json`
- `specialist_findings.md`
- `orchestration_summary.json`
- `orchestration_summary.md`

The JSON artifacts are intended for API consumers and regression tests. The Markdown artifacts
are intended for operators reviewing why a stage ran, skipped, failed, or warned.

## API

Backend-only endpoints:

- `POST /orchestration/preview`
- `GET /runs/{thread_id}/task-graph`
- `GET /runs/{thread_id}/stage-outputs`

`/run` executes orchestration after source prefetch and before the agent prompt is built, so the
agent receives source-aware instructions while existing required artifacts remain unchanged.
