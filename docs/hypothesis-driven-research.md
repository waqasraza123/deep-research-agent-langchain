# Hypothesis-Driven Research

The backend includes an offline-only hypothesis subsystem under
`deep_research_agent.hypotheses`. It proposes possible answers from the research question,
subquestions, `plan.md`, `notes.md`, `report.md`, `sources.json`, source audit artifacts,
retrieval context packs, evidence ledger, and synthesis output when available.

The subsystem does not make hidden model calls. It tests each hypothesis with deterministic
signals: phrase overlap, keyword overlap, entity overlap, numeric/date overlap, source quality,
citation readiness, contradiction warnings, source diversity, primary-source signals, and freshness
signals.

Generated artifacts:

- `hypotheses.json`
- `hypotheses.md`
- `hypothesis_tests.json`
- `hypothesis_tests.md`
- `hypothesis_graph.json`
- `hypothesis_graph.md`
- `confidence_updates.json`
- `confidence_updates.md`

Confidence starts conservatively and is adjusted by supporting and opposing evidence, authority,
freshness, primary-source availability, evidence specificity, contradiction count, unsupported
strong language, missing source types, and domain sensitivity.

Limitations:

- It is deterministic evidence triage, not proof.
- Weak or missing sources intentionally produce low confidence.
- Legal, medical, financial, security, and policy-sensitive conclusions require human review.
- The subsystem does not replace `report.md`; it writes separate artifacts for later synthesis and
  evaluation stages.
