# Quantitative Intelligence

The backend includes an offline quantitative subsystem under
`deep_research_agent.quantitative`. It extracts numeric values and numeric claims from fetched
source text, `notes.md`, and `report.md`; profiles document tables and CSV files; builds conservative
metric comparisons; runs simple safe calculations; and writes warnings when report numbers are
unsupported or not directly comparable.

Generated artifacts:

- `quantitative_profile.json` and `quantitative_profile.md`
- `numeric_claims.json` and `numeric_claims.md`
- `table_profiles.json`
- `csv_profiles.json`
- `quantitative_comparisons.json` and `quantitative_comparisons.md`
- `quantitative_warnings.md`

API endpoints:

- `POST /runs/{thread_id}/quantitative/rebuild`
- `GET /runs/{thread_id}/quantitative`
- `GET /runs/{thread_id}/numeric-claims`
- `GET /runs/{thread_id}/quantitative-warnings`

The subsystem is deterministic and does not call a model. It uses standard-library parsing for
numbers, percentages, currencies, ratios, ranges, dates, version numbers, CSVs, and table-like rows.
It deliberately separates version strings such as `v0.2.14` from benchmark values.

## Limitations

Quantitative extraction is conservative heuristic parsing, not statistical inference. Ambiguous
units, undocumented benchmark setups, mixed currencies, and prices without billing periods are
reported as warnings instead of being normalized away. The subsystem never fabricates missing
numbers and does not fetch live metrics. If values are incomparable, downstream report writing and
evaluation should state that clearly.
