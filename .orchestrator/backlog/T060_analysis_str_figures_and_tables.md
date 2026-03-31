---
task_id: T060
title: "Analysis: release STR figures and tables from validated artifacts"
workstream: W6
task_kind: analysis
allow_network: false
role: Worker
priority: high
dependencies:
  - "T040"
  - "T050"
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "src/analysis/build_str_release_outputs.py"
  - "reports/figures/str_ecosystem_timeseries.svg"
  - "reports/figures/str_post_dencun_regimes.svg"
  - "reports/tables/str_regime_summary.csv"
  - "reports/tables/str_regime_summary.md"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "src/etl/"
  - "src/analysis/metrics_str.py"
  - "data/raw/"
  - "reports/catalog.yaml"
outputs:
  - "src/analysis/build_str_release_outputs.py"
  - "reports/figures/str_ecosystem_timeseries.svg"
  - "reports/figures/str_post_dencun_regimes.svg"
  - "reports/tables/str_regime_summary.csv"
  - "reports/tables/str_regime_summary.md"
gates:
  - "make gate"
stop_conditions:
  - "Validated inputs are missing or failing"
  - "Need to change metric definitions"
---

# Task T060 — Analysis: release STR figures and tables from validated artifacts

## Context

This task turns validated empirical artifacts into the minimum release analysis bundle. It replaces the old sample-only figure slice with stable figures and tables that the paper can include directly.

## Assignment

- Workstream: W6 Analysis
- Assigned role: Worker
- Suggested branch/worktree name: `T060_str_release_outputs`
- Allowed paths: one release-output builder script plus the locked figure/table filenames
- Stop conditions: block with `@human` if the requested outputs require protocol or validation changes

## Inputs

- `reports/validation/rollup_panel_validation.json`
- `reports/validation/l1_rent_decomposition_validation.json`
- `reports/validation/cross_source_reconciliation.json`
- `data/processed_manifest/daily_rollup_panel_<YYYY-MM-DD>.json`
- `data/processed_manifest/daily_l1_rent_decomposition_<YYYY-MM-DD>.json`
- `src/analysis/metrics_str.py`

## Outputs

- Analysis code: `src/analysis/build_str_release_outputs.py`
- Figures:
  - `reports/figures/str_ecosystem_timeseries.svg`
  - `reports/figures/str_post_dencun_regimes.svg`
- Tables:
  - `reports/tables/str_regime_summary.csv`
  - `reports/tables/str_regime_summary.md`

## Success Criteria

- [ ] The script reads validated local artifacts only and makes no network calls
- [ ] Both figures and both table formats are generated at the locked filenames
- [ ] The Markdown table fragment is suitable for direct inclusion in the paper
- [ ] Output filenames are stable and deterministic across reruns
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any interpretation caveat needed by T070 is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- `python src/analysis/build_str_release_outputs.py --sample`
- `python src/analysis/build_str_release_outputs.py --as-of YYYY-MM-DD`

## Status

- State: backlog
- Last updated: 2026-03-29

## Notes / Decisions

- 2026-03-29: v1 rewrite expands T060 from one sample figure to the minimum release figure-and-table bundle.
