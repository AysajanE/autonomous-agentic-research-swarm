---
task_id: T060
title: "Analysis: generate STR timeseries figure from sample panel"
workstream: W6
role: Worker
priority: medium
dependencies:
  - "T040"
  - "T050"
allowed_paths:
  - "src/analysis/plot_str_timeseries_sample.py"
  - "reports/figures/str_timeseries_sample.svg"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "src/etl/"
  - "src/analysis/metrics_str.py"
  - "data/raw/"
outputs:
  - "src/analysis/plot_str_timeseries_sample.py"
  - "reports/figures/str_timeseries_sample.svg"
gates:
  - "make gate"
stop_conditions:
  - "Missing sample inputs"
---

# Task T060 — Analysis: generate STR timeseries figure from sample panel

## Context

Create the first “vertical slice” research output artifact: a STR timeseries figure generated from the deterministic sample dataset.

This is primarily a workflow test:
- analysis code reads local inputs only (no network),
- outputs go to `reports/figures/`,
- and the figure can be regenerated via a single command.

## Inputs

- `data/samples/growthepie/vendor_daily_rollup_panel_sample.csv`
- `src/analysis/metrics_str.py` (read-only; produced by T040)
- `reports/validation/vendor_panel_validation.json` (optional; used to gate/annotate)

## Outputs

- `src/analysis/plot_str_timeseries_sample.py`
  - Deterministic (no randomness; no network).
  - Prefer stdlib-only output (e.g., write an SVG directly) to avoid adding heavy plotting deps in the first slice.
- `reports/figures/str_timeseries_sample.svg`
  - Stable filename.
  - Should include a simple title and axis labels (date vs STR).

## Success Criteria

- [ ] Running the script produces `reports/figures/str_timeseries_sample.svg`
- [ ] Output is deterministic for the committed sample
- [ ] `make gate` passes

## Validation / Commands

- `make gate`
- Example:
  - `python src/analysis/plot_str_timeseries_sample.py`

## Status

- State: backlog
- Last updated: 2026-01-22

## Notes / Decisions

- 2026-01-22: Task created (Planner) to produce the first reproducible figure artifact.
