---
task_id: T040
title: "Metrics: STR computation module + unit tests (sample data only)"
workstream: W4
role: Worker
priority: high
dependencies:
  - "T030"
allowed_paths:
  - "src/analysis/metrics_str.py"
  - "tests/test_metrics_str.py"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "src/etl/"
  - "data/raw/"
outputs:
  - "src/analysis/metrics_str.py"
  - "tests/test_metrics_str.py"
gates:
  - "make gate"
  - "make test"
stop_conditions:
  - "Contract ambiguity"
---

# Task T040 — Metrics: STR computation module + unit tests (sample data only)

## Context

Implement the core primary metric from `docs/protocol.md` in code:

- `STR_t = (Σ_i RentPaid_{i,t}) / (Σ_i L2Fees_{i,t})` (daily)

This task focuses on **deterministic metric construction** using only the committed golden sample produced by T030 (no network, no full data builds).

## Inputs

- `docs/protocol.md` (read-only): STR definition + ETH-native priority
- `contracts/schemas/panel_schema_str_v1.yaml` (read-only): required field names/units
- `data/samples/growthepie/vendor_daily_rollup_panel_sample.csv` (committed)

## Outputs

- `src/analysis/metrics_str.py`
  - A small, testable module that can:
    - load the sample panel (CSV)
    - compute rollup-level STR components and ecosystem aggregate STR by day
    - handle missing/zero denominators explicitly (per the schema decision)
- `tests/test_metrics_str.py`
  - Unit tests using only `data/samples/` inputs.

## Success Criteria

- [ ] Metric implementation matches protocol definition and uses ETH-native series
- [ ] Tests cover:
  - basic STR calculation on a known small slice
  - handling of zero/empty denominator days (explicit behavior)
- [ ] `make gate` and `make test` pass

## Status

- State: backlog
- Last updated: 2026-01-22

## Notes / Decisions

- 2026-01-22: Task created (Planner) to lock STR math in code with tests before analysis.
