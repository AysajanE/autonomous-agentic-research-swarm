---
task_id: T040
title: "Metrics: STR computation module and sample-only tests"
workstream: W4
task_kind: metrics
allow_network: false
role: Worker
priority: high
dependencies:
  - "T035"
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "src/analysis/metrics_str.py"
  - "tests/test_metrics_str.py"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "src/etl/"
  - "src/validation/"
  - "data/raw/"
outputs:
  - "src/analysis/metrics_str.py"
  - "tests/test_metrics_str.py"
gates:
  - "make gate"
  - "make test"
stop_conditions:
  - "Contract ambiguity"
  - "Canonical sample panel is missing"
---

# Task T040 — Metrics: STR computation module and sample-only tests

## Context

This task locks the reusable STR math against the canonical sample panel produced by T035. It is the computation layer for validation, figures, and tables; it is not the release-writing or release-assembly task.

## Assignment

- Workstream: W4 Metrics
- Assigned role: Worker
- Suggested branch/worktree name: `T040_metrics_str`
- Allowed paths: `src/analysis/metrics_str.py`, `tests/test_metrics_str.py`
- Stop conditions: block with `@human` instead of changing definitions for missingness, denominators, or units

## Inputs

- `docs/protocol.md`
- `contracts/data_dictionary.md`
- `contracts/decisions.md`
- `contracts/schemas/panel_schema_str_v1.yaml`
- `data/samples/panels/daily_rollup_panel_sample.csv`

## Outputs

- `src/analysis/metrics_str.py`
- `tests/test_metrics_str.py`

## Success Criteria

- [ ] The module computes rollup-level and ecosystem-level STR from the canonical sample panel
- [ ] Denominator-zero behavior is explicit and matches the locked `NaN` rule
- [ ] Missingness handling matches the locked row-omission rule
- [ ] Tests cover happy-path math, denominator-zero days, and missingness behavior
- [ ] `make gate` and `make test` pass

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any assumptions about sample construction are recorded in the task note or a handoff note
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- `make test`
- `pytest tests/test_metrics_str.py`

## Status
- State: done
- Last updated: 2026-04-08
## Notes / Decisions

- 2026-03-29: v1 rewrite retargets T040 to the canonical panel sample from T035 instead of the vendor-only sample from the old vertical slice.
- 2026-04-08: Claimed by local swarm runtime on branch T040_metrics_str_module_and_tests.
- 2026-04-08: Added `src/analysis/metrics_str.py` with `compute_rollup_str(panel)` and `compute_ecosystem_str(panel)`. Both helpers require `date_utc`, `rollup_id`, `l2_fees_eth`, and `rent_paid_eth`; rows missing either metric column are omitted before aggregation; zero denominators return `NaN` instead of `0`.
- 2026-04-08: Added `tests/test_metrics_str.py` covering canonical sample-panel rollup/ecosystem values, row-omission missingness behavior, and denominator-zero days.
- 2026-04-08: Reproduction commands run:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_metrics_str.py` → passed (`4 passed`)
  - `make gate` → passed
  - `make test` → passed (`35 tests`)
- 2026-04-08: Files changed: `src/analysis/metrics_str.py`, `tests/test_metrics_str.py`.
- 2026-04-08: Limitation/process note: plain `pytest tests/test_metrics_str.py` on this workstation fails before collection because auto-loaded third-party `web3` plugins crash in the host environment. Repo code is green under `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest ...` and the declared `make test` path. This run was not recorded by the local swarm runtime, so Operator still needs to capture a durable run manifest before review.
- 2026-04-08: Runtime passed: outputs, gates, manifests, and run manifest are present. Ready for Judge review. Run manifest: reports/status/swarm_runs/T040_20260408T150755Z.json
- 2026-04-08: Judge approved; review log: reports/status/reviews/T040_20260408T151504Z.json
