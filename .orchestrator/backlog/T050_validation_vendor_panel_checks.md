---
task_id: T050
title: "Validation: vendor panel checks (coverage + identity + sanity)"
workstream: W5
role: Worker
priority: medium
dependencies:
  - "T030"
  - "T040"
allowed_paths:
  - "src/validation/validate_vendor_panel.py"
  - "reports/validation/vendor_panel_validation.json"
  - "reports/validation/vendor_panel_validation.md"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "src/etl/"
  - "data/raw/"
outputs:
  - "src/validation/validate_vendor_panel.py"
  - "reports/validation/vendor_panel_validation.json"
  - "reports/validation/vendor_panel_validation.md"
gates:
  - "make gate"
stop_conditions:
  - "Validation failure beyond tolerance"
---

# Task T050 — Validation: vendor panel checks (coverage + identity + sanity)

## Context

Before doing any econometrics/figures, we need deterministic “anti-dashboard-science” checks.

This task creates a validation script that consumes only local artifacts (sample panel by default) and emits:
- a machine-readable JSON summary, and
- a short Markdown report with next steps if checks fail.

## Inputs

- `docs/protocol.md` (read-only): tolerances and identity checks
- `data/samples/growthepie/vendor_daily_rollup_panel_sample.csv` (committed)
- `src/analysis/metrics_str.py` (from T040)

## Outputs

- `src/validation/validate_vendor_panel.py`
  - Must be deterministic and must not call the network.
  - Should support `--sample` mode (default) that reads from `data/samples/`.
- `reports/validation/vendor_panel_validation.json`
  - Include pass/fail, summary stats, and pointers to inputs used.
- `reports/validation/vendor_panel_validation.md`
  - Short narrative: what passed/failed + minimal next experiment.

Suggested checks (keep small, but useful):
- Coverage: days present per rollup (on the sample slice)
- Non-negativity: fees/rent/txcount should not be negative
- Accounting identity (vendor): `profit ≈ fees − rent_paid` with tolerance from `docs/protocol.md`
- STR sanity: STR should be finite and interpretable when fees > 0

## Success Criteria

- [ ] Script runs deterministically on the committed sample and writes JSON+MD outputs
- [ ] If checks fail beyond tolerance, the report includes the smallest actionable next step
- [ ] `make gate` passes

## Validation / Commands

- `make gate`
- Example:
  - `python src/validation/validate_vendor_panel.py --sample`

## Status
- State: backlog
- Last updated: 2026-01-22
## Notes / Decisions

- 2026-01-22: Task created (Planner) to add deterministic validation before analysis.
