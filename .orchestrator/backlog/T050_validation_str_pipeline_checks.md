---
task_id: T050
title: "Validation: canonical STR pipeline checks and cross-source reconciliation"
workstream: W5
task_kind: validation
allow_network: false
role: Worker
priority: high
dependencies:
  - "T030"
  - "T035"
  - "T040"
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "src/validation/validate_str_pipeline.py"
  - "reports/validation/rollup_panel_validation.json"
  - "reports/validation/rollup_panel_validation.md"
  - "reports/validation/l1_rent_decomposition_validation.json"
  - "reports/validation/l1_rent_decomposition_validation.md"
  - "reports/validation/cross_source_reconciliation.json"
  - "reports/validation/cross_source_reconciliation.md"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "src/etl/"
  - "data/raw/"
outputs:
  - "src/validation/validate_str_pipeline.py"
  - "reports/validation/rollup_panel_validation.json"
  - "reports/validation/rollup_panel_validation.md"
  - "reports/validation/l1_rent_decomposition_validation.json"
  - "reports/validation/l1_rent_decomposition_validation.md"
  - "reports/validation/cross_source_reconciliation.json"
  - "reports/validation/cross_source_reconciliation.md"
gates:
  - "make gate"
stop_conditions:
  - "Validation failure beyond tolerance"
  - "Canonical processed manifests are missing"
---

# Task T050 — Validation: canonical STR pipeline checks and cross-source reconciliation

## Context

Before any release analysis or writing, the repo needs deterministic checks on the canonical rollup panel, the authoritative L1 rent decomposition, and the off-chain vendor cross-check. This task is the release firewall for empirical integrity.

## Assignment

- Workstream: W5 Validation
- Assigned role: Worker
- Suggested branch/worktree name: `T050_validate_str_pipeline`
- Allowed paths: validation code plus the three required JSON/Markdown report pairs
- Stop conditions: block with `@human` when a failure implies contract, registry, or source-priority changes

## Inputs

- `docs/protocol.md`
- `contracts/data_dictionary.md`
- `contracts/decisions.md`
- `data/processed_manifest/vendor_daily_rollup_panel_<YYYY-MM-DD>.json`
- `data/processed_manifest/daily_l1_rent_decomposition_<YYYY-MM-DD>.json`
- `data/processed_manifest/daily_rollup_panel_<YYYY-MM-DD>.json`
- `src/analysis/metrics_str.py`

## Outputs

- Validation code: `src/validation/validate_str_pipeline.py`
- Canonical panel validation: `reports/validation/rollup_panel_validation.json` and `.md`
- L1 decomposition validation: `reports/validation/l1_rent_decomposition_validation.json` and `.md`
- Cross-source reconciliation: `reports/validation/cross_source_reconciliation.json` and `.md`

## Success Criteria

- [ ] The validation script is deterministic and does not call the network
- [ ] All three report pairs are produced from local manifests and artifacts
- [ ] Checks cover schema/coverage sanity, decomposition identities, and vendor-versus-authoritative reconciliation
- [ ] Any failure beyond tolerance records the smallest actionable next step
- [ ] `make gate` passes
- [ ] Downstream analysis and writing tasks are not advanced until this bundle passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any non-obvious reconciliation caveat is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- `python src/validation/validate_str_pipeline.py --sample`
- `python src/validation/validate_str_pipeline.py --as-of YYYY-MM-DD`

## Status
- State: active
- Last updated: 2026-04-08
## Notes / Decisions

- 2026-03-29: v1 rewrite expands T050 from vendor-only checks to the full canonical STR validation bundle.
- 2026-04-08: Claimed by local swarm runtime on branch T050_validation_str_pipeline_checks.
