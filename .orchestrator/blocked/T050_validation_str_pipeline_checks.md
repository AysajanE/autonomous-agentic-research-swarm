---
task_id: T050
title: "Validation: canonical STR integrity, coverage, and benchmark reconciliation"
workstream: W5
task_kind: validation
allow_network: false
role: Worker
priority: high
dependencies:
  - "T030"
  - "T035"
  - "T040"
  - "T048"
  - "T049"
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

# Task T050 — Validation: canonical STR integrity, coverage, and benchmark reconciliation

## Context

Before any release analysis or writing, the repo needs deterministic checks on the canonical rollup panel, the authoritative L1 rent decomposition, and the secondary vendor benchmark. This task is the release firewall for empirical integrity.

Post-`2026-04-09` evidence shows that the old vendor-only/day-universe mismatch has been materially repaired, but the remaining `T050` failure is now dominated by matched-key methodology differences, not just missing attribution. `T050` must therefore follow the W0 benchmark policy locked in `T048` and use the component-level audit surface from `T049` instead of implicitly treating vendor `rent_paid` as the same object as canonical on-chain `rent_paid_eth`.

## Assignment

- Workstream: W5 Validation
- Assigned role: Worker
- Suggested branch/worktree name: `T050_validate_str_pipeline`
- Allowed paths: validation code plus the three required JSON/Markdown report pairs
- Stop conditions: block with `@human` when a failure implies further contract, registry, or source-priority changes beyond the locked T048 policy

## Inputs

- `docs/protocol.md`
- `contracts/data_dictionary.md`
- `contracts/decisions.md`
- `.orchestrator/handoff/H048_t050_contract_resolution_blocker.md`
- `data/processed_manifest/vendor_daily_rollup_panel_<YYYY-MM-DD>.json`
- `data/processed_manifest/daily_l1_rent_decomposition_<YYYY-MM-DD>.json`
- `data/processed_manifest/daily_rollup_panel_<YYYY-MM-DD>.json`
- `data/processed_manifest/daily_rollup_rent_components_<YYYY-MM-DD>.json`
- `src/analysis/metrics_str.py`

## Outputs

- Validation code: `src/validation/validate_str_pipeline.py`
- Canonical panel validation: `reports/validation/rollup_panel_validation.json` and `.md`
- L1 decomposition validation: `reports/validation/l1_rent_decomposition_validation.json` and `.md`
- Cross-source reconciliation: `reports/validation/cross_source_reconciliation.json` and `.md`

## Success Criteria

- [ ] The validation script is deterministic and does not call the network
- [ ] All three report pairs are produced from local manifests and artifacts
- [ ] Checks cover schema/coverage sanity, decomposition identities, key-universe reconciliation, and benchmark-versus-authoritative reconciliation under the locked T048 policy
- [ ] The validator distinguishes contract-breaking failures from benchmark divergences that are expected and documented by policy
- [ ] Any failure beyond tolerance records the smallest actionable next step without silently redefining canonical rent to match the vendor
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

- State: blocked
- Last updated: 2026-04-10

## Notes / Decisions

- 2026-03-29: v1 rewrite expands T050 from vendor-only checks to the full canonical STR validation bundle.
- 2026-04-10: Blocked at the task-contract level. Post-repair evidence shows the remaining failure is no longer primarily a key-universe or generic attribution defect; `T048` must first lock whether growthepie `rent_paid` is a release gate or a secondary benchmark, and `T049` should expose rollup-day rent components so the resumed validator can attribute matched-key deltas rigorously.
