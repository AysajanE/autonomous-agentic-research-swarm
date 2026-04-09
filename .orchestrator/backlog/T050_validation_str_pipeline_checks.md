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
- State: blocked
- Last updated: 2026-04-09
## Notes / Decisions

- 2026-03-29: v1 rewrite expands T050 from vendor-only checks to the full canonical STR validation bundle.
- 2026-04-08: Claimed by local swarm runtime on branch T050_validation_str_pipeline_checks.
- 2026-04-08: Implemented `src/validation/validate_str_pipeline.py` with deterministic `--sample` and `--as-of YYYY-MM-DD` modes. The script resolves manifest-backed inputs, validates schema/key/identity/reconciliation checks, and emits the three required JSON/Markdown report pairs under `reports/validation/`.
- 2026-04-08: Reproduction and outcomes:
  - `python -m py_compile src/validation/validate_str_pipeline.py` → passed.
  - `python src/validation/validate_str_pipeline.py --sample` → exit `0`; sample validation passed for all three bundles. Cross-source monthly aggregate rent reconciliation was `2.08%`, all sampled top-rollup monthly deltas were `< 4%`, vendor profit identity had `0` violations, and the decomposition total-rent identity matched exactly on all sampled days.
  - `python src/validation/validate_str_pipeline.py --as-of 2026-04-01` → exit `2`; wrote blocked canonical reports because the manifests reference missing local artifacts at `data/processed/growthepie/vendor_daily_rollup_panel.csv`, `data/processed/l1_rent/daily_l1_rent_decomposition.csv`, and `data/processed/panels/daily_rollup_panel.csv`.
  - `make gate` → passed.
- 2026-04-08: Outputs written:
  - `src/validation/validate_str_pipeline.py`
  - `reports/validation/rollup_panel_validation.json`
  - `reports/validation/rollup_panel_validation.md`
  - `reports/validation/l1_rent_decomposition_validation.json`
  - `reports/validation/l1_rent_decomposition_validation.md`
  - `reports/validation/cross_source_reconciliation.json`
  - `reports/validation/cross_source_reconciliation.md`
- 2026-04-08: Blocker and minimal next step:
  - `@human` Materialize the manifest-backed processed CSVs for as-of `2026-04-01` in this worktree, or rerun the producing ETL in a workspace where those files exist, then rerun `python src/validation/validate_str_pipeline.py --as-of 2026-04-01`.
- 2026-04-08: Assumptions / limitations:
  - The tracked sample CSVs are sufficient to verify the validation logic and sample-safe gates, but they are not a substitute for the canonical release-firewall run.
  - Because this execution did not record a new swarm runtime manifest, Operator will need the exact commands/outcomes above to log a durable run manifest once the canonical artifacts are available.
- 2026-04-08: Repair rerun after Operator hydrated the canonical manifest-backed CSVs locally:
  - `python src/validation/validate_str_pipeline.py --as-of 2026-04-01` → exit `1`; refreshed the three stable canonical report pairs from local manifests and artifacts instead of writing blocked input-resolution reports.
  - `reports/validation/rollup_panel_validation.{json,md}` → status `fail`; authoritative panel schema, primary-key uniqueness, required non-null, and `metrics_str` compatibility passed, but key coverage failed with `11,164` authoritative keys versus `12,322` vendor keys and `1,158` vendor-only rollup-days.
  - `reports/validation/l1_rent_decomposition_validation.{json,md}` → status `pass`; schema, primary-key uniqueness, required non-null, total-rent identity, and panel-date coverage all passed for `1,551` days.
  - `reports/validation/cross_source_reconciliation.{json,md}` → status `fail`; vendor schema/key/non-null checks passed, but vendor profit identity failed on `543` rows and monthly reconciliation failed because the vendor and authoritative panels do not share the same key coverage.
- 2026-04-08: Canonical failure details for the minimal unblock:
  - Vendor-only key coverage is concentrated in `arbitrum` (`368` dates from `2022-01-01` to `2023-01-03`), `zksync_era` (`540` dates from `2023-03-24` to `2026-01-30`), `linea` (`248` dates from `2023-07-13` to `2026-03-26`), and `taiko` (`2` dates from `2025-11-29` to `2025-11-30`).
  - Vendor profit identity violations are concentrated in `starknet` (`508` rows, max abs diff `82.418787225` ETH, `2024-02-26` to `2025-10-17`), with smaller clusters in `zksync_era` (`29` rows, max abs diff `1.0336628657304` ETH) and `linea` (`6` rows, max abs diff `62.46171051244441` ETH).
  - On matched keys, the cross-source summary reports vendor rent `87,176.30702620494` ETH versus authoritative rent `78,744.6648188145` ETH, for an aggregate absolute delta of `10.707572667673433%`, above the protocol’s `5–10%` target tolerance band.
- 2026-04-08: Reproduction and gates for the repair run:
  - `python src/validation/validate_str_pipeline.py --as-of 2026-04-01`
  - `make gate` → passed after the canonical rerun.
- 2026-04-08: Outputs refreshed by the repair run:
  - `reports/validation/rollup_panel_validation.json`
  - `reports/validation/rollup_panel_validation.md`
  - `reports/validation/l1_rent_decomposition_validation.json`
  - `reports/validation/l1_rent_decomposition_validation.md`
  - `reports/validation/cross_source_reconciliation.json`
  - `reports/validation/cross_source_reconciliation.md`
- 2026-04-08: Blocker and minimal next step:
  - `@human` Confirm whether the vendor-versus-authoritative rollup universe and row-omission behavior are expected to differ for `arbitrum`, `zksync_era`, `linea`, and `taiko`, and whether vendor `profit_eth` changed semantics for `starknet`, `linea`, or `zksync_era`. If these are not expected source-definition differences, fix the upstream attribution/registry logic and rerun `python src/validation/validate_str_pipeline.py --as-of 2026-04-01`.
- 2026-04-08: Assumptions / limitations for the repair run:
  - No change to `src/validation/validate_str_pipeline.py` was required; the validator deterministically reflected the canonical manifest-backed inputs now present in this worktree.
  - This repair execution was not recorded via a fresh `scripts/swarm.py` runtime, so Operator should capture the exact commands and outcomes above in a durable run manifest before any future review attempt after the blocker is resolved.
- 2026-04-09: Operator reopened T050 to `active` on rebased branch `T050-rerun-20260409` after preserving the stale diagnostic branch separately. The earlier review-ready promotion predates the current canonical `main` state and must be re-executed against the synced 2026-04-08 manifests before review.
- 2026-04-09: Refined `src/validation/validate_str_pipeline.py` diagnostics so the stable reports now group key-coverage mismatches by rollup/date range and still emit matched-key reconciliation statistics even when source coverage differs.
- 2026-04-09: Reproduction and outcomes on the synced `2026-04-08` manifests:
  - `python -m py_compile src/validation/validate_str_pipeline.py` → passed.
  - `python src/validation/validate_str_pipeline.py --sample` → exit `0`; sample validation still passes after the diagnostic refinement.
  - `python src/validation/validate_str_pipeline.py --as-of 2026-04-08` → exit `1`; refreshed the three canonical report pairs against the current processed manifests and recorded an actionable reconciliation blocker instead of the stale `2026-04-01` state.
  - `make gate` → passed.
- 2026-04-09: Refreshed canonical output status:
  - `reports/validation/rollup_panel_validation.{json,md}` → status `fail`; authoritative schema, primary-key uniqueness, required non-null, and `metrics_str` compatibility passed, but key coverage still fails with `11,295` authoritative keys versus `12,420` vendor keys and `1,125` vendor-only rollup-days.
  - `reports/validation/l1_rent_decomposition_validation.{json,md}` → status `pass`; schema, primary-key uniqueness, required non-null, total-rent identity, and panel-date coverage all passed for `1,558` days through `2026-04-07`.
  - `reports/validation/cross_source_reconciliation.{json,md}` → status `fail`; vendor schema/key/non-null checks passed, vendor profit identity now passes with `0` violations, but reconciliation still fails with `1,125` vendor-only keys, `32` monthly aggregate deltas above the protocol’s `10%` ceiling, `247` rollup-month violations, and matched-key aggregate rent difference of `10.667051775329645%`.
- 2026-04-09: Canonical failure details for the minimal unblock:
  - Vendor-only key coverage is concentrated in `zksync_era` (`538` dates from `2023-03-24` to `2026-01-30`), `arbitrum` (`368` dates from `2022-01-01` to `2023-01-03`), and `linea` (`219` dates from `2023-07-13` to `2026-03-26`); there are no authoritative-only keys.
  - The matched-key monthly reconciliation still exceeds tolerance after excluding unmatched rows. The report records `32` aggregate month violations and `247` rollup-month violations, led by early-2022 `optimism` deltas and a matched-key aggregate rent gap of `10.667051775329645%` (`88,530.13334555583` vendor ETH versus `79,996.83006400586` authoritative ETH).
- 2026-04-09: Outputs refreshed by the rerun:
  - `src/validation/validate_str_pipeline.py`
  - `reports/validation/rollup_panel_validation.json`
  - `reports/validation/rollup_panel_validation.md`
  - `reports/validation/l1_rent_decomposition_validation.json`
  - `reports/validation/l1_rent_decomposition_validation.md`
  - `reports/validation/cross_source_reconciliation.json`
  - `reports/validation/cross_source_reconciliation.md`
- 2026-04-09: Blocker and minimal next step:
  - `@human` Confirm whether the vendor-only `arbitrum`, `zksync_era`, and `linea` rollup-days in the `2026-04-08` vendor extract are expected to be excluded from the canonical panel. If not, fix the upstream registry/attribution/row-omission logic so the rollup-day key universe aligns, rerun `python src/validation/validate_str_pipeline.py --as-of 2026-04-08`, and then inspect the matched-key monthly rent deltas that still exceed the `10%` reconciliation ceiling.
- 2026-04-09: Assumptions / limitations for the rerun:
  - This execution was run directly in the worktree rather than through a fresh `scripts/swarm.py` runtime, so no new durable swarm run manifest was created.
  - Operator should capture the commands and outcomes above in a run manifest before any future review attempt once the reconciliation blocker is resolved.
- 2026-04-09: @human Runtime blocked: task_marked_blocked. Run manifest: reports/status/swarm_runs/T050_20260409T115203Z.json.
