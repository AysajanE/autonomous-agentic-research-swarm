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
  - "T051"
  - "T052"
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
- 2026-04-10: Operator confirmed the resumed `T049`/`T050` path now reruns cleanly on a coherent `2026-04-09` surface. In `/Users/aeziz-local/Research/swarm-t050-20260409/wt-T050-final`, `make gate` passed and `python src/validation/validate_str_pipeline.py --as-of 2026-04-09` exited `1` only because `cross_source_reconciliation` still fails scientifically, not because of missing inputs, stale manifests, or validator/runtime defects.
- 2026-04-10: Clean rerun result on the corrected surface:
  - `reports/validation/rollup_panel_validation.{json,md}` -> `pass`
  - `reports/validation/l1_rent_decomposition_validation.{json,md}` -> `pass`
  - `reports/validation/cross_source_reconciliation.{json,md}` -> `fail`
  - `mismatched_key_count = 0`
  - `matched_row_count = 12434`
  - `overall_authoritative_total_rent_eth = 144655.91426627047`
  - `overall_vendor_total_rent_eth = 132310.156968212`
  - `overall_aggregate_pct_difference = 8.534567950905517%`
- 2026-04-10: Component audit now cleanly separates explained from unexplained benchmark divergence:
  - `starknet` is explained under the locked T048 policy because canonical excess `13626.815080978078 ETH` matches `batch_submissions_eth + proof_submissions_eth` within `2.483e-06 ETH`
  - `taiko` remains the only material unexplained rollup: vendor exceeds canonical by `1550.802261591275 ETH` (`63.15693216795472%` of canonical Taiko rent)
  - after excluding the explained Starknet methodology difference, unresolved aggregate gap falls to `0.9945498153224685%`, but `taiko` still drives `9` unexplained monthly aggregate violations
- 2026-04-10: Operator ran an upstream Taiko selector experiment in `/Users/aeziz-local/Research/wt-T049` by adding `0xe4882785` (`proposeBlocksV2Conditionally`) as an exact-window Taiko supplement in `src/etl/build_l1_rent_panel.py`, rebuilding the authoritative `2026-04-09` snapshot, and rehydrating `wt-T050-final` from those refreshed outputs. The observed-window cache confirmed the selector exists locally (`77431` observed txs from `2025-03-18T10:36:11Z` to `2025-05-21T10:34:23Z`), but the Taiko monthly canonical surface and final `T050` failure did not materially move. This falsifies that selector as the root cause of the remaining blocker.
- 2026-04-10: Minimal next step: do not keep rerunning `T050` under the same Taiko hypothesis. The remaining blocker is a dedicated Taiko benchmark-resolution question:
  1. canonical is still missing a non-L2BEAT Taiko contract/selector surface that belongs in authoritative on-chain rent, or
  2. growthepie Taiko `rent_paid` is counting a broader object than canonical on-chain fee accounting and requires an explicit W0-reviewed exception before `T050` can pass under the locked benchmark policy.
- 2026-04-10: This Taiko-first blocker is now stale. A deeper Starknet root-cause pass established that the remaining dominant failure is a Starknet-specific methodology error, not another missing-attribution hunt. On the refreshed `wt-T049` surface, Taiko canonical rent increased by `1307.915367717757 ETH`, reducing the Taiko gap to `-242.886893873518 ETH`, while the overall aggregate widened mechanically because vendor totals were unchanged.
- 2026-04-10: Current final blocker is recorded in `.orchestrator/handoff/H052_starknet_shared_sharp_rootcause_2026-04-10.md`:
  - vendor Starknet `rent_paid_eth` matches canonical `state_updates_eth` to within `2.483e-06 ETH`
  - canonical Starknet excess matches canonical `batch_submissions_eth + proof_submissions_eth` to within floating noise
  - those excess tx families live on shared SHARP verifier-stack contracts, so the remaining problem is canonical over-attribution of shared SHARP costs, not a stale Starknet selector set
- 2026-04-10: `T050` must remain blocked until:
  1. `T051` locks the Starknet shared-settlement attribution contract in W0, and
  2. `T052` repairs the Starknet ETL implementation under that contract and rebuilds the authoritative artifacts.
