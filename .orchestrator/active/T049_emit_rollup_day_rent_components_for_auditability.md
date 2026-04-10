---
task_id: T049
title: "Emit rollup-day rent component surfaces for auditability"
workstream: W2
task_kind: data
allow_network: true
role: Worker
priority: high
dependencies:
  - "T035"
  - "T048"
allowed_paths:
  - "src/etl/build_l1_rent_panel.py"
  - "data/raw/l1_rent/"
  - "data/raw_manifest/l1_rent_<YYYY-MM-DD>.json"
  - "data/processed/l1_rent/"
  - "data/processed/panels/"
  - "data/processed_manifest/daily_l1_rent_decomposition_<YYYY-MM-DD>.json"
  - "data/processed_manifest/daily_rollup_panel_<YYYY-MM-DD>.json"
  - "data/processed_manifest/daily_rollup_rent_components_<YYYY-MM-DD>.json"
  - "data/samples/l1_rent/"
  - "data/samples/panels/"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "registry/"
outputs:
  - "src/etl/build_l1_rent_panel.py"
  - "data/processed/l1_rent/daily_rollup_rent_components.csv"
  - "data/processed_manifest/daily_rollup_rent_components_<YYYY-MM-DD>.json"
  - "data/samples/l1_rent/daily_rollup_rent_components_sample.csv"
gates:
  - "make gate"
stop_conditions:
  - "Need protocol or contract changes"
  - "Required lineage cannot be captured from the existing raw snapshot"
  - "Would require edits outside W2-owned paths"
---

# Task T049 — Emit rollup-day rent component surfaces for auditability

## Context

The repaired `T046`/`T050` sequence showed that once the key universe matches, the hard remaining questions are component-level. Today the canonical pipeline emits a final daily rent number, but it does not persist a compact audit surface that lets validation separate:

- batch submissions
- proofs / custom settlement
- state updates
- execution-layer burn and tips
- blob burn

Without that component surface, every residual benchmark delta risks another expensive rerun or another ad hoc notebook. The repo needs a durable rollup-day component artifact so future validation can identify root cause directly instead of inferring from replay logs.

## Assignment

- Workstream: W2 Data: on-chain
- Assigned role: Worker
- Suggested branch/worktree name: `T049_rollup_day_rent_components`
- Allowed paths: authoritative ETL plus raw/processed/manifests/samples under the L1 rent path
- Stop conditions: block with `@human` instead of silently redefining metric semantics or inventing uncontracted component labels

## Inputs

- `docs/protocol.md`
- `contracts/data_dictionary.md`
- `contracts/decisions.md`
- `.orchestrator/handoff/H048_t050_contract_resolution_blocker.md`
- `.orchestrator/done/T035_onchain_l1_rent_etl_and_decomposition.md`
- `data/raw_manifest/l1_rent_<YYYY-MM-DD>.json`
- `data/processed_manifest/vendor_daily_rollup_panel_<YYYY-MM-DD>.json`

## Outputs

- Update `src/etl/build_l1_rent_panel.py` so the canonical run also emits `data/processed/l1_rent/daily_rollup_rent_components.csv`
- Emit a matching processed manifest `data/processed_manifest/daily_rollup_rent_components_<YYYY-MM-DD>.json`
- Emit a tracked sample at `data/samples/l1_rent/daily_rollup_rent_components_sample.csv`

Required component coverage in the new artifact:

- `date_utc`
- `rollup_id`
- `batch_submissions_eth`
- `proof_submissions_eth`
- `state_updates_eth`
- `blob_fee_burn_eth`
- `execution_base_fee_burn_eth`
- `execution_priority_fee_eth`
- `rent_paid_eth`

The component surface must sum back to canonical `rent_paid_eth` under the T048 contract.

## Success Criteria

- [ ] The canonical ETL emits a manifest-backed rollup-day component surface from the authoritative raw snapshot
- [ ] Component rows reconcile exactly to canonical `rent_paid_eth`
- [ ] The new surface is deterministic and sample-backed
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any remaining component attribution caveat is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- `python src/etl/build_l1_rent_panel.py --run-date YYYY-MM-DD`

## Status

- State: active
- Last updated: 2026-04-10

## Notes / Decisions

- 2026-04-10: Created to eliminate the current replay tax for matched-key reconciliation failures by persisting rollup-day rent components directly.
- 2026-04-10: Blocked under the task stop conditions. The current T048/T049 contract requires one `daily_rollup_rent_components` table to carry both a tx-family decomposition (`batch_submissions_eth`, `proof_submissions_eth`, `state_updates_eth`) and a fee-class decomposition (`blob_fee_burn_eth`, `execution_base_fee_burn_eth`, `execution_priority_fee_eth`) while also requiring the component columns to sum exactly to `rent_paid_eth`. Those are two parallel identities, so summing all seven component columns would double-count canonical rent. Implementing either interpretation without a W0 amendment would silently redefine the metric contract. Smallest `@human` unblocker: approve a narrow W0 contract clarification that the tx-family columns must sum to `rent_paid_eth` and the fee-class columns must separately sum to `rent_paid_eth`.
- 2026-04-10: Unblocked after W0 clarification committed as `a714958`/`57e37c5`. `daily_rollup_rent_components` now has a valid target contract: tx-family columns reconcile to canonical `rent_paid_eth`, and fee-class columns reconcile separately to the same canonical total.
- 2026-04-10: Implemented the Taiko historical-attribution repair in `src/etl/build_l1_rent_panel.py` by adding legacy tracked-call supplements for the missing `0x68d30f47f19c07bccef4ac7fae2dc12fca3e0dc9` surface (`0x10d008bd`, `0xef16e845`, `0x0c8f4a10`, `0x440b6e18`) plus the tiny `0x06a9ab27c7e2255df1815e6cc0168d7755feb19a/0x8778209d` legacy window. This was derived from BigQuery exact-window evidence and Taiko deployment docs identifying `0x68d...` as the historical `labprover` / ProverSet surface.
- 2026-04-10: Rebuilt the authoritative `2026-04-09` surface with `python src/etl/build_l1_rent_panel.py --run-date 2026-04-09 --resume-manifested-run`. The run completed successfully and rewrote `data/raw/l1_rent/2026-04-09`, `data/raw_manifest/l1_rent_2026-04-09.json`, `data/processed/l1_rent/daily_l1_rent_decomposition.csv`, `data/processed/l1_rent/daily_rollup_rent_components.csv`, `data/processed/panels/daily_rollup_panel.csv`, the matching samples, and the matching processed manifests.
- 2026-04-10: `make gate` passes in `wt-T049` after the rebuild.
- 2026-04-10: Outcome of the Taiko repair: canonical Taiko total rent increased from `2455.474337270198 ETH` to `3763.389704987955 ETH` (`+1307.915367717757 ETH`), reducing the Taiko canonical-vendor gap from about `-1550.80 ETH` to `-242.89 ETH`. The remaining global benchmark blocker is no longer Taiko-dominant: matched keys are still exact (`12434`, `0` mismatches), and excluding `starknet` the refreshed aggregate gap is only `0.020646%`. T050 itself was not rerun from this task worktree because writing validation artifacts is outside T049-owned paths.
