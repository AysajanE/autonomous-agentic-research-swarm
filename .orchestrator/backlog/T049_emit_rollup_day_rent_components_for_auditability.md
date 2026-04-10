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

- State: backlog
- Last updated: 2026-04-10

## Notes / Decisions

- 2026-04-10: Created to eliminate the current replay tax for matched-key reconciliation failures by persisting rollup-day rent components directly.
