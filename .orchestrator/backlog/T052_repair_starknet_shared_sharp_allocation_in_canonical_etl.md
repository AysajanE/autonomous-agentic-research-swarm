---
task_id: T052
title: "Repair Starknet shared SHARP attribution in the canonical ETL"
workstream: W2
task_kind: data
allow_network: true
role: Worker
priority: high
dependencies:
  - "T049"
  - "T051"
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
  - "data/processed/panels/daily_rollup_panel.csv"
  - "data/processed_manifest/daily_rollup_rent_components_<YYYY-MM-DD>.json"
  - "data/processed_manifest/daily_rollup_panel_<YYYY-MM-DD>.json"
gates:
  - "make gate"
stop_conditions:
  - "Need protocol or contract changes beyond T051"
  - "An evidence-backed Starknet shared-settlement source or T051-approved fallback cannot be materialized"
  - "Would require edits outside W2-owned paths"
---

# Task T052 — Repair Starknet shared SHARP attribution in the canonical ETL

## Context

The Starknet root-cause investigation showed that the current canonical ETL is over-attributing generic SHARP verifier-stack Ethereum costs directly to Starknet.

Today the Starknet canonical tx universe includes:

- direct Starknet Core `updateState*` surfaces
- generic SHARP contracts for:
  - `registerContinuousMemoryPage`
  - `registerContinuousPageBatch`
  - `verifyMerkle`
  - `verifyFRI`
  - `verifyProofAndRegister`

That raw tracked-tx method is acceptable only if those costs are Starknet-exclusive. The protocol evidence gathered in the Starknet deep dive indicates they are shared / amortized costs that need Starknet-specific treatment rather than direct full-fee attribution.

This task repairs the ETL under the T051 contract so the canonical Starknet surface no longer silently measures the wrong object.

## Assignment

- Workstream: W2 Data: on-chain
- Assigned role: Worker
- Suggested branch/worktree name: `T052_starknet_shared_sharp_allocation`
- Allowed paths: authoritative ETL plus the matching raw/processed/manifests/samples under the L1 rent path
- Stop conditions: block with `@human` instead of inventing a Starknet allocation source or fallback not authorized by T051

## Inputs

- `docs/protocol.md`
- `contracts/data_dictionary.md`
- `contracts/decisions.md`
- `.orchestrator/handoff/H052_starknet_shared_sharp_rootcause_2026-04-10.md`
- `data/raw_manifest/l1_rent_<YYYY-MM-DD>.json`
- `data/processed_manifest/vendor_daily_rollup_panel_<YYYY-MM-DD>.json`
- existing `data/raw/l1_rent/<YYYY-MM-DD>/` Starknet snapshot contents

## Outputs

- Update `src/etl/build_l1_rent_panel.py` so Starknet shared SHARP settlement is handled according to T051, not by naive full raw tx attribution on generic SHARP contracts
- Rebuild the authoritative `daily_rollup_panel` and `daily_rollup_rent_components` surfaces for the chosen run date
- Emit matching manifests and samples
- Capture any remaining Starknet-specific methodological caveat in `.orchestrator/handoff/`

## Success Criteria

- [ ] Canonical Starknet no longer charges full generic SHARP verifier-stack tx fees directly to Starknet unless T051 explicitly authorizes that model
- [ ] The Starknet component surface reflects the T051 contract coherently
- [ ] The authoritative `daily_rollup_panel` and `daily_rollup_rent_components` artifacts rebuild successfully with manifests and samples
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any residual Starknet caveat is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `python src/etl/build_l1_rent_panel.py --run-date YYYY-MM-DD --resume-manifested-run`
- `make gate`

## Status

- State: backlog
- Last updated: 2026-04-10

## Notes / Decisions

- 2026-04-10: Created after the Starknet deep-dive established that the remaining `T050` blocker is a canonical over-attribution of shared SHARP settlement costs, not a missing Starknet contract window.
- 2026-04-10: This task must not proceed before `T051` locks the Starknet shared-settlement contract explicitly.
