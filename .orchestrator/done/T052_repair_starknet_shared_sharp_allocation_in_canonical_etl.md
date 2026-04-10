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
- State: done
- Last updated: 2026-04-10
## Notes / Decisions

- 2026-04-10: Created after the Starknet deep-dive established that the remaining `T050` blocker is a canonical over-attribution of shared SHARP settlement costs, not a missing Starknet contract window.
- 2026-04-10: This task must not proceed before `T051` locks the Starknet shared-settlement contract explicitly.
- 2026-04-10: Repaired `src/etl/build_l1_rent_panel.py` by adding an explicit canonical-attribution scope hook for rollup/subtype pairs and excluding Starknet `batchSubmissions` + `proofSubmissions` from canonical aggregation under the locked T051 contract. Starknet `stateUpdates` remains in scope, so canonical Starknet `rent_paid_eth` now equals the direct-exclusive `state_updates_eth` family.
- 2026-04-10: Rebuilt the authoritative `2026-04-09` raw/processed surfaces with `python src/etl/build_l1_rent_panel.py --run-date 2026-04-09 --resume-manifested-run`. This worktree initially lacked `data/raw/l1_rent/2026-04-09`, so the raw snapshot was hydrated from sibling worktree `wt-T049` before replay. The replay then reused the copied post-partition checkpoint plus cached receipt/base-fee lookup DBs.
- 2026-04-10: Output outcome after rebuild:
  - Starknet aggregate canonical rent moved from `15848.109651251500773179 ETH` to `2221.294567790417570022 ETH`.
  - Starknet aggregate `batch_submissions_eth` and `proof_submissions_eth` are now both `0`, while `state_updates_eth` remains `2221.294567790417570022 ETH`.
  - The authoritative artifacts refreshed successfully at `data/processed/l1_rent/daily_rollup_rent_components.csv`, `data/processed/panels/daily_rollup_panel.csv`, `data/processed_manifest/daily_rollup_rent_components_2026-04-09.json`, `data/processed_manifest/daily_rollup_panel_2026-04-09.json`, `data/raw_manifest/l1_rent_2026-04-09.json`, and matching samples.
- 2026-04-10: Validation commands run:
  - `python src/etl/build_l1_rent_panel.py --run-date 2026-04-09 --resume-manifested-run`
  - `make gate`
  Outcome: the ETL replay passed, and `make gate` passed while this task was still `backlog`. After changing the task state to `ready_for_review`, a second `make gate` failed only on the expected `missing_run_manifest` review-bundle check for `T052`. Operator still needs to attach the durable run manifest before the next review-bundle gate under the new state.
- 2026-04-10: Residual caveats captured in `.orchestrator/handoff/H055_t052_starknet_direct_exclusive_rebuild_2026-04-10.md`. Besides the intended Starknet change, replay-side raw cache normalization shifted one `scroll` row by `+0.002711875740893184 ETH` and one `linea` row by `+1.31072e-13 ETH`; neither change alters the locked Starknet contract, but they should remain visible in review.
- 2026-04-10: Judge approved; review log: reports/status/reviews/T052_20260410T194240Z.json
