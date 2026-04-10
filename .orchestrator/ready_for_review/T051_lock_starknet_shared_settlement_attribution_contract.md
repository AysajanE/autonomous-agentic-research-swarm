---
task_id: T051
title: "Lock Starknet shared-settlement attribution contract"
workstream: W0
task_kind: protocol
allow_network: false
role: Worker
priority: high
dependencies:
  - "T048"
  - "T049"
allowed_paths:
  - "docs/protocol.md"
  - "contracts/data_dictionary.md"
  - "contracts/decisions.md"
  - "contracts/project.yaml"
  - "contracts/CHANGELOG.md"
disallowed_paths:
  - "src/"
  - "registry/"
  - "data/raw/"
  - "data/processed/"
outputs:
  - "docs/protocol.md"
  - "contracts/data_dictionary.md"
  - "contracts/decisions.md"
  - "contracts/project.yaml"
gates:
  - "make gate"
stop_conditions:
  - "Definition ambiguity remains between direct-exclusive and allocated shared Starknet settlement"
  - "No evidence-backed Starknet shared-settlement allocation policy can be locked from available sources"
  - "Need edits outside W0-owned paths"
---

# Task T051 — Lock Starknet shared-settlement attribution contract

## Context

`T049` and the post-repair `T050` root-cause work eliminated the old Taiko-first explanation and exposed a deeper Starknet-specific flaw.

Current evidence shows:

- growthepie Starknet `rent_paid_eth` is effectively equal to canonical `state_updates_eth`
- the entire remaining Starknet canonical excess matches canonical `batch_submissions_eth + proof_submissions_eth`
- those excess tx families live on SHARP verifier-stack contracts (`registerContinuousMemoryPage`, `verifyMerkle`, `verifyFRI`, `verifyProofAndRegister`), not the Starknet Core `updateState*` contract
- Starknet protocol sources describe SHARP proof verification as a **shared / amortized** cost rather than a cost that should be assigned to Starknet by simply summing raw Ethereum tx fees on generic SHARP contracts

That means the current canonical Starknet method is not just vendor-inconsistent. It is likely measuring the wrong Starknet-specific object.

Before any more ETL rewrites or validation reruns, the repo must explicitly lock what counts as canonical Starknet `rent_paid_eth`:

1. direct-exclusive Starknet L1 settlement / DA only
2. direct-exclusive settlement plus an allocated Starknet share of shared SHARP settlement
3. some other evidence-backed split

Without this contract, another W2 implementation would silently redefine the metric.

## Assignment

- Workstream: W0 Protocol/Contracts
- Assigned role: Worker
- Suggested branch/worktree name: `T051_starknet_shared_settlement_contract`
- Allowed paths: protocol and contracts only
- Stop conditions: block with `@human` instead of inventing a Starknet allocation policy or forcing benchmark parity without an explicit scientific contract

## Inputs

- `docs/protocol.md`
- `contracts/data_dictionary.md`
- `contracts/decisions.md`
- `contracts/project.yaml`
- `.orchestrator/handoff/H048_t050_contract_resolution_blocker.md`
- `.orchestrator/handoff/H052_starknet_shared_sharp_rootcause_2026-04-10.md`
- `.orchestrator/blocked/T050_validation_str_pipeline_checks.md`

## Outputs

- Update `docs/protocol.md` to lock Starknet-specific attribution treatment for shared SHARP settlement
- Update `contracts/data_dictionary.md` so Starknet canonical `rent_paid_eth` and the component surface are compatible with the chosen shared-settlement policy
- Add a `contracts/decisions.md` entry recording the Starknet decision, rationale, and benchmark implications
- Update `contracts/project.yaml` if the release path or required artifacts need to reflect the Starknet-specific contract explicitly

## Success Criteria

- [ ] The protocol explicitly decides how Starknet shared SHARP settlement enters canonical `rent_paid_eth`
- [ ] The decision distinguishes direct-exclusive Starknet settlement from shared SHARP settlement where needed
- [ ] The contract states how Starknet benchmark comparison to growthepie should be interpreted under the chosen policy
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any unresolved Starknet allocation caveat is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`

## Status
- State: done
- Last updated: 2026-04-10
## Notes / Decisions

- 2026-04-10: Created after the Starknet deep-dive established that the remaining `T050` blocker is a shared-SHARP attribution problem, not another missing sender/selector issue.
- 2026-04-10: The motivating root cause is recorded in `.orchestrator/handoff/H052_starknet_shared_sharp_rootcause_2026-04-10.md`.
- 2026-04-10: Locked the Starknet contract to the narrower evidence-backed choice: canonical Starknet `rent_paid_eth` is the direct-exclusive settlement / DA surface only under the current contract. Raw generic SHARP verifier-stack tx fees are excluded unless a future W0 task locks an evidence-backed historical allocation model.
- 2026-04-10: Updated `docs/protocol.md`, `contracts/data_dictionary.md`, `contracts/decisions.md`, `contracts/project.yaml`, and `contracts/CHANGELOG.md` so Starknet shared SHARP costs cannot silently re-enter canonical rent through naive raw-fee attribution.
- 2026-04-10: Benchmark implication: growthepie Starknet `rent_paid` is interpreted as a benchmark for the direct-exclusive Starknet surface, which current evidence shows aligns with the `state_updates_eth` family rather than with full raw SHARP verifier-stack fees.
- 2026-04-10: Reproduction/gate command: `make gate` from repo root. Outcome in this worktree: fail, but only on repo-wide review-bundle issues outside T051-owned paths:
  - `.orchestrator/backlog/T051_lock_starknet_shared_settlement_attribution_contract.md:missing_run_manifest`
  - `.orchestrator/ready_for_review/T049_emit_rollup_day_rent_components_for_auditability.md:missing_outputs:data/processed/l1_rent/daily_rollup_rent_components.csv=missing_file`
- 2026-04-10: Assumption: available evidence is strong enough to reject naive full-fee SHARP attribution but not strong enough to lock a reviewed historical allocation formula, so the contract excludes shared SHARP cost from canonical Starknet rent until such evidence exists.
- 2026-04-10: No new handoff note was added. The durable downstream caveat is already captured in `.orchestrator/handoff/H052_starknet_shared_sharp_rootcause_2026-04-10.md`, and the locked W0 contract files now provide the implementation target for `T052`.
- 2026-04-10: Judge approved; review log: reports/status/reviews/T051_20260410T192822Z.json
