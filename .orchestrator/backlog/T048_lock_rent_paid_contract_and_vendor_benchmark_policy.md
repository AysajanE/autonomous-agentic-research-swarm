---
task_id: T048
title: "Lock canonical rent contract and vendor benchmark policy"
workstream: W0
task_kind: protocol
allow_network: false
role: Worker
priority: high
dependencies:
  - "T000"
  - "T020"
  - "T035"
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
  - "Definition ambiguity"
  - "Need to redefine vendor methodology without evidence-backed rationale"
  - "Need edits outside W0-owned paths"
---

# Task T048 — Lock canonical rent contract and vendor benchmark policy

## Context

`T035` and the repaired `T046` rebuild proved that the canonical on-chain panel can now reproduce the vendor key universe for `2026-04-09`, but `T050` still fails on matched keys. The current blocker is not generic ETL incompleteness anymore. It is a contract conflict between:

- the repo protocol claim that `rent_paid_eth` is the authoritative on-chain amount paid to Ethereum L1 for settlement/DA/proofs, and
- the release-time expectation that the canonical series should reconcile tightly to the growthepie vendor benchmark.

Current evidence shows those are not the same measurement object:

- the latest `T050` report has `mismatched_key_count = 0` with an `8.53%` aggregate matched-key gap
- `starknet` dominates the residual divergence, with `taiko` the second-largest contributor
- vendor methodology evidence shows `rent_paid_eth` is assembled from vendor-side cost components and does not necessarily include every settlement-like cost family that their own `profit_eth` subtracts
- the vendor economics mapping is a chain-specific curated transaction taxonomy, not a literal statement that every attributable on-chain fee belongs in vendor `rent_paid_eth`

Until this contract is locked, more `T035`/`T046`/`T050` reruns will keep rediscovering the same ambiguity under different rollups.

## Assignment

- Workstream: W0 Protocol/Contracts
- Assigned role: Worker
- Suggested branch/worktree name: `T048_rent_contract_policy`
- Allowed paths: protocol and contracts only
- Stop conditions: block with `@human` if the repo cannot decide between canonical-on-chain truth and vendor-consistent benchmark policy from the available evidence

## Inputs

- `docs/protocol.md`
- `contracts/data_dictionary.md`
- `contracts/decisions.md`
- `contracts/project.yaml`
- `.orchestrator/handoff/H048_t050_contract_resolution_blocker.md`
- `reports/validation/cross_source_reconciliation.md`

## Outputs

- Update `docs/protocol.md` to explicitly lock:
  - what `rent_paid_eth` means for release purposes
  - whether vendor `rent_paid` is a diagnostic benchmark or a release-truth proxy
  - what classes of benchmark divergence are expected vs release-blocking
- Update `contracts/data_dictionary.md` so the daily rollup panel definition no longer conflates canonical rent with vendor benchmark semantics
- Add a `contracts/decisions.md` entry capturing the chosen benchmark policy, rationale, and expected downstream impact
- Update `contracts/project.yaml` if the scientific contract or release blockers need to reflect the new benchmark policy explicitly

## Success Criteria

- [ ] The protocol makes an explicit, reviewable choice between canonical on-chain truth and vendor-consistent benchmark semantics
- [ ] The role of growthepie `rent_paid` in validation is locked unambiguously as either release truth or secondary benchmark
- [ ] The contracts explicitly state how matched-key benchmark divergences are treated in release gating
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any unresolved benchmark-definition caveat is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`

## Status

- State: active
- Last updated: 2026-04-10

## Notes / Decisions

- 2026-04-10: Created after post-repair `T050` evidence showed the remaining failure is now a contract-level benchmark conflict, not a generic canonical coverage defect.
- 2026-04-10: Claimed on branch `T048_rent_contract_policy` to lock the benchmark policy explicitly before any further `T050` rerun or ETL replay.
- 2026-04-10: Updated `docs/protocol.md`, `contracts/data_dictionary.md`, `contracts/decisions.md`, `contracts/project.yaml`, and `contracts/CHANGELOG.md` to lock canonical on-chain `rent_paid_eth` as release truth, define growthepie `rent_paid` as a secondary benchmark, and add the `daily_rollup_rent_components` contract/project requirements needed for downstream auditability.
- 2026-04-10: Reproduction/gate command: `make gate` from repo root. Outcome: pass.
- 2026-04-10: Downstream note: `T049` now has a contract-backed target artifact and `T050` should stay blocked until the component surface exists and validation logic is updated to apply the locked benchmark policy rather than forcing raw vendor parity.
