---
task_id: T045
title: "Repair registry attribution hooks for canonical-vendor reconciliation gaps"
workstream: W3
task_kind: registry
allow_network: true
role: Worker
priority: high
dependencies:
  - "T025"
  - "T050"
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "registry/rollup_registry_v1.csv"
  - "registry/CHANGELOG.md"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "src/"
  - "data/"
outputs:
  - "registry/rollup_registry_v1.csv"
  - "registry/CHANGELOG.md"
gates:
  - "make gate"
stop_conditions:
  - "Need to reinterpret rollup inclusion or attribution criteria"
  - "Evidence for historical sender hooks is unavailable or contradictory"
---

# Task T045 — Repair registry attribution hooks for canonical-vendor reconciliation gaps

## Context

T050 is blocked on a real upstream coverage mismatch. BigQuery checks over the current registry sender sets show that the canonical panel's dominant vendor-only gaps line up with stale or incomplete registry attribution hooks, especially for `arbitrum` pre-`2023-01-04`, `linea` pre-Dencun, `zksync_era` pre-Dencun, and the `taiko` late-2025 tail.

## Assignment

- Workstream: W3 Registry
- Assigned role: Worker
- Suggested branch/worktree name: `T045_registry_attribution_hooks`
- Allowed paths: `registry/rollup_registry_v1.csv`, `registry/CHANGELOG.md`
- Stop conditions: block with `@human` instead of guessing historical sender identities or time coverage

## Inputs

- `docs/protocol.md`
- `contracts/project.yaml`
- `registry/README.md`
- `registry/rollup_registry_v1.csv`
- `.orchestrator/handoff/H050_canonical_validation_reconciliation_blocker.md`
- `.orchestrator/done/T035_onchain_l1_rent_etl_and_decomposition.md`

## Outputs

- Evidence-backed updates to `registry/rollup_registry_v1.csv`
- `registry/CHANGELOG.md` entry describing the attribution-hook repair and any residual caveat

## Success Criteria

- [ ] The registry sender coverage for the T050 blocker rollups is re-audited against evidence rather than inferred from the current stale rows
- [ ] `batcher_addresses_json` is updated only with evidence-backed historical sender hooks needed for canonical attribution
- [ ] No invented address, date, or attribution rule is added
- [ ] Any unresolved historical gap is explicitly documented as a registry caveat rather than hidden
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any downstream attribution caveat is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`

## Status
- State: active
- Last updated: 2026-04-08
## Notes / Decisions

- 2026-04-08: Added as the first upstream repair task for the T050 blocker. Current evidence shows the dominant vendor-only canonical gaps coincide with stale registry sender hooks, not a validator bug.
- 2026-04-08: Claimed by local swarm runtime on branch main.
