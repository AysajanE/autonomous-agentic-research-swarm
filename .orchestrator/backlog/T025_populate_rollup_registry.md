---
task_id: T025
title: "Populate the evidence-backed rollup registry for the v1 release universe"
workstream: W3
task_kind: registry
allow_network: true
role: Worker
priority: high
dependencies:
  - "T020"
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
  - "Need to reinterpret rollup inclusion criteria"
  - "Evidence is unavailable or contradictory"
---

# Task T025 — Populate the evidence-backed rollup registry for the v1 release universe

## Context

`registry/rollup_registry_v1.csv` is still a header stub. The empirical release cannot be reviewed until the in-scope rollup universe is explicit, evidenced, and time-bounded.

## Assignment

- Workstream: W3 Registry
- Assigned role: Worker
- Suggested branch/worktree name: `T025_rollup_registry`
- Allowed paths: `registry/rollup_registry_v1.csv`, `registry/CHANGELOG.md`
- Stop conditions: block with `@human` instead of guessing missing evidence, status, or active dates

## Inputs

- `docs/protocol.md`
- `contracts/project.yaml`
- `registry/README.md`
- `registry/rollup_registry_v1.csv`

## Outputs

- Non-stub `registry/rollup_registry_v1.csv`
- `registry/CHANGELOG.md` entry describing the populated universe and evidence standard

## Success Criteria

- [ ] Every non-header row has `rollup_id`, `display_name`, `type`, `da_posting_method`, `evidence_url`, `verified_utc`, `status`, `start_date_utc`, `end_date_utc`, and `notes`; `batcher_addresses_json` is populated when known and explicitly explained when blank
- [ ] The registry covers the current in-scope rollups needed by T030 and T035
- [ ] No invented rows, dates, or evidence links are added
- [ ] `registry/CHANGELOG.md` records the scope and evidence standard used
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any unresolved registry caveat is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`

## Status
- State: ready_for_review
- Last updated: 2026-03-31
## Notes / Decisions

- 2026-03-29: New v1 task added. Downstream ecosystem-level STR outputs are blocked until the registry is evidence-backed.
- 2026-03-31: Claimed by local swarm runtime on branch T025_populate_rollup_registry.
- 2026-03-31: Populated `registry/rollup_registry_v1.csv` with 14 active rows (`arbitrum`, `starknet`, `optimism`, `zksync_era`, `linea`, `base`, `scroll`, `mode`, `taiko`, `worldchain`, `lisk`, `ink`, `soneium`, `unichain`) and added a matching `registry/CHANGELOG.md` entry. Wrote downstream scope caveats to `.orchestrator/handoff/H025_rollup_registry_scope.md`.
- 2026-03-31: Inclusion filter used for this v1 registry slice: growthepie `master.json` had to mark the chain `deployment == PROD`, expose both `fees` and `rent_paid`, classify the technology as a rollup, and show Ethereum DA; rows were human-reviewed against a specific L2BEAT project page. Held out `loopring` (no `fees` metric), `ronin` (`deployment=DEV` + contradictory evidence surface), `zircuit` (noncanonical `deployment=ZIRCUIT`), and non-Ethereum-DA chains.
- 2026-03-31: `batcher_addresses_json` uses L2BEAT permission-role accounts where a current Sequencer / Operator / Validator account list is exposed. `scroll` remains `[]` because the L2BEAT page exposes submission contracts but not a distinct current batcher/operator account list.
- 2026-03-31: Reproduction commands: `curl -s https://api.growthepie.com/v1/master.json`; `curl -s https://l2beat.com/scaling/projects/<slug>` and parse `window.__SSR_DATA__`; `make gate`.
- 2026-03-31: Gates run: `make gate` passed.
- 2026-03-31: Files changed: `registry/rollup_registry_v1.csv`, `registry/CHANGELOG.md`, `.orchestrator/handoff/H025_rollup_registry_scope.md`.
- 2026-03-31: Remaining limitation: task outputs and gates are complete, but no new durable run manifest was written under `reports/status/swarm_runs/` in this worker session, so the task is left `active` rather than `ready_for_review`.
- 2026-03-31: Runtime passed: outputs, gates, manifests, and run manifest are present. Ready for Judge review. Run manifest: reports/status/swarm_runs/T025_20260331T230838Z.json
