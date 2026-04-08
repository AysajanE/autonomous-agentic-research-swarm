---
task_id: T047
title: "Make vendor profit series contract-compatible for STR validation"
workstream: W1
task_kind: etl
allow_network: true
role: Worker
priority: high
dependencies:
  - "T030"
  - "T050"
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "src/etl/growthepie_fetch.py"
  - "data/raw/growthepie/"
  - "data/raw_manifest/growthepie_"
  - "data/processed/growthepie/"
  - "data/processed_manifest/vendor_daily_rollup_panel_"
  - "data/samples/growthepie/"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "registry/"
  - "src/validation/"
outputs:
  - "src/etl/growthepie_fetch.py"
  - "data/raw/growthepie/<YYYY-MM-DD>/..."
  - "data/raw_manifest/growthepie_<YYYY-MM-DD>.json"
  - "data/processed/growthepie/vendor_daily_rollup_panel.csv"
  - "data/processed_manifest/vendor_daily_rollup_panel_<YYYY-MM-DD>.json"
  - "data/samples/growthepie/vendor_daily_rollup_panel_sample.csv"
gates:
  - "make gate"
stop_conditions:
  - "Need protocol or contract changes"
  - "Source semantics are contradictory or unsupported by evidence"
  - "Source instability or breaking API changes"
---

# Task T047 — Make vendor profit series contract-compatible for STR validation

## Context

T050 confirmed that the current vendor `profit_eth` series is materially inconsistent with `fees − rent_paid` on live growthepie data, especially for `starknet`, `linea`, and `zksync_era`. The raw source can remain append-only, but the normalized vendor panel must not claim a contract-compatible `profit_eth` field when the upstream series is incoherent.

## Assignment

- Workstream: W1 Data: off-chain
- Assigned role: Worker
- Suggested branch/worktree name: `T047_vendor_profit_compat`
- Allowed paths: `src/etl/growthepie_fetch.py` and the growthepie raw/processed/manifest/sample surfaces
- Stop conditions: block with `@human` instead of redefining protocol semantics or inventing a replacement vendor metric

## Inputs

- `docs/protocol.md`
- `contracts/project.yaml`
- `src/etl/growthepie_fetch.py`
- `.orchestrator/handoff/H050_canonical_validation_reconciliation_blocker.md`

## Outputs

- A repaired vendor ETL path that keeps raw growthepie evidence append-only while making the normalized `profit_eth` field contract-compatible or explicitly absent
- Updated processed vendor panel, manifests, and tracked sample for the rerun date

## Success Criteria

- [ ] The raw growthepie snapshot remains append-only and unchanged in meaning
- [ ] The normalized vendor panel no longer emits materially incoherent `profit_eth` values
- [ ] Any contract-compatible handling choice is documented in the task notes/handoff instead of hidden
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any downstream caveat needed by T050 is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- `python src/etl/growthepie_fetch.py --run-date YYYY-MM-DD`

## Status
- State: active
- Last updated: 2026-04-08
## Notes / Decisions

- 2026-04-08: Added as the W1 half of the T050 unblock path. Live growthepie responses currently include rows where `profit_eth` is not contract-compatible with `fees − rent_paid`, so the normalized vendor panel needs repair or explicit omission rather than silent mirroring.
- 2026-04-08: Claimed by local swarm runtime on branch main.
