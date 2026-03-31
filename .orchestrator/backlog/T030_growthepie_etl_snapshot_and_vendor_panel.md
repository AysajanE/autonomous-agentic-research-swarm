---
task_id: T030
title: "growthepie ETL: raw snapshots, vendor panel, and deterministic sample"
workstream: W1
task_kind: etl
allow_network: true
role: Worker
priority: high
dependencies:
  - "T020"
  - "T025"
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
  - "src/analysis/"
  - "src/validation/"
outputs:
  - "src/etl/growthepie_fetch.py"
  - "data/raw/growthepie/YYYY-MM-DD/..."
  - "data/raw_manifest/growthepie_YYYY-MM-DD.json"
  - "data/processed/growthepie/vendor_daily_rollup_panel.csv"
  - "data/processed_manifest/vendor_daily_rollup_panel_YYYY-MM-DD.json"
  - "data/samples/growthepie/vendor_daily_rollup_panel_sample.csv"
gates:
  - "make gate"
stop_conditions:
  - "Need credentials"
  - "Source instability or breaking API changes"
  - "Registry identifiers required for normalization are missing"
---

# Task T030 — growthepie ETL: raw snapshots, vendor panel, and deterministic sample

## Context

growthepie is the primary denominator source for `l2_fees_eth` and the secondary vendor cross-check source for rent and profit series. This task owns the off-chain acquisition path only; it does not establish the authoritative rent path for release.

## Assignment

- Workstream: W1 Data: off-chain
- Assigned role: Worker
- Suggested branch/worktree name: `T030_growthepie_vendor_panel`
- Allowed paths: `src/etl/growthepie_fetch.py`, growthepie raw/processed/manifests, growthepie samples
- Stop conditions: block with `@human` instead of guessing API meaning, credentials, or rollup mapping

## Inputs

- `docs/protocol.md`
- `contracts/data_dictionary.md`
- `contracts/schemas/panel_schema_str_v1.yaml`
- `registry/rollup_registry_v1.csv`
- growthepie endpoints required for denominator and vendor cross-check series

## Outputs

- ETL code: `src/etl/growthepie_fetch.py`
- Append-only raw snapshots: `data/raw/growthepie/<YYYY-MM-DD>/...`
- Raw provenance: `data/raw_manifest/growthepie_<YYYY-MM-DD>.json`
- Normalized vendor panel: `data/processed/growthepie/vendor_daily_rollup_panel.csv`
- Processed provenance: `data/processed_manifest/vendor_daily_rollup_panel_<YYYY-MM-DD>.json`
- Tracked sample: `data/samples/growthepie/vendor_daily_rollup_panel_sample.csv`

## Success Criteria

- [ ] Raw snapshots are written to dated folders without overwriting prior pulls
- [ ] Raw manifest includes file hashes and the exact reproduction command
- [ ] `vendor_daily_rollup_panel.csv` is deterministic and uses registry-backed `rollup_id`
- [ ] The processed manifest points to the producing script, git SHA, raw manifest input, and output hashes
- [ ] The tracked sample is tiny, documented, and stable across runs
- [ ] This task does not claim vendor `rent_paid` is authoritative for release
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any downstream mapping caveat is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- `python src/etl/growthepie_fetch.py --run-date YYYY-MM-DD`
- `python scripts/make_raw_manifest.py growthepie data/raw/growthepie/YYYY-MM-DD --as-of YYYY-MM-DD -- python src/etl/growthepie_fetch.py --run-date YYYY-MM-DD`

## Status

- State: backlog
- Last updated: 2026-03-29

## Notes / Decisions

- 2026-03-29: v1 rewrite narrows T030 to the off-chain denominator and vendor-panel slice. Authoritative `rent_paid_eth` is deferred to T035.
