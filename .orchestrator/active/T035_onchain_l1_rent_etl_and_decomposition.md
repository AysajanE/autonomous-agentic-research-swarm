---
task_id: T035
title: "On-chain L1 rent extraction, decomposition, and canonical panel build"
workstream: W2
task_kind: etl
allow_network: true
role: Worker
priority: high
dependencies:
  - "T025"
  - "T030"
integration_ready_dependencies: []
requires_tools:
  - "python"
  - "git"
requires_env: []
allowed_paths:
  - "src/etl/build_l1_rent_panel.py"
  - "data/raw/l1_rent/"
  - "data/raw_manifest/l1_rent_"
  - "data/processed/l1_rent/"
  - "data/processed/panels/"
  - "data/processed_manifest/daily_l1_rent_decomposition_"
  - "data/processed_manifest/daily_rollup_panel_"
  - "data/samples/l1_rent/"
  - "data/samples/panels/"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
  - "registry/"
  - "src/analysis/"
  - "src/validation/"
outputs:
  - "src/etl/build_l1_rent_panel.py"
  - "data/raw/l1_rent/YYYY-MM-DD/..."
  - "data/raw_manifest/l1_rent_YYYY-MM-DD.json"
  - "data/processed/l1_rent/daily_l1_rent_decomposition.csv"
  - "data/processed/panels/daily_rollup_panel.csv"
  - "data/processed_manifest/daily_l1_rent_decomposition_YYYY-MM-DD.json"
  - "data/processed_manifest/daily_rollup_panel_YYYY-MM-DD.json"
  - "data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv"
  - "data/samples/panels/daily_rollup_panel_sample.csv"
gates:
  - "make gate"
stop_conditions:
  - "Need credentials"
  - "On-chain attribution is ambiguous"
  - "Required growthepie manifest or registry inputs are missing"
---

# Task T035 — On-chain L1 rent extraction, decomposition, and canonical panel build

## Context

This task creates the authoritative rent path for release. It turns raw on-chain L1 inputs into the daily L1 rent decomposition and the canonical `daily_rollup_panel` that combines registry-backed rollup attribution, authoritative `rent_paid_eth`, and growthepie `l2_fees_eth`.

## Assignment

- Workstream: W2 Data: on-chain
- Assigned role: Worker
- Suggested branch/worktree name: `T035_l1_rent_panel`
- Allowed paths: one on-chain ETL script plus raw/manifests/processed/sample paths for the L1 rent and canonical panel surfaces
- Stop conditions: block with `@human` instead of guessing attribution, source priority, or chain-specific rules

## Inputs

- `docs/protocol.md`
- `contracts/data_dictionary.md`
- `contracts/decisions.md`
- `contracts/schemas/panel_schema_decomp_v1.yaml`
- `contracts/schemas/panel_schema_str_v1.yaml`
- `registry/rollup_registry_v1.csv`
- `data/raw_manifest/growthepie_<YYYY-MM-DD>.json`
- `data/processed_manifest/vendor_daily_rollup_panel_<YYYY-MM-DD>.json`

## Outputs

- ETL code: `src/etl/build_l1_rent_panel.py`
- Append-only raw snapshots: `data/raw/l1_rent/<YYYY-MM-DD>/...`
- Raw provenance: `data/raw_manifest/l1_rent_<YYYY-MM-DD>.json`
- Processed decomposition: `data/processed/l1_rent/daily_l1_rent_decomposition.csv`
- Canonical panel: `data/processed/panels/daily_rollup_panel.csv`
- Processed provenance:
  - `data/processed_manifest/daily_l1_rent_decomposition_<YYYY-MM-DD>.json`
  - `data/processed_manifest/daily_rollup_panel_<YYYY-MM-DD>.json`
- Tracked samples:
  - `data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv`
  - `data/samples/panels/daily_rollup_panel_sample.csv`

## Success Criteria

- [ ] Raw on-chain pulls are append-only and have a matching raw manifest
- [ ] `daily_l1_rent_decomposition.csv` is produced deterministically from local raw inputs
- [ ] `daily_rollup_panel.csv` uses growthepie `l2_fees_eth` and authoritative on-chain `rent_paid_eth`
- [ ] The canonical panel manifest cites both the growthepie raw manifest and the on-chain raw manifest
- [ ] Tracked samples exist for the decomposition and canonical panel surfaces
- [ ] `make gate` passes

## Review Bundle Requirements

- [ ] A durable run manifest exists under `reports/status/swarm_runs/`
- [ ] Any attribution or lineage caveat needed by T050 is captured in `.orchestrator/handoff/`
- [ ] Judge review is recorded under `reports/status/reviews/`

## Validation / Commands

- `make gate`
- `python src/etl/build_l1_rent_panel.py --run-date YYYY-MM-DD`
- `python scripts/make_raw_manifest.py l1_rent data/raw/l1_rent/YYYY-MM-DD --as-of YYYY-MM-DD -- python src/etl/build_l1_rent_panel.py --run-date YYYY-MM-DD`

## Status
- State: blocked
- Last updated: 2026-04-01
## Notes / Decisions

- 2026-03-29: New v1 task added to make the on-chain rent path authoritative before metrics, validation, or release analysis.
- 2026-04-01: Operator activated isolated worktree `/Users/aeziz-local/Research/wt-T035` on branch `T035_l1_rent_panel` to execute the authoritative on-chain rent path end to end and supervise it through Judge review.
- 2026-04-01: @human Runtime blocked: executor_timeout, path_ownership_violation, missing_outputs, missing_required_manifests. Run manifest: reports/status/swarm_runs/T035_20260401T135106Z.json. ownership=.orchestrator/backlog/T025_populate_rollup_registry.md[committed]=orchestrator_write_forbidden; .orchestrator/backlog/T030_growthepie_etl_snapshot_and_vendor_panel.md[committed]=orchestrator_write_forbidden; .orchestrator/backlog/T035_onchain_l1_rent_etl_and_decomposition.md[committed]=orchestrator_write_forbidden; .orchestrator/done/T025_populate_rollup_registry.md[committed]=orchestrator_write_forbidden; .orchestrator/done/T030_growthepie_etl_snapshot_and_vendor_panel.md[committed]=orchestrator_write_forbidden; data/AGENTS.md[committed]=outside_allowed_paths; scripts/quality_gates.py[committed]=outside_allowed_paths; src/etl/AGENTS.md[committed]=outside_allowed_paths; tests/test_quality_gates_processed_manifests.py[committed]=outside_allowed_paths outputs=data/raw_manifest/l1_rent_YYYY-MM-DD.json=missing_file; data/processed/l1_rent/daily_l1_rent_decomposition.csv=missing_file; data/processed/panels/daily_rollup_panel.csv=missing_file; data/processed_manifest/daily_l1_rent_decomposition_YYYY-MM-DD.json=missing_file; data/processed_manifest/daily_rollup_panel_YYYY-MM-DD.json=missing_file; data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv=missing_file; data/samples/panels/daily_rollup_panel_sample.csv=missing_file manifests=missing_raw_manifest_file,missing_processed_manifest_file
