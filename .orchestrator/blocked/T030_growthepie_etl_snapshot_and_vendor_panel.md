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
- State: blocked
- Last updated: 2026-04-01
## Notes / Decisions

- 2026-03-29: v1 rewrite narrows T030 to the off-chain denominator and vendor-panel slice. Authoritative `rent_paid_eth` is deferred to T035.
- 2026-04-01: Claimed by local swarm runtime on branch T030_growthepie_etl_snapshot_and_vendor_panel.
- 2026-04-01: Implemented `src/etl/growthepie_fetch.py` to mirror the live per-chain `metrics/chains/<rollup>/<metric>.json` API, snapshot `master.json` plus 56 chain-metric payloads under `data/raw/growthepie/2026-04-01/`, and enforce registry-backed filtering to the 14 rollups in `registry/rollup_registry_v1.csv`.
- 2026-04-01: Produced `data/processed/growthepie/vendor_daily_rollup_panel.csv` with 12,322 rows covering `2022-01-01` through `2026-03-31`; panel rows are emitted only when both `fees` and `rent_paid` exist, with optional `profit` and `txcount` filled when present.
- 2026-04-01: Produced `data/samples/growthepie/vendor_daily_rollup_panel_sample.csv` as a fixed 9-row sample for `arbitrum`, `base`, and `optimism` on `2024-03-13` through `2024-03-15` to keep the tracked sample deterministic across reruns.
- 2026-04-01: Wrote provenance files `data/raw_manifest/growthepie_2026-04-01.json` and `data/processed_manifest/vendor_daily_rollup_panel_2026-04-01.json`. The raw manifest hashes 58 files in the dated snapshot; the processed manifest records hashes for the panel CSV and sample CSV.
- 2026-04-01: Reproduction commands run:
  - `python src/etl/growthepie_fetch.py --run-date 2026-04-01`
  - `python scripts/make_raw_manifest.py growthepie data/raw/growthepie/2026-04-01 --as-of 2026-04-01 -- python src/etl/growthepie_fetch.py --run-date 2026-04-01`
  - `make gate`
- 2026-04-01: Gate summary: `make gate` passed, including `raw_manifest_validity`, `processed_manifest_validity`, and `review_bundle_integrity`.
- 2026-04-01: Task remains `active` because this non-swarm execution cannot write the required Operator-owned durable run manifest under `reports/status/swarm_runs/`; exact commands and artifact details were handed off for Operator recording before review.
- 2026-04-01: Runtime passed: outputs, gates, manifests, and run manifest are present. Ready for Judge review. Run manifest: reports/status/swarm_runs/T030_20260401T115602Z.json
- 2026-04-01: Judge approved; review log: reports/status/reviews/T030_20260401T121619Z.json
- 2026-04-01: Operator preflight on `main` reran `make gate` and found the branch is missing required task outputs `data/raw/growthepie/2026-04-01/` and `data/processed/growthepie/vendor_daily_rollup_panel.csv`, even though the run manifest and Judge log recorded a passing review in the worker worktree. Operationally blocked pending Worker repair to restore or regenerate the missing W1 outputs on a task branch and rerun gate/review from that repaired branch. @human: if those outputs were intentionally omitted from version control, reopen or rescope T030 before release.
