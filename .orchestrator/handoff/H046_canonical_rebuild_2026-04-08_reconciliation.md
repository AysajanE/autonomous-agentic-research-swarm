# H046 — Canonical L1 Rent Rebuild Reconciliation Summary

Date: 2026-04-08
Task: T046

## Summary

- Rebuilt the authoritative L1 rent surfaces for `2026-04-08` after the T045 registry attribution repair.
- The rebuild now emits fresh canonical raw and processed manifests for the repaired sender hooks, and `make gate` passes on the resulting repo state.
- The rerun materially narrows the earlier blocker, but it does not eliminate all vendor-only coverage gaps. `arbitrum` is now cleanly limited to the explicit pre-`2023-01-04` registry caveat; `linea` is mostly reduced to the expected pre-`2024-02-13` interval plus four isolated later dates; `zksync_era` still shows a meaningful late-2025 through January 2026 tail that T050 should keep flagged as unresolved.

## Outputs produced

- Raw snapshot: `data/raw/l1_rent/2026-04-08/`
- Raw manifest: `data/raw_manifest/l1_rent_2026-04-08.json`
- Processed decomposition: `data/processed/l1_rent/daily_l1_rent_decomposition.csv` with 1,558 rows
- Canonical panel: `data/processed/panels/daily_rollup_panel.csv` with 11,295 rows
- Processed manifests:
  - `data/processed_manifest/daily_l1_rent_decomposition_2026-04-08.json`
  - `data/processed_manifest/daily_rollup_panel_2026-04-08.json`
- Samples:
  - `data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv`
  - `data/samples/panels/daily_rollup_panel_sample.csv`

## Reproduction

- `CLOUDSDK_CONFIG=/tmp/codex-gcloud-auth python src/etl/build_l1_rent_panel.py --run-date 2026-04-08 --blockscout-page-size 250`
- `make gate`

## Gate result

- `make gate` passed on 2026-04-08 after the successful rerun.

## Reconciliation summary for T050

- Vendor panel rows: 12,420
- Canonical panel rows: 11,295
- Vendor-only rows after the rerun: 1,125
- Canonical-only rows after the rerun: 0

- Remaining vendor-only coverage gaps by rollup:
  - `zksync_era`: 538 rows total, including 185 post-Dencun rows from `2025-07-30` through `2026-01-30`
  - `arbitrum`: 368 rows, all confined to `2022-01-01` through `2023-01-03`
  - `linea`: 219 rows, mostly `2023-07-13` through `2024-02-12`, plus isolated later dates `2024-07-31`, `2024-09-01`, `2025-08-24`, and `2026-03-26`

- Largest overlapping canonical-minus-vendor rent deltas by total absolute ETH:
  - `optimism`: 4,657.942152044860025718 ETH absolute, net `-4519.756370991947120190`
  - `taiko`: 1,868.432417919638931340 ETH absolute, net `-1856.921790715121673334`
  - `base`: 891.167301920345007391 ETH absolute, net `-891.167301920345007391`
  - `worldchain`: 519.885192330111192848 ETH absolute, net `-519.885192330111192848`

## Notes

- The rebuild required one ETL repair in `src/etl/build_l1_rent_panel.py`: sender-scoped Blobscan zero-row windows for batcher-address rollups no longer perform their own rollup-filter fallback when the caller already retries the empty window once at rollup scope. This prevents redundant rollup-wide Blobscan replays across multiple sender hooks in the same window.
- The successful rerun happened outside the local swarm runtime. The only existing T046 run manifest in `reports/status/swarm_runs/` is the earlier blocked manifest `T046_20260408T165337Z.json`, so Operator should record a fresh durable run manifest for this successful pass before moving T046 to review.
