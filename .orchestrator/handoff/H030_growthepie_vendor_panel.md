# H030 — growthepie Vendor Panel ETL Outputs

Date: 2026-04-01
Task: T030

## Summary

- Implemented `src/etl/growthepie_fetch.py`.
- Raw snapshot written to `data/raw/growthepie/2026-04-01/`.
- Raw manifest written to `data/raw_manifest/growthepie_2026-04-01.json`.
- Processed panel written to `data/processed/growthepie/vendor_daily_rollup_panel.csv`.
- Processed manifest written to `data/processed_manifest/vendor_daily_rollup_panel_2026-04-01.json`.
- Tracked sample written to `data/samples/growthepie/vendor_daily_rollup_panel_sample.csv`.

## Artifact details

- Snapshot contents: `master.json`, `fetch_manifest.json`, and 56 chain metric payloads at `metrics/chains/<rollup>/{fees,rent_paid,profit,txcount}.json`.
- Raw manifest file count: 58.
- Panel row count: 12,322.
- Rollup universe: `arbitrum`, `base`, `ink`, `linea`, `lisk`, `mode`, `optimism`, `scroll`, `soneium`, `starknet`, `taiko`, `unichain`, `worldchain`, `zksync_era`.
- Date coverage after protocol and registry filters: `2022-01-01` through `2026-03-31`.

## Deterministic sample

- The tracked sample is fixed to:
  - rollups: `arbitrum`, `base`, `optimism`
  - dates: `2024-03-13`, `2024-03-14`, `2024-03-15`
- The ETL fails if any of those nine rows disappear, so the sample selection does not drift silently.

## Reproduction

- `python src/etl/growthepie_fetch.py --run-date 2026-04-01`
- `python scripts/make_raw_manifest.py growthepie data/raw/growthepie/2026-04-01 --as-of 2026-04-01 -- python src/etl/growthepie_fetch.py --run-date 2026-04-01`
- `make gate`

## Gate result

- `make gate` passed on 2026-04-01.

## Operator follow-up

- This run was executed outside the local swarm runtime path that writes `reports/status/swarm_runs/T030_*.json`.
- Before review, Operator should record a durable run manifest with the command list above and the changed paths:
  - `.orchestrator/backlog/T030_growthepie_etl_snapshot_and_vendor_panel.md`
  - `.orchestrator/handoff/H030_growthepie_vendor_panel.md`
  - `src/etl/growthepie_fetch.py`
  - `data/raw_manifest/growthepie_2026-04-01.json`
  - `data/processed_manifest/vendor_daily_rollup_panel_2026-04-01.json`
  - `data/samples/growthepie/vendor_daily_rollup_panel_sample.csv`

## Caveats

- The ETL mirrors the live per-chain `growthepie` endpoints at `https://api.growthepie.com/v1/metrics/chains/<rollup>/<metric>.json`. Aggregate metric endpoints such as `https://api.growthepie.com/v1/metrics/fees.json` returned `403` during implementation, so the script intentionally uses the per-chain paths referenced by the site frontend instead of the aggregate files.
- Authoritative release `rent_paid_eth` remains deferred to T035; this panel keeps the growthepie vendor series as a secondary cross-check surface.
