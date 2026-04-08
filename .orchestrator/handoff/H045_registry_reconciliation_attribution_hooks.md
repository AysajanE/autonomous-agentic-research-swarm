# H045 — Registry Attribution Repair for T050

Date: 2026-04-08
Task: T045

## Summary

- `registry/rollup_registry_v1.csv` now carries evidence-backed historical sender hooks for the three rollups that accounted for the dominant stale-sender reconciliation gaps: `linea`, `zksync_era`, and `taiko`.
- `arbitrum` was re-audited but left unchanged because the available evidence still does not identify a defensible pre-`2023-01-04` historical sender. That unresolved interval is now explicit in the registry notes instead of being hidden.

## What changed / what exists now

- Files/paths:
  - `registry/rollup_registry_v1.csv`
  - `registry/CHANGELOG.md`
- Outputs produced:
  - `linea` now includes historical sender `0xa9268341831efa4937537bc3e9eb36dbece83c7e`; notes cite Blockscout first-seen `2024-02-13` to tracked contract `0xd19d4b5d358258f05d7b411e21a1460d11b0876f` with selector `0x7a776315`, and retain an explicit caveat for coverage before `2024-02-13`.
  - `zksync_era` now includes supplemental historical senders `0x0d3250c3d5facb74ac15834096397a3ef790ec99` and `0xe1d8d4c8656949764c2c9fa9fab2c15d3f42e6c2`; notes cite the tracked contract link for `0x0d325...` and live Blobscan rollup-filter windows showing the historical sender rotation beyond the current L2BEAT validator set.
  - `taiko` now includes launch-era and late-tail blob senders `0x000000633b68f5d8d3a86593ebb815b4663bcbe0`, `0x41f2f55571f9e8e3ba511adc48879bd67626a2b6`, `0x7a853a6480f4d7db79ae91c16c960dbbb6710d25`, `0x5f62d006c10c009ff50c878cd6157ac861c99990`, and `0xe2da8ac2e550cd141198a117520d4edc8692ab74`; notes cite the live Blobscan windows that exposed each rotation.
  - `arbitrum` notes now state that the currently listed hooks are not evidenced before `2023-01-04`, so any canonical-vendor comparison that still shows an earlier Arbitrum gap should treat it as a known registry limitation, not a validator regression.

## How to reproduce / verify

- Commands:
  - `python - <<'PY'` against `data/raw_manifest/l1_rent_2026-04-01.json` to enumerate preserved sender trees under `blockscout/txlist/<rollup>/<address>/...`
  - `python - <<'PY'` importing `src.etl.build_l1_rent_panel.extract_l2beat_tracked_transactions` to pull the tracked posting contract + selector metadata from the L2BEAT project pages
  - `curl -s 'https://eth.blockscout.com/api?...' | jq ...` to get first-seen outgoing tx timestamps for candidate senders
  - `curl -s 'https://api.blobscan.com/transactions?...' | jq ...` to sample sender sets from rollup-filter windows
  - `make gate`
- Expected results:
  - `registry/rollup_registry_v1.csv` parses cleanly and includes the added sender hooks for `linea`, `zksync_era`, and `taiko`
  - `make gate` passes

## Assumptions / risks

- Local `bq` access was unavailable in this session because the CLI had no authenticated account; this repair relies on preserved raw manifests plus live Blockscout/Blobscan evidence rather than fresh BigQuery validation.
- Live Linea rollup-filter history was contradictory on rerun relative to the preserved 2026-04-06 evidence, so the registry intentionally stops at the defensible `0xa926...` fix and keeps pre-`2024-02-13` history caveated.
- Arbitrum still lacks an evidence-backed sender before `2023-01-04`. Downstream reconciliation should continue to treat any earlier Arbitrum canonical gap as unresolved registry coverage, not a silent omission.

## Open questions / next steps

- Operator should record a durable run manifest for this worker pass before moving T045 to review.
- T050 should rerun its canonical-vendor reconciliation against the updated registry and confirm whether the Linea, ZKsync Era, and Taiko vendor-only gaps close as expected, with only the explicit Arbitrum/early-Linea caveats remaining.
