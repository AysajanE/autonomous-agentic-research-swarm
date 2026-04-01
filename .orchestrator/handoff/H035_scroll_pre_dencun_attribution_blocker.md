# H035 — Scroll Pre-Dencun Attribution Blocker

Date: 2026-04-01
Task: T035

## Summary

- `src/etl/build_l1_rent_panel.py` was updated in two ways:
  - Blockscout resume is now append-only across cached page-size changes, so an older 250-row cache can resume into new 1000-row requests without skipping rows.
  - The ETL now fails fast when a rollup is active before Dencun but the registry lacks the batcher attribution needed to compute authoritative pre-Dencun calldata rent.
- The task remains blocked because `scroll` is active from `2023-10-17` through `2024-03-12` before Dencun, but `registry/rollup_registry_v1.csv` has `batcher_addresses_json=[]`.

## Evidence

- Registry row:
  - `rollup_id=scroll`
  - `start_date_utc=2023-10-17`
  - `da_posting_method=calldata_then_blobs`
  - `batcher_addresses_json=[]`
  - note says L2BEAT exposes submission contracts but not a distinct batcher/operator account list
- Vendor panel coverage:
  - `data/processed/growthepie/vendor_daily_rollup_panel.csv` contains 148 `scroll` rows before `2024-03-13`
- Current ETL failure:
  - `python src/etl/build_l1_rent_panel.py --run-date 2026-04-01`
  - output: `required registry attribution inputs are missing for pre-Dencun rollups: scroll[active_pre_dencun=2023-10-17..2024-03-12, growthepie_pre_dencun_rows=148]`

## Impact

- Without additional registry evidence, the authoritative canonical panel would silently omit 148 growthepie-covered `scroll` rollup-days in the pre-Dencun interval.
- That violates the task stop condition for missing registry attribution inputs, so no `daily_l1_rent_decomposition.csv`, `daily_rollup_panel.csv`, or T035 manifests/samples were produced.

## Reproduction

- `python -m py_compile src/etl/build_l1_rent_panel.py`
- `python src/etl/build_l1_rent_panel.py --run-date 2026-04-01`

## Open Question

- Should Scroll's pre-Dencun rent be attributed via additional registry-backed evidence such as submission contract hooks, or should Scroll be explicitly excluded from the canonical panel for `2023-10-17` through `2024-03-12`?
