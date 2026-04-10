# H049 — Taiko historical surface fix outcome (2026-04-10)

## Summary

Implemented the Taiko historical-attribution repair in `wt-T049` by extending `LEGACY_TRACKED_CALLS_BY_ROLLUP["taiko"]` in `src/etl/build_l1_rent_panel.py` to cover the missing official `0x68d30f47f19c07bccef4ac7fae2dc12fca3e0dc9` `labprover` / ProverSet surface plus one tiny `0x06a9...` legacy selector window.

Added supplements:

- `0x68d30f47f19c07bccef4ac7fae2dc12fca3e0dc9 / 0x10d008bd / stateUpdates / [1716625487, 1730973167]`
- `0x68d30f47f19c07bccef4ac7fae2dc12fca3e0dc9 / 0xef16e845 / batchSubmissions / [1717845743, 1730973071]`
- `0x68d30f47f19c07bccef4ac7fae2dc12fca3e0dc9 / 0x0c8f4a10 / batchSubmissions / [1730973119, 1738367975]`
- `0x68d30f47f19c07bccef4ac7fae2dc12fca3e0dc9 / 0x440b6e18 / stateUpdates / [1730973227, 1738367759]`
- `0x06a9ab27c7e2255df1815e6cc0168d7755feb19a / 0x8778209d / stateUpdates / [1722088679, 1725839183]`

## Evidence basis

- Local BigQuery exact-window analysis showed the missing `0x68d...` selector family carried the bulk of unresolved Taiko rent in `2024-05` through `2025-01`.
- Earlier root-cause work identified the dominant missing sender as `0x000000629FBCf27A347d1AEbA658435230D74a5f`, contributing about `1307.63 ETH`, and proved its hashes were absent from the canonical tx universe while ordinary `0x06a9...` Inbox traffic from `0x66cc...` was already present.
- Taiko deployment documentation in the local `taiko-mono` clone identified `0x68d...` as the official historical proving/proposal surface.

## Reproduction

Executed:

```bash
python src/etl/build_l1_rent_panel.py --run-date 2026-04-09 --resume-manifested-run
make gate
```

Both succeeded in `/Users/aeziz-local/Research/wt-T049`.

## Materialized outputs

The rebuild rewrote:

- `data/raw/l1_rent/2026-04-09/`
- `data/raw_manifest/l1_rent_2026-04-09.json`
- `data/processed/l1_rent/daily_l1_rent_decomposition.csv`
- `data/processed/l1_rent/daily_rollup_rent_components.csv`
- `data/processed/panels/daily_rollup_panel.csv`
- `data/samples/l1_rent/daily_l1_rent_decomposition_sample.csv`
- `data/samples/l1_rent/daily_rollup_rent_components_sample.csv`
- `data/samples/panels/daily_rollup_panel_sample.csv`
- `data/processed_manifest/daily_l1_rent_decomposition_2026-04-09.json`
- `data/processed_manifest/daily_rollup_rent_components_2026-04-09.json`
- `data/processed_manifest/daily_rollup_panel_2026-04-09.json`

## Outcome

Taiko moved by exactly the missing historical mass:

- old canonical Taiko total: `2455.474337270198 ETH`
- new canonical Taiko total: `3763.389704987955 ETH`
- change: `+1307.915367717757 ETH`
- current vendor Taiko total: `4006.276598861474 ETH`
- remaining Taiko gap: `-242.886893873518 ETH`

Month-level change is concentrated entirely in `2024-05` through `2025-01`; later months were unchanged.

The refreshed matched-key global surface from the new canonical panel versus the current vendor panel is:

- `matched_rows = 12434`
- `mismatched_key_count = 0`
- canonical total = `145963.829633988236 ETH`
- vendor total = `132310.156968211988 ETH`
- aggregate delta = `+13653.672665776248 ETH` (`+10.319444083992%`)

Top rollup deltas after the fix:

- `starknet`: `+13626.815080978067 ETH`
- `taiko`: `-242.886893873514 ETH`
- `arbitrum`: `+181.700702925395 ETH`
- `zksync_era`: `+177.912799182843 ETH`

Excluding `starknet`, the refreshed aggregate gap is only `+26.857584798170 ETH` (`+0.020646%`).

## Interpretation

This repair addresses the dominant Taiko historical-attribution bug. It does not make T050 pass by itself, because the remaining release blocker is now clearly `starknet`, not Taiko. A downstream T050 rerun should therefore be treated as a Starknet benchmark-resolution check rather than another Taiko hunt.
