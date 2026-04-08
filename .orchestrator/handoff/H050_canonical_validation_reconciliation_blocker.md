# H050 — Canonical Validation Reconciliation Blocker

Date: 2026-04-08
Origin task: T047

## Summary

- Repaired `src/etl/growthepie_fetch.py` so the normalized vendor panel no longer mirrors upstream `profit` blindly.
- The normalized `profit_eth` field is now emitted only when the vendor `profit` value satisfies the protocol accounting identity against the same vendor `l2_fees_eth` and `rent_paid_eth` inputs within the locked tolerance.
- Incompatible upstream vendor profit values are left blank in the normalized panel. No replacement profit series is computed.

## 2026-04-08 rerun outputs

- Local raw snapshot: `data/raw/growthepie/2026-04-08/` with 58 files.
- Raw manifest: `data/raw_manifest/growthepie_2026-04-08.json`.
- Processed vendor panel: `data/processed/growthepie/vendor_daily_rollup_panel.csv` with 12,420 rows.
- Processed manifest: `data/processed_manifest/vendor_daily_rollup_panel_2026-04-08.json`.
- Tracked sample: `data/samples/growthepie/vendor_daily_rollup_panel_sample.csv` remained present and unchanged.

## Profit handling guidance for T046 / T050

- Treat blank `profit_eth` as explicit absence caused by upstream vendor incoherence, not as a missing fetch or row omission.
- Validation should not expect `profit_eth` to be populated for every rollup-day even when `l2_fees_eth` and `rent_paid_eth` are present.
- The 2026-04-08 live rerun blanked 547 rows under the protocol tolerance:
  - `starknet`: 508
  - `zksync_era`: 29
  - `linea`: 6
  - `taiko`: 4
- `taiko` is a new live inconsistency relative to the earlier 2026-04-01 baseline, so downstream reconciliation should use the 2026-04-08 manifests rather than assume only the earlier `starknet` / `linea` / `zksync_era` scope.

## Reproduction

- `python src/etl/growthepie_fetch.py --run-date 2026-04-08`
- `python scripts/make_raw_manifest.py growthepie data/raw/growthepie/2026-04-08 --as-of 2026-04-08 -- python src/etl/growthepie_fetch.py --run-date 2026-04-08`
- `make gate`

## Gate result

- `make gate` passed on 2026-04-08.

## Operator follow-up

- This run was executed outside the local swarm runtime that writes `reports/status/swarm_runs/T047_*.json`.
- Before review, Operator should record a durable run manifest with the commands above and the changed tracked paths:
  - `.orchestrator/backlog/T047_make_vendor_profit_contract_compatible.md`
  - `.orchestrator/handoff/H050_canonical_validation_reconciliation_blocker.md`
  - `src/etl/growthepie_fetch.py`
  - `data/raw_manifest/growthepie_2026-04-08.json`
  - `data/processed_manifest/vendor_daily_rollup_panel_2026-04-08.json`
  - `data/processed/growthepie/vendor_daily_rollup_panel.csv`
