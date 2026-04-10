# H055 — Scientific integrity audit of completed L2-L1 rent frontier

Date: 2026-04-10
Role: Operator

## Scope

- Audited the current finished frontier on:
  - `wt-T052` at `T052_repair_starknet_shared_sharp_allocation`
  - `wt-T050` at `T050_validation_str_pipeline_checks_resumed`
- Re-ran local gates and validator entrypoints, inspected canonical ETL and validator code paths, spot-checked raw lineage, and performed independent surface-to-surface consistency checks.

## Reproduction commands

- `make gate` in `wt-T052`
- `make gate` in `wt-T050`
- `python src/validation/validate_str_pipeline.py --as-of 2026-04-09` in `wt-T050`
- `python src/validation/validate_str_pipeline.py --sample` in `wt-T050`
- `make test` in `wt-T050`
- ad hoc comparisons:
  - canonical panel vs canonical component identity
  - canonical panel daily totals vs canonical decomposition daily totals
  - canonical raw checkpoint rollup-day keys vs vendor panel keys
  - processed manifest hashes vs current mutable output files

## Audit result

- Do not proceed to downstream analysis or paper-writing from the current `T050` authoritative surfaces.
- The scientific frontier is blocked on a canonical-surface integrity defect plus a validator coverage gap that allowed the defect to pass review.

## Findings

### 1. Canonical panel rows are keyed off vendor coverage, not the canonical on-chain universe

- In `src/etl/build_l1_rent_panel.py`, the canonical on-chain rollup/day totals are first accumulated into `rollup_daily` and `ecosystem_daily`.
- But the emitted canonical panel and component surfaces are then materialized by iterating `vendor_rows`, not `rollup_daily`:
  - `panel_rows` / `component_rows`: `src/etl/build_l1_rent_panel.py:6361-6397`
  - `decomp_rows`: `src/etl/build_l1_rent_panel.py:6399-6421`
- This means any canonical rollup-day present in the raw tracked transaction universe but absent from the vendor panel is omitted from the authoritative panel and component surfaces while still remaining present in the decomposition total.

### 2. The omission is real and quantified

- Independent comparison of `wt-T050` produced surfaces shows:
  - `daily_rollup_panel.csv` rows: `12,434`
  - `daily_rollup_rent_components.csv` rows: `12,434`
  - `daily_l1_rent_decomposition.csv` rows: `1,559`
  - `129` days where summed panel rent differs from decomposition total
  - omitted rent total: `53.57133603213044 ETH`
  - omitted share of total decomposition rent: `0.0004046461051292864` (`0.04046%`)
- Raw checkpoint key audit against the vendor panel shows exactly `129` canonical rollup-day keys missing from vendor coverage, and every discrepant day has exactly one missing key:
  - `starknet`: `110` days, `16.275318 ETH` omitted, `2022-01-26` through `2022-05-15`
  - `base`: `17` days, `37.253258 ETH` omitted, `2023-07-13` through `2023-07-29`
  - `ink`: `2` days, `0.042760 ETH` omitted, `2024-12-07` through `2024-12-08`
- Largest omitted rollup-days observed:
  - `2022-05-11` `starknet`: `2.914860 ETH`
  - `2023-07-29` `base`: `2.904937 ETH`
  - `2023-07-24` `base`: `2.873411 ETH`

### 3. T050 validation passed without checking authoritative panel vs authoritative decomposition coherence

- `rollup_panel_validation` checks panel/component identity and metrics compatibility, but not whether summed panel rent equals decomposition total by day:
  - `src/validation/validate_str_pipeline.py:792-869`
- `l1_rent_decomposition_validation` checks only decomposition internal identity plus date coverage against the panel:
  - `src/validation/validate_str_pipeline.py:907-997`
- `cross_source_reconciliation` compares vendor vs authoritative panel only on matched vendor/panel keys:
  - `src/validation/validate_str_pipeline.py:1075-1205`
- As a result, `wt-T050` currently reports all three validation outputs as `pass` even though the authoritative panel undercounts decomposition by `53.57133603213044 ETH`.

### 4. Processed manifest validity gates are schema-only and do not prove file integrity

- `scripts/quality_gates.py` validates required manifest keys and SHA formatting only:
  - `gate_raw_manifest_validity`: `scripts/quality_gates.py:1239-1272`
  - `gate_processed_manifest_validity`: `scripts/quality_gates.py:1275-1314`
- It does not compare manifest `sha256` / `bytes` against current files.
- In `wt-T050`, a direct hash check found `10` content mismatches across `20` declared processed outputs because older as-of manifests point at mutable live output paths now holding newer rerun contents:
  - `daily_l1_rent_decomposition_2026-04-01.json`
  - `daily_l1_rent_decomposition_2026-04-08.json`
  - `daily_rollup_panel_2026-04-01.json`
  - `daily_rollup_panel_2026-04-08.json`
  - `vendor_daily_rollup_panel_2026-04-01.json`
  - `vendor_daily_rollup_panel_2026-04-08.json`

### 5. The STR metrics test fixture drifted and `make test` is not green on the finished T050 surface

- `make test` fails in `wt-T050` at:
  - `tests/test_metrics_str.py:25-45`
  - `tests/test_metrics_str.py:47-67`
- The tracked sample panel changed in `T049` but the expected STR assertions were not updated:
  - sample CSV rows: `data/samples/panels/daily_rollup_panel_sample.csv:2-10`
  - the `T049` diff changes those sample rents without changing `tests/test_metrics_str.py`
- This is a test-fixture maintenance issue, not the primary scientific blocker, but it means the finished frontier is not fully test-clean.

## Raw-lineage note

- `wt-T052` contains the April 9 raw snapshot and lookup databases needed for raw-lineage verification.
- `wt-T050` is not a self-contained raw reproduction surface; it was hydrated to run validation against processed artifacts and does not contain the full raw snapshot.
- Raw-manifest verification should therefore be performed on `wt-T052`, not on `wt-T050`.

## Required next actions before downstream analysis

1. Repair `T049` canonical panel/component emission so canonical rows are generated from the canonical on-chain rollup-day universe, not from vendor keys.
2. Add an authoritative internal coherence check to `T050` that enforces:
   - daily summed canonical panel rent equals daily decomposition total within the locked tolerance
   - canonical panel key coverage is not silently bounded by vendor coverage
3. Rebuild the canonical surfaces from the repaired ETL and rerun `T050`.
4. Refresh the stale `metrics_str` test expectations so `make test` is green again.
5. Decide whether processed manifests should point to immutable as-of outputs or whether the quality gate should start enforcing live file hash equality against manifest claims.

## Operator conclusion

- The completed frontier is not scientifically clean enough to use as the baseline for the next analysis task.
- The highest-severity issue is not a small tolerance dispute. It is a surface-selection defect in the authoritative panel build path.
