# Cross-source reconciliation

- Status: `fail`
- Mode: `canonical`
- As of: `2026-04-01`

## Summary

- matched_row_count: `11164`
- vendor_total_rent_eth: `87176.30702620494`
- authoritative_total_rent_eth: `78744.6648188145`
- aggregate_pct_difference: `0.10707572667673433`

## Checks

### vendor_panel_schema

- Status: `pass`
- Details: `{"missing_columns": [], "present_columns": ["date_utc", "rollup_id", "l2_fees_eth", "rent_paid_eth", "profit_eth", "txcount"], "required_columns": ["date_utc", "rollup_id", "l2_fees_eth", "rent_paid_eth"]}`

### vendor_panel_primary_key_uniqueness

- Status: `pass`
- Details: `{"duplicate_row_count": 0, "key_columns": ["date_utc", "rollup_id"], "row_count": 12322, "sample_duplicates": []}`

### vendor_panel_required_non_null

- Status: `pass`
- Details: `{"null_counts": {"date_utc": 0, "l2_fees_eth": 0, "rent_paid_eth": 0, "rollup_id": 0}, "violating_columns": {}}`

### vendor_profit_identity

- Status: `fail`
- Plausible causes: `["The vendor profit series drifted away from fees minus rent beyond the protocol tolerance.", "Source rounding or unit conversion changed one of the vendor series without updating the others."]`
- Next step: `Inspect the failing vendor rows and confirm whether the source changed its profit definition or units.`
- Details: `{"evaluated_row_count": 12322, "max_abs_difference_eth": 82.418787225, "max_tolerance_eth": 9.977531751207316, "sample_violations": [{"abs_difference_eth": 62.46171051244441, "date_utc": "2023-07-19", "expected_profit_eth": 62.46171051244441, "profit_eth": 0.0, "rollup_id": "linea", "tolerance_eth": 0.6246171051244441}, {"abs_difference_eth": 40.04713316501004, "date_utc": "2023-07-20", "expected_profit_eth": 40.04713316501004, "profit_eth": 0.0, "rollup_id": "linea", "tolerance_eth": 0.4004713316501004}, {"abs_difference_eth": 76.750797288, "date_utc": "2024-02-26", "expected_profit_eth": 55.60419625107001, "profit_eth": -21.146601036929994, "rollup_id": "starknet", "tolerance_eth": 0.696888440605274}, {"abs_difference_eth": 70.464799903, "date_utc": "2024-02-27", "expected_profit_eth": 56.922719561496116, "profit_eth": -13.54208034150389, "rollup_id": "starknet", "tolerance_eth": 0.6887793599143254}, {"abs_difference_eth": 82.418787225, "date_utc": "2024-02-28", "expected_profit_eth": 68.6565642435863, "profit_eth": -13.7622229814137, "rollup_id": "starknet", "tolerance_eth": 0.8003919481866079}, {"abs_difference_eth": 56.63123705699999, "date_utc": "2024-02-29", "expected_profit_eth": 62.294577662777456, "profit_eth": 5.663340605777464, "rollup_id": "starknet", "tolerance_eth": 0.7027624931802722}, {"abs_difference_eth": 42.551452579, "date_utc": "2024-03-01", "expected_profit_eth": 39.942578577846405, "profit_eth": -2.6088740011535947, "rollup_id": "starknet", "tolerance_eth": 0.4629394505423429}, {"abs_difference_eth": 37.306796969, "date_utc": "2024-03-02", "expected_profit_eth": 39.19109500553176, "profit_eth": 1.88429803653176, "rollup_id": "starknet", "tolerance_eth": 0.4415128491743265}, {"abs_difference_eth": 45.916063152, "date_utc": "2024-03-03", "expected_profit_eth": 49.07580349881881, "profit_eth": 3.1597403468188077, "rollup_id": "starknet", "tolerance_eth": 0.5436033473526558}, {"abs_difference_eth": 35.591426092, "date_utc": "2024-03-04", "expected_profit_eth": 41.66461377331285, "profit_eth": 6.073187681312852, "rollup_id": "starknet", "tolerance_eth": 0.46412694037694674}], "violation_count": 543}`

### monthly_cross_source_reconciliation

- Status: `fail`
- Plausible causes: `["The vendor and authoritative panels do not cover the same rollup-day keys.", "One input was built from a different sample window or rollup registry snapshot."]`
- Next step: `Resolve the key mismatch before interpreting cross-source rent deltas.`
- Details: `{"mismatched_key_count": 1158, "sample_key_mismatches": [{"_merge": "left_only", "date_utc": "2022-01-01", "rollup_id": "arbitrum"}, {"_merge": "left_only", "date_utc": "2022-01-02", "rollup_id": "arbitrum"}, {"_merge": "left_only", "date_utc": "2022-01-03", "rollup_id": "arbitrum"}, {"_merge": "left_only", "date_utc": "2022-01-04", "rollup_id": "arbitrum"}, {"_merge": "left_only", "date_utc": "2022-01-05", "rollup_id": "arbitrum"}, {"_merge": "left_only", "date_utc": "2022-01-06", "rollup_id": "arbitrum"}, {"_merge": "left_only", "date_utc": "2022-01-07", "rollup_id": "arbitrum"}, {"_merge": "left_only", "date_utc": "2022-01-08", "rollup_id": "arbitrum"}, {"_merge": "left_only", "date_utc": "2022-01-09", "rollup_id": "arbitrum"}, {"_merge": "left_only", "date_utc": "2022-01-10", "rollup_id": "arbitrum"}]}`

## Provenance

- vendor_panel: `{"as_of_utc_date": "2026-04-01", "dataset": "vendor_panel", "manifest_path": "data/processed_manifest/vendor_daily_rollup_panel_2026-04-01.json", "path": "data/processed/growthepie/vendor_daily_rollup_panel.csv"}`
- l1_decomposition: `{"as_of_utc_date": "2026-04-01", "dataset": "l1_decomposition", "manifest_path": "data/processed_manifest/daily_l1_rent_decomposition_2026-04-01.json", "path": "data/processed/l1_rent/daily_l1_rent_decomposition.csv"}`
- authoritative_panel: `{"as_of_utc_date": "2026-04-01", "dataset": "authoritative_panel", "manifest_path": "data/processed_manifest/daily_rollup_panel_2026-04-01.json", "path": "data/processed/panels/daily_rollup_panel.csv"}`
- command_hints: `{"canonical": "python src/validation/validate_str_pipeline.py --as-of YYYY-MM-DD", "sample": "python src/validation/validate_str_pipeline.py --sample"}`
