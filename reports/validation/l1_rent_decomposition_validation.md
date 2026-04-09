# L1 rent decomposition validation

- Status: `pass`
- Mode: `canonical`
- As of: `2026-04-08`

## Summary

- row_count: `1558`
- panel_date_count: `1558`
- total_l1_rent_eth_sum: `80068.56176278245`
- date_range: `{"max": "2026-04-07", "min": "2022-01-01"}`

## Checks

### l1_decomposition_schema

- Status: `pass`
- Details: `{"missing_columns": [], "present_columns": ["date_utc", "l1_base_fee_burn_eth", "l1_blob_fee_burn_eth", "l1_priority_fee_eth", "l1_total_rent_eth", "l1_blob_gas_used", "l1_calldata_gas_used", "l1_blob_base_fee_gwei"], "required_columns": ["date_utc", "l1_base_fee_burn_eth", "l1_blob_fee_burn_eth", "l1_priority_fee_eth", "l1_total_rent_eth"]}`

### l1_decomposition_primary_key_uniqueness

- Status: `pass`
- Details: `{"duplicate_row_count": 0, "key_columns": ["date_utc"], "row_count": 1558, "sample_duplicates": []}`

### l1_decomposition_required_non_null

- Status: `pass`
- Details: `{"null_counts": {"date_utc": 0, "l1_base_fee_burn_eth": 0, "l1_blob_fee_burn_eth": 0, "l1_priority_fee_eth": 0, "l1_total_rent_eth": 0}, "violating_columns": {}}`

### l1_total_rent_identity

- Status: `pass`
- Details: `{"max_abs_difference_eth": "0", "row_count": 1558, "violating_rows": []}`

### decomposition_covers_panel_dates

- Status: `pass`
- Details: `{"decomposition_date_count": 1558, "missing_panel_dates": [], "panel_date_count": 1558}`

## Provenance

- vendor_panel: `{"as_of_utc_date": "2026-04-08", "dataset": "vendor_panel", "manifest_path": "data/processed_manifest/vendor_daily_rollup_panel_2026-04-08.json", "path": "data/processed/growthepie/vendor_daily_rollup_panel.csv"}`
- l1_decomposition: `{"as_of_utc_date": "2026-04-08", "dataset": "l1_decomposition", "manifest_path": "data/processed_manifest/daily_l1_rent_decomposition_2026-04-08.json", "path": "data/processed/l1_rent/daily_l1_rent_decomposition.csv"}`
- authoritative_panel: `{"as_of_utc_date": "2026-04-08", "dataset": "authoritative_panel", "manifest_path": "data/processed_manifest/daily_rollup_panel_2026-04-08.json", "path": "data/processed/panels/daily_rollup_panel.csv"}`
- command_hints: `{"canonical": "python src/validation/validate_str_pipeline.py --as-of YYYY-MM-DD", "sample": "python src/validation/validate_str_pipeline.py --sample"}`
