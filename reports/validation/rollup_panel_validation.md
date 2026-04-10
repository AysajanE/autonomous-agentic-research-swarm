# Canonical rollup panel validation

- Status: `pass`
- Mode: `canonical`
- As of: `2026-04-09`

## Summary

- row_count: `12434`
- date_count: `1559`
- rollup_count: `14`
- rent_component_row_count: `12434`
- ecosystem_str_preview: `[{"date_utc": "2022-01-01", "l2_fees_eth": 45.44057692243189, "rent_paid_eth": 25.375401129441972, "str": 0.5584304348238881}, {"date_utc": "2022-01-02", "l2_fees_eth": 48.524693244568226, "rent_paid_eth": 29.318656831421166, "str": 0.6042007660646684}, {"date_utc": "2022-01-03", "l2_fees_eth": 61.89649704773498, "rent_paid_eth": 38.14750140184161, "str": 0.616311152025647}, {"date_utc": "2022-01-04", "l2_fees_eth": 86.24451779735746, "rent_paid_eth": 50.40996680963354, "str": 0.5845005351885463}, {"date_utc": "2022-01-05", "l2_fees_eth": 108.20281729092335, "rent_paid_eth": 66.5912745406949, "str": 0.6154301358129327}]`

## Checks

### authoritative_panel_schema

- Status: `pass`
- Details: `{"missing_columns": [], "present_columns": ["date_utc", "rollup_id", "l2_fees_eth", "rent_paid_eth", "profit_eth", "txcount"], "required_columns": ["date_utc", "rollup_id", "l2_fees_eth", "rent_paid_eth"]}`

### authoritative_panel_primary_key_uniqueness

- Status: `pass`
- Details: `{"duplicate_row_count": 0, "key_columns": ["date_utc", "rollup_id"], "row_count": 12434, "sample_duplicates": []}`

### authoritative_panel_required_non_null

- Status: `pass`
- Details: `{"null_counts": {"date_utc": 0, "l2_fees_eth": 0, "rent_paid_eth": 0, "rollup_id": 0}, "violating_columns": {}}`

### rent_component_schema

- Status: `pass`
- Details: `{"missing_columns": [], "present_columns": ["date_utc", "rollup_id", "batch_submissions_eth", "proof_submissions_eth", "state_updates_eth", "execution_base_fee_burn_eth", "execution_priority_fee_eth", "blob_fee_burn_eth", "rent_paid_eth"], "required_columns": ["date_utc", "rollup_id", "batch_submissions_eth", "proof_submissions_eth", "state_updates_eth", "blob_fee_burn_eth", "execution_base_fee_burn_eth", "execution_priority_fee_eth", "rent_paid_eth"]}`

### rent_component_primary_key_uniqueness

- Status: `pass`
- Details: `{"duplicate_row_count": 0, "key_columns": ["date_utc", "rollup_id"], "row_count": 12434, "sample_duplicates": []}`

### rent_component_required_non_null

- Status: `pass`
- Details: `{"null_counts": {"batch_submissions_eth": 0, "blob_fee_burn_eth": 0, "date_utc": 0, "execution_base_fee_burn_eth": 0, "execution_priority_fee_eth": 0, "proof_submissions_eth": 0, "rent_paid_eth": 0, "rollup_id": 0, "state_updates_eth": 0}, "violating_columns": {}}`

### rent_component_vs_panel_key_coverage

- Status: `pass`
- Details: `{"authoritative_panel_key_count": 12434, "only_in_authoritative_panel": [], "only_in_authoritative_panel_count": 0, "only_in_rent_components": [], "only_in_rent_components_count": 0, "rent_component_key_count": 12434}`

### rent_component_tx_family_identity

- Status: `pass`
- Details: `{"component_columns": ["batch_submissions_eth", "proof_submissions_eth", "state_updates_eth"], "identity_tolerance_eth": "1E-9", "max_abs_difference_eth": "0", "row_count": 12434, "target_column": "rent_paid_eth", "violating_rows": []}`

### rent_component_fee_class_identity

- Status: `pass`
- Details: `{"component_columns": ["blob_fee_burn_eth", "execution_base_fee_burn_eth", "execution_priority_fee_eth"], "identity_tolerance_eth": "1E-9", "max_abs_difference_eth": "0", "row_count": 12434, "target_column": "rent_paid_eth", "violating_rows": []}`

### rent_component_panel_identity

- Status: `pass`
- Details: `{"identity_tolerance_eth": "1E-9", "max_abs_difference_eth": "0", "row_count": 12434, "violating_rows": []}`

### metrics_module_compatibility

- Status: `pass`
- Details: `{"authoritative_panel_row_count": 12434, "distinct_date_count": 1559, "ecosystem_str_preview": [{"date_utc": "2022-01-01", "l2_fees_eth": 45.44057692243189, "rent_paid_eth": 25.375401129441972, "str": 0.5584304348238881}, {"date_utc": "2022-01-02", "l2_fees_eth": 48.524693244568226, "rent_paid_eth": 29.318656831421166, "str": 0.6042007660646684}, {"date_utc": "2022-01-03", "l2_fees_eth": 61.89649704773498, "rent_paid_eth": 38.14750140184161, "str": 0.616311152025647}, {"date_utc": "2022-01-04", "l2_fees_eth": 86.24451779735746, "rent_paid_eth": 50.40996680963354, "str": 0.5845005351885463}, {"date_utc": "2022-01-05", "l2_fees_eth": 108.20281729092335, "rent_paid_eth": 66.5912745406949, "str": 0.6154301358129327}], "ecosystem_str_row_count": 1559, "rollup_str_row_count": 12434}`

## Provenance

- vendor_panel: `{"as_of_utc_date": "2026-04-09", "dataset": "vendor_panel", "manifest_path": "data/processed_manifest/vendor_daily_rollup_panel_2026-04-09.json", "path": "data/processed/growthepie/vendor_daily_rollup_panel.csv"}`
- l1_decomposition: `{"as_of_utc_date": "2026-04-09", "dataset": "l1_decomposition", "manifest_path": "data/processed_manifest/daily_l1_rent_decomposition_2026-04-09.json", "path": "data/processed/l1_rent/daily_l1_rent_decomposition.csv"}`
- rent_components: `{"as_of_utc_date": "2026-04-09", "dataset": "rent_components", "manifest_path": "data/processed_manifest/daily_rollup_rent_components_2026-04-09.json", "path": "data/processed/l1_rent/daily_rollup_rent_components.csv"}`
- authoritative_panel: `{"as_of_utc_date": "2026-04-09", "dataset": "authoritative_panel", "manifest_path": "data/processed_manifest/daily_rollup_panel_2026-04-09.json", "path": "data/processed/panels/daily_rollup_panel.csv"}`
- command_hints: `{"canonical": "python src/validation/validate_str_pipeline.py --as-of YYYY-MM-DD", "sample": "python src/validation/validate_str_pipeline.py --sample"}`
