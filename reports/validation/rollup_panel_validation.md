# Canonical rollup panel validation

- Status: `fail`
- Mode: `canonical`
- As of: `2026-04-01`

## Summary

- row_count: `11164`
- date_count: `1551`
- rollup_count: `14`
- vendor_key_count: `12322`
- ecosystem_str_preview: `[{"date_utc": "2022-01-01", "l2_fees_eth": 16.619587052666716, "rent_paid_eth": 4.849817543, "str": 0.2918133602014989}, {"date_utc": "2022-01-02", "l2_fees_eth": 18.36587431739781, "rent_paid_eth": 4.316014659, "str": 0.2350018618450134}, {"date_utc": "2022-01-03", "l2_fees_eth": 28.251544508495524, "rent_paid_eth": 6.777715538, "str": 0.23990601773867168}, {"date_utc": "2022-01-04", "l2_fees_eth": 35.6697178902309, "rent_paid_eth": 11.67455726, "str": 0.3272960356997213}, {"date_utc": "2022-01-05", "l2_fees_eth": 54.23458436198227, "rent_paid_eth": 9.300801068, "str": 0.1714920687125195}]`

## Checks

### authoritative_panel_schema

- Status: `pass`
- Details: `{"missing_columns": [], "present_columns": ["date_utc", "rollup_id", "l2_fees_eth", "rent_paid_eth", "profit_eth", "txcount"], "required_columns": ["date_utc", "rollup_id", "l2_fees_eth", "rent_paid_eth"]}`

### authoritative_panel_primary_key_uniqueness

- Status: `pass`
- Details: `{"duplicate_row_count": 0, "key_columns": ["date_utc", "rollup_id"], "row_count": 11164, "sample_duplicates": []}`

### authoritative_panel_required_non_null

- Status: `pass`
- Details: `{"null_counts": {"date_utc": 0, "l2_fees_eth": 0, "rent_paid_eth": 0, "rollup_id": 0}, "violating_columns": {}}`

### authoritative_vs_vendor_key_coverage

- Status: `fail`
- Plausible causes: `["The authoritative panel and vendor panel were built from different rollup universes or sample windows.", "One pipeline emitted rows with missing paired metrics while the other followed the row-omission rule."]`
- Next step: `Reconcile the key-level coverage mismatch before trusting STR comparisons or downstream figures.`
- Details: `{"authoritative_panel_key_count": 11164, "only_in_authoritative_panel": [], "only_in_vendor_panel": [{"date_utc": "2022-01-01", "rollup_id": "arbitrum"}, {"date_utc": "2022-01-02", "rollup_id": "arbitrum"}, {"date_utc": "2022-01-03", "rollup_id": "arbitrum"}, {"date_utc": "2022-01-04", "rollup_id": "arbitrum"}, {"date_utc": "2022-01-05", "rollup_id": "arbitrum"}, {"date_utc": "2022-01-06", "rollup_id": "arbitrum"}, {"date_utc": "2022-01-07", "rollup_id": "arbitrum"}, {"date_utc": "2022-01-08", "rollup_id": "arbitrum"}, {"date_utc": "2022-01-09", "rollup_id": "arbitrum"}, {"date_utc": "2022-01-10", "rollup_id": "arbitrum"}], "vendor_panel_key_count": 12322}`

### metrics_module_compatibility

- Status: `pass`
- Details: `{"authoritative_panel_row_count": 11164, "distinct_date_count": 1551, "ecosystem_str_preview": [{"date_utc": "2022-01-01", "l2_fees_eth": 16.619587052666716, "rent_paid_eth": 4.849817543, "str": 0.2918133602014989}, {"date_utc": "2022-01-02", "l2_fees_eth": 18.36587431739781, "rent_paid_eth": 4.316014659, "str": 0.2350018618450134}, {"date_utc": "2022-01-03", "l2_fees_eth": 28.251544508495524, "rent_paid_eth": 6.777715538, "str": 0.23990601773867168}, {"date_utc": "2022-01-04", "l2_fees_eth": 35.6697178902309, "rent_paid_eth": 11.67455726, "str": 0.3272960356997213}, {"date_utc": "2022-01-05", "l2_fees_eth": 54.23458436198227, "rent_paid_eth": 9.300801068, "str": 0.1714920687125195}], "ecosystem_str_row_count": 1551, "rollup_str_row_count": 11164}`

## Provenance

- vendor_panel: `{"as_of_utc_date": "2026-04-01", "dataset": "vendor_panel", "manifest_path": "data/processed_manifest/vendor_daily_rollup_panel_2026-04-01.json", "path": "data/processed/growthepie/vendor_daily_rollup_panel.csv"}`
- l1_decomposition: `{"as_of_utc_date": "2026-04-01", "dataset": "l1_decomposition", "manifest_path": "data/processed_manifest/daily_l1_rent_decomposition_2026-04-01.json", "path": "data/processed/l1_rent/daily_l1_rent_decomposition.csv"}`
- authoritative_panel: `{"as_of_utc_date": "2026-04-01", "dataset": "authoritative_panel", "manifest_path": "data/processed_manifest/daily_rollup_panel_2026-04-01.json", "path": "data/processed/panels/daily_rollup_panel.csv"}`
- command_hints: `{"canonical": "python src/validation/validate_str_pipeline.py --as-of YYYY-MM-DD", "sample": "python src/validation/validate_str_pipeline.py --sample"}`
