# Canonical rollup panel validation

- Status: `blocked`
- Mode: `canonical`
- As of: `2026-04-01`

## Summary

- message: `"Canonical validation could not run because manifest-backed inputs are incomplete."`
- requested_as_of_utc_date: `"2026-04-01"`

## Checks

### canonical_input_resolution

- Status: `blocked`
- Plausible causes: `["The processed manifests exist but their referenced CSV outputs were not materialized in this worktree.", "The ETL run that produced the manifests may have been executed outside this sandbox and only the manifests were committed."]`
- Next step: `Restore the manifest-backed processed CSVs locally or rerun the producing ETL for 2026-04-01 before re-running this validator.`
- Details: `{"missing_artifacts": [{"dataset": "vendor_panel", "expected_artifact_path": "data/processed/growthepie/vendor_daily_rollup_panel.csv", "manifest_path": "data/processed_manifest/vendor_daily_rollup_panel_2026-04-01.json", "reason": "processed artifact is absent in the worktree"}, {"dataset": "l1_decomposition", "expected_artifact_path": "data/processed/l1_rent/daily_l1_rent_decomposition.csv", "manifest_path": "data/processed_manifest/daily_l1_rent_decomposition_2026-04-01.json", "reason": "processed artifact is absent in the worktree"}, {"dataset": "authoritative_panel", "expected_artifact_path": "data/processed/panels/daily_rollup_panel.csv", "manifest_path": "data/processed_manifest/daily_rollup_panel_2026-04-01.json", "reason": "processed artifact is absent in the worktree"}], "missing_manifests": [], "next_step": "Restore the manifest-backed processed CSVs locally or rerun the producing ETL for 2026-04-01 before re-running this validator.", "requested_as_of_utc_date": "2026-04-01"}`

## Provenance

