# Contracts changelog

All interface-relevant changes to canonical contracts must be recorded here.

Format:
- `YYYY-MM-DD` — what changed, why, and expected downstream impact.

Rules:
- If a change is interface-breaking, bump the contract version (e.g., `panel_schema_v2.yaml`) and add a migration note.

- 2026-01-22 — Added a minimal non-empty `contracts/schemas/panel_schema.yaml` stub so contract gates can prevent “comment-only” schemas.
- 2026-01-23 — Added versioned STR + decomposition schemas (`panel_schema_str_v1.yaml`, `panel_schema_decomp_v1.yaml`) and updated `contracts/data_dictionary.md` to lock field names/units early.
- 2026-02-05 — Added `contracts/framework.json` (framework config contract) to make gates/supervisor generic across empirical/modeling/hybrid, and added a hybrid interface contract template (`contracts/hybrid_interface_v1.yaml`) to define the empirical→modeling boundary in hybrid projects.
- 2026-04-10 — Locked canonical `rent_paid_eth` as on-chain release truth, formalized growthepie `rent_paid` as a secondary benchmark rather than a source-priority override, and added the `daily_rollup_rent_components` contract plus matching project-contract requirements so validation can distinguish coverage defects from benchmark-definition differences.
