# `contracts/schemas/`

Tracked schemas / data contracts used by the pipeline.

All files on this surface are executable JSON documents, including files kept
with a `.yaml` suffix for compatibility. They are parsed with `json.loads`; the
runtime has no PyYAML or dataframe-schema dependency.

Canonical schemas for this project:
- `panel_schema_str_v1.yaml` (minimum daily rollup STR panel)
- `panel_schema_decomp_v1.yaml` (daily Ethereum L1 rent decomposition)
- `rent_components_v1.json` (pack-owned STR rent-component field order and required columns)
- `instance_manifest_v1.json` (bridge-generated and synthetic variants)
- `experiment_spec_v1.json` (Lock A experiment-design surface)
- `experiment_manifest_v1.json` (solver-run registration surface)
- `referee_report_v1.json` (cross-family per-criterion/check findings surface)
- `integrity_audit_v1.json` (scratch-worktree recomputation and family-separation report)
- `literature_manifest_v1.json` (append-only W-Lit snapshot provenance and mini-PRISMA strategy)
- `raw_manifest_v1.json` and `processed_manifest_v2.json` (data lineage)
- `swarm_run_manifest_v{1,2}` and `judge_review_log_v{1,2}` (runtime/review records)
- `pack_config_v1.json` and `kernel_interface_v1.json` (kernel-pack boundary)

Manifest schemas are loaded by `scripts/quality_gates.py` (and release assembly
for release manifests). Dataframe field order/nullability is loaded from the
versioned dataframe schemas by ETL writers and validation. Do not duplicate those
structures in Python.
