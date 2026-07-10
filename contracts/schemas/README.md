# `contracts/schemas/`

Tracked schemas / data contracts used by the pipeline.

Placeholders you may add early:
- `panel_schema.yaml`
- `raw_<source>_schema.yaml`

Canonical schemas for this project:
- `panel_schema_str_v1.yaml` (minimum daily rollup STR panel)
- `panel_schema_decomp_v1.yaml` (daily Ethereum L1 rent decomposition)
- `instance_manifest_v1.json` (bridge-generated and synthetic variants)
- `experiment_spec_v1.json` (Lock A experiment-design surface)
- `experiment_manifest_v1.json` (solver-run registration surface)
- `referee_report_v1.json` (cross-family per-criterion/check findings surface)

The three M3a JSON Schemas above are loaded directly by
`scripts/quality_gates.py`; do not duplicate their required-field lists in
Python.
