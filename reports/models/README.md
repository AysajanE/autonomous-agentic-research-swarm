# `reports/models/`

Model outputs (model runs, experiment outputs) for modeling/hybrid projects.

- Generate via code; do not hand-edit results.
- Register each solver run as a top-level `reports/models/<run_id>.json`
  experiment manifest; larger outputs may live in stable experiment subfolders.
- Store headline-claim sweep coverage at
  `reports/models/sweeps/<claim_id>.json`.
