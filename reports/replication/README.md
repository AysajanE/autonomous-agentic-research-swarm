# Replication package

Profile: **empirical**

Run `./MASTER.sh` to execute the package master command: `make reproduce-analysis && make paper`.

## Reproduction levels

- **Functional:** package construction and the declared master traversal are machine-audited.
- **Reproduced:** requires an observed clean-workspace master run whose regenerated outputs satisfy every declared verification bar. The live empirical package reports this level as pending a non-author run; identity is never inferred from an authoring run.

## Verification bars

- Byte identity: deterministic tables, `paper_values.json`, exhibits manifest, package manifest, and generated disclosure are compared by SHA-256/bytes.
- Content equivalence: SVG figures and their numeric sidecars are compared structurally/numerically because Matplotlib rendering bytes are not cross-platform deterministic.
- These bars are distinct and may not be substituted for one another.

## Declared and logged package versions

- `matplotlib`: declared `matplotlib>=3.8,<4`; exact runtime version unlogged.
- `pandas`: declared `pandas>=2.2,<3`; exact runtime version unlogged.

### Logged processed-manifest runtime environments

- No `environment` block is logged in the committed processed manifests; no exact runtime value is inferred.

## Data availability and access

- `data/raw_manifest/growthepie_2026-04-01.json` (growthepie, as of 2026-04-01): **raw_evidence_unavailable**. Access/reacquisition instruction recorded at ingest: `--as-of 2026-04-01 -- python src/etl/growthepie_fetch.py --run-date 2026-04-01`.
- `data/raw_manifest/growthepie_2026-04-08.json` (growthepie, as of 2026-04-08): **raw_evidence_unavailable**. Access/reacquisition instruction recorded at ingest: `--as-of 2026-04-08 -- python src/etl/growthepie_fetch.py --run-date 2026-04-08`.
- `data/raw_manifest/growthepie_2026-04-09.json` (growthepie, as of 2026-04-09): **raw_evidence_unavailable**. Access/reacquisition instruction recorded at ingest: `--as-of 2026-04-09 -- python src/etl/growthepie_fetch.py --run-date 2026-04-09`.
- `data/raw_manifest/l1_rent_2026-04-01.json` (l1_rent, as of 2026-04-01): **raw_evidence_unavailable**. Access/reacquisition instruction recorded at ingest: `python src/etl/build_l1_rent_panel.py --run-date 2026-04-01 --retries 2 --timeout-seconds 20.0 --blockscout-page-size 2000 --blobscan-page-size 1000`.
- `data/raw_manifest/l1_rent_2026-04-08.json` (l1_rent, as of 2026-04-08): **raw_evidence_unavailable**. Access/reacquisition instruction recorded at ingest: `python src/etl/build_l1_rent_panel.py --run-date 2026-04-08 --blockscout-page-size 250`.
- `data/raw_manifest/l1_rent_2026-04-09.json` (l1_rent, as of 2026-04-09): **raw_evidence_unavailable**. Access/reacquisition instruction recorded at ingest: `python src/etl/build_l1_rent_panel.py --run-date 2026-04-09 --resume-manifested-run`.

### AEA-style partial-reproducibility statement

The manifested raw evidence is unavailable and is not included or represented as recoverable. This package supports processed-data-to-results reproduction only. The release amendment named above is the truthful retention satisfier; users must reacquire source data using the recorded ingest commands and provider access conditions for any future raw-to-processed replay.

## Exhibit-to-source mapping

- `str_ecosystem_timeseries` → `reports/figures/str_ecosystem_timeseries.svg`; builder `src/analysis/build_str_release_outputs.py`; sources: `contracts/claims.yaml`, `data/processed/l1_rent/daily_l1_rent_decomposition.csv`, `data/processed/panels/daily_rollup_panel.csv`, `docs/protocol.md`, `reports/validation/cross_source_reconciliation.json`, `reports/validation/l1_rent_decomposition_validation.json`, `reports/validation/rollup_panel_validation.json`, `src/analysis/build_str_release_outputs.py`.
- `str_post_dencun_regimes` → `reports/figures/str_post_dencun_regimes.svg`; builder `src/analysis/build_str_release_outputs.py`; sources: `contracts/claims.yaml`, `data/processed/l1_rent/daily_l1_rent_decomposition.csv`, `data/processed/panels/daily_rollup_panel.csv`, `docs/protocol.md`, `reports/validation/cross_source_reconciliation.json`, `reports/validation/l1_rent_decomposition_validation.json`, `reports/validation/rollup_panel_validation.json`, `src/analysis/build_str_release_outputs.py`.
- `str_regime_summary` → `reports/tables/str_regime_summary.md`; builder `src/analysis/build_str_release_outputs.py`; sources: `contracts/claims.yaml`, `data/processed/l1_rent/daily_l1_rent_decomposition.csv`, `data/processed/panels/daily_rollup_panel.csv`, `docs/protocol.md`, `reports/validation/cross_source_reconciliation.json`, `reports/validation/l1_rent_decomposition_validation.json`, `reports/validation/rollup_panel_validation.json`, `src/analysis/build_str_release_outputs.py`.
- Catalog artifact roots: `{"figures":"reports/figures/","paper_build":"reports/paper/build/","processed_manifests":"data/processed_manifest/","raw_manifests":"data/raw_manifest/","tables":"reports/tables/","validation":"reports/validation/"}`.
