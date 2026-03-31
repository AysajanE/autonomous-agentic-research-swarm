# Workstreams (v1 coordination map)

Workstreams define path ownership so parallel agents can work safely. The current repo is battle-testing the empirical path, while modeling and hybrid surfaces remain contract-ready.

| Workstream | Purpose | Owns paths | Does NOT own | Example outputs | Network | `integration_ready` eligible |
|---|---|---|---|---|---|---|
| W0 Protocol/Contracts | Lock protocol, project contracts, framework policy, and interface definitions | `docs/protocol.md`, `contracts/` | `src/`, `registry/`, `reports/paper/build/` | protocol changes, contract revisions, decision log entries | no | yes |
| W1 Data: off-chain | Pull off-chain rollup metrics and vendor cross-check series | `src/etl/`, `data/raw/growthepie/`, `data/raw_manifest/growthepie_*.json`, `data/processed/growthepie/`, `data/processed_manifest/vendor_daily_rollup_panel_*.json`, `data/samples/growthepie/` | protocol, registry, analysis outputs, paper/release surfaces | growthepie raw manifests, vendor panel, tracked sample | yes | no |
| W2 Data: on-chain | Build the authoritative L1 rent path and canonical panel | `src/etl/`, `data/raw/l1_rent/`, `data/raw_manifest/l1_rent_*.json`, `data/processed/l1_rent/`, `data/processed/panels/`, `data/processed_manifest/daily_*`, `data/samples/l1_rent/`, `data/samples/panels/` | protocol, registry definitions, figures/tables, paper/release surfaces | L1 rent decomposition, canonical rollup panel, samples | yes | no |
| W3 Registry | Maintain the evidence-backed rollup universe | `registry/` | protocol, contracts, `src/`, figures/paper | populated `rollup_registry_v1.csv`, registry changelog | yes | yes |
| W4 Metrics | Lock reusable metric math and sample-only tests | `src/analysis/metrics*`, `tests/test_metrics*` | ETL, protocol, registry, release figures/tables | `metrics_str.py`, unit tests | no | no |
| W5 Validation | Reconcile canonical artifacts before analysis or writing | `src/validation/`, `reports/validation/` | ETL acquisition, protocol, paper build, catalog | validation JSON/MD bundle | no | no |
| W6 Analysis | Generate release figures and tables from validated artifacts only | `src/analysis/` except `metrics*`, `reports/figures/`, `reports/tables/` | ETL acquisition, protocol, registry, paper build, catalog | release figures, release tables | no | no |
| W7 Writing | Maintain Quarto manuscript source and narrative surfaces | `reports/paper/_quarto.yml`, `reports/paper/index.qmd`, `reports/paper/references.bib`, `reports/deck/` | protocol, contracts, raw/processed data, paper build outputs | manuscript source, bibliography | no | no |
| W8 Modeling / Bridge | Build explicit empirical-to-model interfaces and modeling runs | `src/model/`, `contracts/instances/`, `contracts/experiments/`, `reports/models/` | protocol, registry, empirical release catalog | instance manifests, model run manifests, modeling outputs | no | yes |
| W9 Ops / Release | Runtime stewardship, catalog compilation, final paper build, and release assembly | `scripts/release_assembly.py`, `reports/catalog.yaml`, `reports/paper/build/`, `reports/status/`, `data/tmp/swarm_logs/` | protocol, contracts, registry content, core ETL/metrics logic | compiled catalog, paper build, release manifest | no | yes |

## Ownership rules

- Keep `allowed_paths` narrow and task-specific; workstreams are the outer boundary, not permission to edit everything in a directory.
- Only workstreams allowlisted in `contracts/framework.json` may set `allow_network: true`. v1 allowlists W1, W2, and W3.
- W6 and W7 tasks may not advance using unvalidated empirical artifacts.
- W9 tasks are Operator-owned. Workers do not own `reports/catalog.yaml`, `reports/paper/build/`, or release manifests.
- If a task needs edits across workstreams, split the work or block with `@human` rather than widening scope silently.
