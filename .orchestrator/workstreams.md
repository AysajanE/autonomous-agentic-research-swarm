# Workstreams (project coordination)

Workstreams define ownership boundaries so agents can work in parallel safely.

This is an initial, project-specific set. Refine as the pipeline grows.

## Workstreams table

| Workstream | Purpose | Owns paths | Does NOT own | Outputs (examples) | Gate(s) |
|---|---|---|---|---|---|
| W0 Protocol/Contracts | Canonical definitions, regimes, tolerances, specs | `docs/protocol.md`, `contracts/` | `src/`, `registry/`, `data/raw/` | Locked protocol + canonical contracts | `make gate` |
| W1 Data: off-chain | Ingest off-chain sources (APIs/dashboards) | `src/etl/`, `data/raw_manifest/` | `docs/protocol.md`, `registry/`, `src/analysis/` | Snapshot manifests; normalized tables (not committed) | `make gate` |
| W2 Data: on-chain | Extract Ethereum L1 data for costs/decomposition | `src/etl/`, `data/raw_manifest/` | `docs/protocol.md`, `registry/`, `src/analysis/` | Extraction scripts; manifests; processed tables (not committed) | `make gate` |
| W3 Registry | Attribution registries (rollups/addresses/labels) | `registry/` | `docs/protocol.md`, `contracts/`, `src/etl/` | Versioned registries + `registry/CHANGELOG.md` | `make gate` |
| W4 Metrics | Metric construction + unit tests | `src/analysis/metrics*`, `tests/` | `src/etl/`, `docs/protocol.md` | Metric module(s) + tests | `make gate` |
| W5 Validation | Reconciliation and sanity checks | `src/validation/`, `reports/validation/` | `src/etl/`, `docs/protocol.md` | Validation reports (MD/JSON) | `make gate` |
| W6 Analysis | Econometrics/figures built from processed data | `src/analysis/`, `reports/figures/`, `reports/tables/` | `src/etl/`, `docs/protocol.md`, `src/analysis/metrics*` | Figures + analysis scripts | `make gate` |
| W7 Writing | Narrative outputs (paper/deck/notes) | `docs/` (except `docs/protocol.md`), `reports/paper/`, `reports/deck/` | `docs/protocol.md`, `contracts/`, `src/` | Draft writeups and figure/table references | `make gate` |

## Ownership rules

- If a task needs to edit outside its workstream ownership, it must:
  1) add an `@human` note in the task file, and
  2) be re-scoped or split into multiple tasks with clear ownership.
