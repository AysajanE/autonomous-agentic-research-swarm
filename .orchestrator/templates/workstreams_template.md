# Workstreams (template)

Workstreams are how we prevent agent clashes: each workstream has **explicit ownership** of paths and artifacts.

## Workstreams table

| Workstream | Purpose | Owns paths | Does NOT own | Outputs (examples) | Gate(s) |
|---|---|---|---|---|---|
| W0 Protocol/Contracts | Canonical definitions + contracts | `docs/protocol.md`, `contracts/` | `src/`, `registry/` | locked protocol + schemas | `make gate` |
| W1 Data: off-chain | External sources ETL | `src/etl/`, `data/raw/`, `data/raw_manifest/` | `docs/protocol.md`, `src/analysis/` | snapshots + manifests | `make gate` |
| W2 Data: on-chain | Ethereum L1 extraction | `src/etl/`, `data/raw/`, `data/raw_manifest/` | `docs/protocol.md`, `src/analysis/` | snapshots + manifests | `make gate` |
| W3 Registry | Attribution mapping | `registry/` | `docs/protocol.md`, `src/etl/` | registry CSV/JSON | `make gate` |
| W4 Metrics | Metric construction + tests | `src/analysis/metrics*`, `tests/` | `src/etl/` | metric modules + tests | `make gate` |
| W5 Validation | Reconciliation + tolerances | `src/validation/`, `reports/validation/` | `src/etl/` | validation reports | `make gate` |
| W6 Analysis | Figures + econometrics | `src/analysis/`, `reports/figures/` | `src/etl/`, `src/analysis/metrics*` | figures/tables | `make gate` |
| W7 Writing | Paper/deck/notes | `docs/` (except protocol), `reports/paper/` | `docs/protocol.md`, `src/` | narrative drafts | `make gate` |

## Notes

- Keep “Owns paths” strict; propose cross-cutting changes via PR and explicit review.
- Prefer parallelism only when interfaces (schemas, function signatures, output paths) are locked.
