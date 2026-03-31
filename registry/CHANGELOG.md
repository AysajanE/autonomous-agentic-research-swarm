# Registry changelog

All changes to registry artifacts must be recorded here.

Format:
- `YYYY-MM-DD` — what changed, why, and expected impact.

- 2026-01-23 — Added `registry/rollup_registry_v1.csv` header stub to lock the rollup identifier interface early and reduce ad-hoc ID drift.
- 2026-03-31 — Populated `registry/rollup_registry_v1.csv` with 14 active v1-release rollups that satisfy the ETL-facing inclusion filter: growthepie metadata says the chain is a non-stale Ethereum-posting rollup with both `fees` and `rent_paid` coverage, and a specific L2BEAT project page exists for human review. Launch date, rollup type, DA mode, and metric coverage were cross-checked against growthepie `master.json`; `batcher_addresses_json` uses L2BEAT permission-role accounts when exposed and is intentionally left `[]` where L2BEAT only exposes submission contracts. Expected impact: unblocks T030/T035 normalization against explicit `rollup_id` values and provides usable L1 attribution hooks for most current rows while deliberately holding out ambiguous candidates such as Loopring, Ronin, and Zircuit.
