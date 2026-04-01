# src/etl/AGENTS.md — ETL Rules

ETL is allowed to touch the network. Everything must be reproducible.

## Non-negotiables

- Every external fetch must be cached/snapshotted.
- Never overwrite snapshots; write dated folders/files.
- Record endpoint, parameters, and timestamp in a small manifest file.

## Outputs

- Raw snapshots (append-only, not committed): `data/raw/<source>/<YYYY-MM-DD>/...`
- Normalized outputs (rebuildable, not committed): `data/processed/<source>/...`
- Provenance manifests (tracked): `data/raw_manifest/<source>_<YYYY-MM-DD>.json`

## Review semantics

- During task execution and Judge review, ETL outputs must exist locally at the canonical `data/raw/` and `data/processed/` paths.
- After merge, repo-wide review integrity is carried by tracked code, manifests, tracked samples, and run/review logs rather than by committing full ETL datasets.

## Reliability

- Add retries with exponential backoff for APIs.
- Log failures with enough detail to replay.

## No hidden transforms

All transformation steps must be code, not manual edits.
