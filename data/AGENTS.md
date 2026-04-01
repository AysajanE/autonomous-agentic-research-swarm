# data/AGENTS.md — Data Handling Rules

## Git policy

- Do not commit large data artifacts.
- Track schemas, manifests, and small samples only.
- Keep canonical ETL artifacts under `data/raw/` and `data/processed/` as durable local-only surfaces; they are gitignored by policy, not disposable scratch space.

## Append-only raw snapshots

- Raw snapshots are immutable and dated.
- Never overwrite; create a new dated snapshot.

## Provenance

Every raw snapshot must have a corresponding manifest entry:
- source name
- date fetched (UTC)
- command used
- file list + hashes (sha256)

## Review semantics

- ETL runtime and Judge review must verify the full local artifacts in `data/raw/` and `data/processed/`.
- Repo-wide review-complete evidence is the tracked bridge: manifests, small samples, run manifests, review logs, and code.

## Golden samples

Small, tracked sample datasets live in `data/samples/` and are used for fast tests and gates.
