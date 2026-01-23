# `data/processed_manifest/` — tracked provenance for processed artifacts

Raw snapshots are append-only and kept out of git (`data/raw/`). Processed artifacts under
`data/processed/` are also kept out of git, but they still need reproducible provenance.

This directory holds **small, tracked** manifest JSON files that describe how to reproduce any
processed artifact(s).

## Naming convention

- `data/processed_manifest/<name>_<YYYY-MM-DD>.json`
  - Use the UTC “as-of” date for the processed dataset.

## Recommended fields

At minimum:
- `as_of_utc_date` (YYYY-MM-DD)
- `inputs` (list)
  - raw manifest path(s) under `data/raw_manifest/` and/or other processed manifests
- `transform`
  - `script_path` (repo-relative)
  - `git_sha` (commit hash)
  - `command` (string)
- `outputs` (list of `{path, sha256, bytes}`) relative to repo root

## Policy

- Manifests are append-only (new file per run-date); do not overwrite old manifests.
- If a processed dataset is regenerated, create a new manifest file with the new `git_sha`.
