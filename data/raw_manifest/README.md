# `data/raw_manifest/` — tracked provenance

This directory holds **tracked** provenance manifests for raw snapshots that are kept out of git.

Recommended fields for each manifest:
- `source` (string)
- `fetched_at_utc` (ISO 8601)
- `command` (string)
- `parameters` (object)
- `files` (list of `{path, sha256, bytes}`) relative to repo root

Naming convention:
- `data/raw_manifest/<source>_<YYYY-MM-DD>.json`

Helper:
- `python scripts/make_raw_manifest.py <source> <snapshot_dir> --as-of <YYYY-MM-DD> -- <command...>`
