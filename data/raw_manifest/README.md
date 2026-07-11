# `data/raw_manifest/` — tracked provenance

This directory holds **tracked** provenance manifests for raw snapshots that are kept out of git.

Recommended fields for each manifest:
- `source` (string)
- `fetched_at_utc` (ISO 8601)
- `command` (string)
- `parameters` (object)
- `files` (list of `{path, sha256, bytes}`) relative to repo root

Retention requirements:
- An absent or empty raw inventory must be covered by either a verified `file://` archive plus `archive_sha256`, a hash-bound in-repo `archive_receipt` for a remote archive, or an explicit `raw_evidence_unavailable` release amendment.
- A remote `https://`, `s3://`, or `gs://` URL is not resolvable evidence by syntax alone. Its receipt must record the same `archive_url` and `archive_sha256` plus non-empty retrieval metadata, and the manifest must bind the receipt as `{path, sha256}`.
- Live remote retrieval and cold-storage upload are out-of-band Operator/M5 operations. Offline gates verify the committed receipt, not the remote service.
- `command` (or `access_instruction`) must be nonblank; an empty inventory does not waive access documentation.

Naming convention:
- `data/raw_manifest/<source>_<YYYY-MM-DD>.json`

Helper:
- `python scripts/make_raw_manifest.py <source> <snapshot_dir> --as-of <YYYY-MM-DD> -- <command...>`
