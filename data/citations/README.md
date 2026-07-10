# Citation verification snapshots

Network-enabled refresh tasks write normalized snapshots to
`data/citations/<YYYY-MM-DD>/<citekey>.json` and write their deterministic
evaluation date to `data/citations/AS_OF`. The quality gate never performs
network access; it reads only committed snapshots and the checked-in `AS_OF`.

Use `scripts/refresh_citations.py --offline-fixture <dir>` with pre-captured
fixture responses in this batch. Live retrieval is intentionally unavailable.
