# Literature snapshots

Each dated directory is append-only and contains source payloads plus one or more
`manifest_<acquisition-id>.json` files. Manifests record the mini-PRISMA search
strategy, URL/DOI, retrieval date, and payload SHA-256. Use
`python scripts/swarm.py lit-review acquire --help`; tests must supply fixtures.
