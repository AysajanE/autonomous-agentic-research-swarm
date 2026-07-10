# `contracts/instances/`

Model-instance manifests for modeling and hybrid projects.

- Store executable v1 manifests as `contracts/instances/<instance_id>.json`.
- Synthetic manifests declare parameter ranges, seeds, generator command, git
  SHA, and content-bound outputs.
- Bridge manifests declare content-bound processed-manifest inputs, generation
  time, content-bound outputs, and generation-time empirical validation records.
- The single executable schema is
  `contracts/schemas/instance_manifest_v1.json`; modeling tasks consume these
  manifests and never `data/processed/**` directly.
