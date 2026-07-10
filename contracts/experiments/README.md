# `contracts/experiments/`

Experiment specifications (locked grids, seeds, solver, budgets, convergence
tolerance, and sweep survival criterion) for modeling projects.

Specs are JSON-compatible files validated against
`contracts/schemas/experiment_spec_v1.json`. Lock A must content-bind exactly
one active spec before experiment manifests may register computational claims.
