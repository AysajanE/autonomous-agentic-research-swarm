# src/model/AGENTS.md — Modeling Rules

## Non-negotiables

- Implement exactly what is in `contracts/model_spec.*`.
- Do not invent missing assumptions; surface them and block if needed.
- Keep variants explicit (e.g., `model_v1`, `model_v2`) with clear differences.

## Outputs

- Must produce:
  - a callable solver/runner
  - a baseline run on benchmark instances
  - machine-readable results (JSON/CSV)
  - reproduction commands

## Validation

- Feasibility checks (constraints satisfied)
- Baseline replication (if defined)
- Sensitivity sanity checks (small perturbations behave as expected)
