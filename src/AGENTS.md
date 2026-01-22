# src/AGENTS.md — Codebase Rules

`src/` contains all project code. Keep modules separated and reproducible.

## Module boundaries

- `src/etl/`: data acquisition and transformation (network allowed)
- `src/validation/`: deterministic checks and reconciliation (no network)
- `src/analysis/`: analysis and figures (no network)
- `src/model/`: modeling/simulation/solvers (no invented assumptions; follow contracts)

## General standards

- Prefer small, testable modules.
- No network calls outside `src/etl/`.
- Do not change contracts silently; update `contracts/` first if interfaces change.
