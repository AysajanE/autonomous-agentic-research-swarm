---
schema_version: research_swarm.prereg_lock.v1
phase: lock_a
status: draft
locked_at_utc: null
locked_sha256: null
locked_by: null
lock_version: 0
---

# Lock A — parametric model and experiment design

Before locking, replace each `pending` digest with the SHA-256 of the exact
referenced file. The locked experiment grid, seeds, solver, budgets,
convergence tolerance, sweep survival criterion, and counterfactual functional
forms must remain parametric over the to-be-estimated instance parameters.

- path: contracts/model_spec.md
  sha256: pending
- path: contracts/experiments/exp_001.yaml
  sha256: pending

## Counterfactual definitions

- Declare functions of estimated parameters before activating this lock.
