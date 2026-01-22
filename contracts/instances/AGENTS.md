# contracts/instances/AGENTS.md — Instance Discipline

Instances are part of the contract surface for modeling projects.

## Rules

- Do not modify instances unless your task explicitly authorizes it.
- Version instance sets (e.g., `instance_set_v1/`) and treat them as immutable once used in results.
- Every instance set must have a manifest (inputs, hashes, generator command, timestamp).
