# contracts/AGENTS.md — Contract Discipline

This directory contains canonical specs. Downstream work must not reinterpret them.

## Change policy

- Only tasks in the Protocol/Contracts workstream may modify contracts.
- All contract changes require:
  - rationale
  - expected downstream impact
  - version bump if interfaces change

## No implicit changes

If you need a new field/variable/assumption:
- update the contract first
- then update downstream code/tasks

## Stop condition

If a contract is ambiguous: block with @human and propose the smallest clarification.
