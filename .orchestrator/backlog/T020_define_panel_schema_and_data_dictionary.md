---
task_id: T020
title: "Define minimal STR panel schema + data dictionary"
workstream: W0
role: Worker
priority: high
dependencies:
  - "T000"
  - "T010"
allowed_paths:
  - "contracts/schemas/panel_schema.yaml"
  - "contracts/schemas/panel_schema_str_v1.yaml"
  - "contracts/data_dictionary.md"
  - "contracts/decisions.md"
  - "contracts/CHANGELOG.md"
  - "docs/protocol.md"
disallowed_paths:
  - "src/"
  - "registry/"
  - "data/raw/"
outputs:
  - "contracts/schemas/panel_schema_str_v1.yaml"
  - "contracts/data_dictionary.md"
  - "contracts/decisions.md"
gates:
  - "make gate"
stop_conditions:
  - "Definition ambiguity"
  - "Need to reinterpret inclusion criteria"
---

# Task T020 — Define minimal STR panel schema + data dictionary

## Context

We need a minimal, explicit schema contract for the **analysis-ready daily rollup panel** used to compute the primary metric in `docs/protocol.md`:

- `STR_t = (Σ_i RentPaid_{i,t}) / (Σ_i L2Fees_{i,t})` (daily, ETH-native primary)

This task defines the canonical fields/units and documents them in the project data dictionary so downstream ETL/metrics/validation work doesn’t invent structure ad-hoc.

## Inputs

- `docs/protocol.md` (primary metric, units, source priority, regime dates, tolerances)
- `contracts/project.yaml` (mode)
- `docs/end_to_end_research_plan.md` (Phase 1–4 expectations for the panel)

## Outputs

- Update `contracts/schemas/panel_schema_str_v1.yaml` to include (at minimum) the fields required to compute STR from ETH-native series:
  - date (UTC day)
  - rollup_id (stable identifier; document the convention)
  - l2_fees_eth
  - rent_paid_eth
  - (optional but recommended for validation) profit_eth, txcount
  - (optional) USD counterparts (explicitly labeled) if planned for interpretation only
- Update `contracts/data_dictionary.md` with a concrete entry for the panel table: keys, grain, units, and field descriptions.
- Add an entry in `contracts/decisions.md` capturing:
  - the chosen field names and units,
  - the rollup identifier convention for `rollup_id`,
  - any explicit null/zero handling rules required for STR computation (if any).

## Success Criteria

- [ ] `contracts/schemas/panel_schema.yaml` points to the canonical versioned schema file(s)
- [ ] `contracts/schemas/panel_schema_str_v1.yaml` exists and defines the minimum daily rollup panel required for STR
- [ ] `contracts/data_dictionary.md` includes a filled-in table entry for the STR panel
- [ ] `contracts/decisions.md` records the schema decision with rationale
- [ ] `make gate` passes

## Status

- State: backlog
- Last updated: 2026-01-22

## Notes / Decisions

- 2026-01-22: Task created (Planner) to lock minimal schema before ETL/metrics work starts.
