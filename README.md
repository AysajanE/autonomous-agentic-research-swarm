# Research Swarm Template (membership-friendly)

This template is a lightweight, *file-based* orchestration system for running multiple coding/research agents in parallel (e.g., Claude Code + OpenAI Codex), while keeping coordination sane.

## Core idea
- **Planner** writes/updates task specs in `.orchestrator/`.
- **Workers** execute *one task each* in isolated git worktrees/branches.
- **Judge** runs quality gates + reviews diffs, then either requests fixes or approves merge.

## Why this structure
- A shared “Kanban file” where agents self-coordinate is fragile at scale.
- Git branches/worktrees + task specs + automated gates are the simplest robust coordination mechanism.

## Directory layout
- `.orchestrator/` — task queue + state (single source of truth)
- `docs/` — protocol, definitions, data dictionary
- `contracts/` — canonical specs (schemas/model spec/assumptions/decisions)
- `src/` — ETL, validation, analysis
- `data/` — raw/processed + tracked provenance manifests (keep large artifacts out of git)

## Minimal operating procedure
1. Fill project contracts (Phase 0):
   - Empirical/hybrid: `docs/protocol.md` + `contracts/schemas/panel_schema_str_v1.yaml`
   - Modeling/hybrid: `contracts/model_spec.*` + `contracts/instances/benchmark_small/`
2. Planner creates tasks in `.orchestrator/backlog/` with success criteria.
3. Start multiple Workers (tmux panes) and assign each a task file.
4. Worker opens PR (or commits to its branch).
5. Judge runs `make gate` and reviews diff vs success criteria.
6. Merge + repeat.

## Runbook

See `docs/runbook_swarm.md`.
Automation (tmux supervisor loop): `docs/runbook_swarm_automation.md`.

## Vertical slice quickstart (STR)

Once tasks `T030` → `T060` are implemented/merged and the golden sample exists at
`data/samples/growthepie/vendor_daily_rollup_panel_sample.csv`, you should be able to generate the first STR artifact via:

```bash
make gate
make test
python src/validation/validate_vendor_panel.py --sample
python src/analysis/plot_str_timeseries_sample.py
```

Expected outputs:
- `reports/validation/vendor_panel_validation.md`
- `reports/validation/vendor_panel_validation.json`
- `reports/figures/str_timeseries_sample.svg`

## Safety defaults
- Don’t run “auto-approve everything” on your laptop.
- Prefer a sandboxed environment (Codespaces/VM/devcontainer).
- Restrict file access and dangerous shell commands where possible.

## Practical notes

- Run your agent CLI from the correct worktree directory so nested `AGENTS.md` rules are applied.
- Hosted environments may stop after inactivity by default; verify timeouts before relying on overnight runs.
